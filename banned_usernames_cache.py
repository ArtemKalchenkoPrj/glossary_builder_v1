"""In-memory cache of banned Telegram usernames with live PG LISTEN/NOTIFY updates.

Lifecycle
---------
1. load_from_file()   — синхронно, заполняет set из ignored_usernames.txt (fallback)
2. load_from_db()     — async, SELECT из banned_usernames, мерджит поверх файла
3. start_listener()   — async infinite loop: LISTEN → keepalive → reconnect on failure

Порядок в lifespan webhook-а
-----------------------------
    cache = BannedUsernamesCache()
    cache.load_from_file(path)                     # sync, до await-ов
    async with pool.acquire() as conn:
        await cache.load_from_db(conn)
    task = asyncio.create_task(cache.start_listener(dsn))

Проверка в горячем пути
------------------------
    if cache.is_banned(username):
        return

NOTIFY payload
--------------
    INSERT → триггер шлёт pg_notify('banned_usernames_changed', NEW.username)
             кэш точечно добавляет username
    DELETE → триггер шлёт pg_notify('banned_usernames_changed', 'RELOAD')
             кэш перезапрашивает всю таблицу из БД
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

_CHANNEL = "banned_usernames_changed"
_RELOAD_SIGNAL   = "RELOAD"
_KEEPALIVE_INTERVAL = 30   # секунд между SELECT 1 пингами
_RECONNECT_BASE     = 1    # начальная задержка реконнекта (сек)
_RECONNECT_MAX      = 60   # максимальная задержка реконнекта (сек)


class BannedUsernamesCache:
    """Thread-safe (asyncio) in-memory set забаненных username-ов."""

    def __init__(self) -> None:
        self._usernames: set[str] = set()
        self._dsn: Optional[str] = None   # сохраняется в start_listener для reload

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def is_banned(self, username: Optional[str]) -> bool:
        """O(1) проверка. Нет I/O, нет await — вызывается на каждом сообщении."""
        if not username:
            return False
        return username in self._usernames

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def load_from_file(self, path: Path) -> None:
        """Синхронная загрузка из ignored_usernames.txt (fallback).

        Формат: одно имя на строку, строки с # — комментарии.
        Вызывать ДО старта event loop (или просто в начале lifespan до первого await).
        """
        if not path.exists():
            logger.info("[banned_cache] fallback file not found, skipping: %s", path)
            return

        loaded: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                loaded.add(name)

        self._usernames.update(loaded)
        logger.info(
            "[banned_cache] loaded %d usernames from file %s (total in cache: %d)",
            len(loaded), path, len(self._usernames),
        )

    async def load_from_db(self, conn: asyncpg.Connection) -> None:
        """SELECT username FROM banned_usernames — мерджит поверх уже загруженного из файла."""
        try:
            rows = await conn.fetch("SELECT username FROM banned_usernames")
        except asyncpg.UndefinedTableError:
            logger.warning(
                "[banned_cache] table banned_usernames not found — "
                "run migration first. Skipping DB load."
            )
            return

        db_names = {row["username"] for row in rows}
        self._usernames.update(db_names)
        logger.info(
            "[banned_cache] loaded %d usernames from DB (total in cache: %d)",
            len(db_names), len(self._usernames),
        )

    # ------------------------------------------------------------------
    # LISTEN / NOTIFY loop
    # ------------------------------------------------------------------

    async def start_listener(self, dsn: str) -> None:
        """Бесконечный цикл: коннект → LISTEN → keepalive → reconnect при падении.

        Запускать через asyncio.create_task(), держать task в _AppState
        и отменять при shutdown через task.cancel().
        """
        self._dsn = dsn   # сохраняем для _reload_from_db
        delay = _RECONNECT_BASE

        while True:
            conn: Optional[asyncpg.Connection] = None
            try:
                conn = await asyncpg.connect(dsn)

                # 1. Сначала регистрируем listener — PG начнёт буферить NOTIFY
                await conn.add_listener(_CHANNEL, self._on_notify)

                # 2. Потом читаем актуальный список из БД (не пропустим то,
                #    что добавили в щель между load_from_db и LISTEN)
                await self.load_from_db(conn)

                logger.info("[banned_cache] LISTEN on channel %r — ready", _CHANNEL)
                delay = _RECONNECT_BASE  # сброс backoff после успешного коннекта

                # 3. Keepalive loop — не даём соединению умереть от idle timeout
                while True:
                    await asyncio.sleep(_KEEPALIVE_INTERVAL)
                    await conn.fetchval("SELECT 1")

            except asyncio.CancelledError:
                # Нормальный shutdown — выходим тихо
                logger.info("[banned_cache] listener cancelled, shutting down")
                return

            except Exception as exc:
                logger.warning(
                    "[banned_cache] listener connection lost (%s), "
                    "reconnecting in %ds...",
                    exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RECONNECT_MAX)

            finally:
                if conn and not conn.is_closed():
                    try:
                        await conn.remove_listener(_CHANNEL, self._on_notify)
                        await conn.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Перезагрузка из БД (вызывается при получении RELOAD сигнала)
    # ------------------------------------------------------------------

    async def _reload_from_db(self) -> None:
        """Полная перезагрузка кэша из banned_usernames.

        Открывает отдельное соединение, заменяет _usernames целиком
        (а не мерджит), чтобы удалённые юзеры пропали из кэша.
        """
        if not self._dsn:
            logger.error("[banned_cache] _reload_from_db called before start_listener, skipping")
            return

        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                rows = await conn.fetch("SELECT username FROM banned_usernames")
                self._usernames = {row["username"] for row in rows}
                logger.info(
                    "[banned_cache] loaded %d usernames from DB (total in cache: %d)",
                    len(self._usernames),
                )
            finally:
                await conn.close()
        except Exception as exc:
            logger.error("[banned_cache] reload failed: %s", exc)

    # ------------------------------------------------------------------
    # Callback (вызывается asyncpg синхронно)
    # ------------------------------------------------------------------

    def _on_notify(
        self,
        conn: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        """Вызывается asyncpg при получении NOTIFY.

        Payload = username  → точечное добавление в кэш
        Payload = 'RELOAD'  → полная перезагрузка из БД (при DELETE)
        """
        username = payload.strip() if payload else ""
        if not username:
            logger.warning("[banned_cache] received empty NOTIFY payload, ignoring")
            return

        if username == _RELOAD_SIGNAL:
            logger.info("[banned_cache] NOTIFY received — DELETE signal, refreshing from DB")
            asyncio.create_task(self._reload_from_db())
            return

        self._usernames.add(username)
        logger.info(
            "[banned_cache] NOTIFY received — added %r (cache size: %d)",
            username, len(self._usernames),
        )