"""Compare a .env file with .env.example (and optionally with the env vars the code reads).

    python check_env.py [path/to/.env] [--code]

Never prints values of secret-looking names (KEY / DSN / PASS / LOGIN / SECRET / TOKEN);
for those it only says whether they are set. Exit code 1 if there is drift.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SECRET = re.compile(r"KEY|DSN|PASS|LOGIN|SECRET|TOKEN")
_LINE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")
_COMMENTED = re.compile(r"^\s*#\s*([A-Z][A-Z0-9_]*)\s*=")
_USE = re.compile(r"""(?:environ\.get|getenv|environ\.setdefault|environ\.pop)\(\s*["']([A-Z][A-Z0-9_]+)["']|environ\[\s*["']([A-Z][A-Z0-9_]+)["']""")
_SKIP_DIRS = {"impovements", "tests", "node_modules", "__pycache__", "data"}


def parse(path: Path):
    active: dict[str, str] = {}
    optional: set[str] = set()
    dups: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _LINE.match(line)
        if m:
            key, value = m.group(1), re.sub(r"\s+#.*$", "", m.group(2)).strip().strip('"').strip("'")
            if key in active:
                dups.append(key)
            active[key] = value
            continue
        m = _COMMENTED.match(line)
        if m:
            optional.add(m.group(1))
    return active, optional, dups


def code_env_names() -> dict[str, str]:
    found: dict[str, str] = {}
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).parts
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel):
            continue
        for m in _USE.finditer(p.read_text(encoding="utf-8", errors="ignore")):
            found.setdefault(m.group(1) or m.group(2), "/".join(rel))
    return found


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    env_path = Path(args[0]) if args else ROOT / ".env"
    example, example_optional, _ = parse(ROOT / ".env.example")
    env, env_optional, dups = parse(env_path)
    known = set(example) | example_optional
    bad = False

    print(f"{env_path}: {len(env)} active keys | .env.example: {len(example)} active, {len(example_optional)} optional\n")
    missing = sorted(k for k in example if k not in env)
    extra = sorted(k for k in env if k not in known)
    if missing:
        bad = True; print("MISSING in env (present in .env.example):"); [print("  ", k) for k in missing]
    if extra:
        bad = True; print("EXTRA in env (not in .env.example: add it there or remove it):"); [print("  ", k) for k in extra]
    if dups:
        bad = True; print("DUPLICATE keys in env (the last one wins):", ", ".join(sorted(set(dups))))
    empty = sorted(k for k in env if k in example and SECRET.search(k) and not env[k])
    if empty:
        print("Secrets present but EMPTY in env:", ", ".join(empty))
    diffs = [(k, env[k], example[k]) for k in env if k in example and not SECRET.search(k) and env[k] != example[k]]
    if diffs:
        print("\nNon-secret values that differ from .env.example (env | example):")
        for k, a, b in diffs:
            print(f"   {k}: {a!r} | {b!r}")
    if "--code" in sys.argv:
        undeclared = {k: f for k, f in code_env_names().items() if k not in known}
        if undeclared:
            bad = True; print("\nUsed in code but NOT in .env.example:"); [print(f"   {k}  ({f})") for k, f in sorted(undeclared.items())]
        else:
            print("\nAll env vars read by the code are declared in .env.example.")
    print("\nOK: no drift." if not bad else "\nDRIFT found.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
