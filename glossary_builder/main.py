from glossary_builder.lead_extraction import LeadExtractor, LeadExtractionConfig, load_glossary
from glossary_builder.llm import LLMClient
from glossary_builder.loader import Message
from glossary_builder.single_classifier import classify_single_message
from datetime import datetime
from glossary_builder.lead_prompt_versions import BY_VERSION
from pathlib import Path

def create_classifier():
    project_root = Path(__file__).parent.parent

    glossary = load_glossary(project_root / "data/output/glossary_primary_sense_only.json")
    llm = LLMClient()

    cfg = LeadExtractionConfig()
    cfg.lead_definition = BY_VERSION["v5-short"]
    cfg.pre_filter_db_path = str(project_root / "data/output/glossary.db")

    return glossary, llm, cfg


if __name__ == "__main__":
    glossary, llm, cfg = create_classifier()

    # Контекст — список Message объектов
    context = [
        Message(
            message_id=1001,
            date=datetime(2026, 5, 20, 21, 38, 0),
            text="Хто може запропонувати PSP для iGaming на KZ?",
            username="some_user",
            group_id=None, user_id=None, first_name=None, last_name=None,
        ),
    ]

    # Запуск c контекстом
    result = classify_single_message(
        text="Я хочу пиццы 4 сыра и картошку фри с кокаколлой",
        glossary=glossary,
        llm=llm,
        message_id=1002,
        username="other_user",
        timestamp=datetime(2026, 5, 20, 21, 40, 0),
        #context=context, # если мы передаем контекст. типо несколко сообщений до/после. не обязательный параметр
        cfg=cfg
    )

    print(result)
