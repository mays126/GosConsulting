import os
from contextlib import asynccontextmanager

from fastapi import HTTPException, FastAPI
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from datetime import timedelta
from fastapi.responses import JSONResponse
from models.schemas import LLMQuestion, HistoryItem
from models.core import Request, init_models
from utils import rag_pipeline

config = None
pipeline = None

@asynccontextmanager
async def lifespan(application: FastAPI):
    init_models()
    try:
        config = rag_pipeline.load_config()
    except Exception as e:
        print(f"Критическая ошибка: не удалось загрузить конфигурацию: {e}. Завершение работы.")
        return
    logger = rag_pipeline.setup_logging(config.log_level)

    if not config.google_api_key or "YOUR_GOOGLE_API_KEY_HERE" in config.google_api_key:
        logger.critical("GOOGLE_API_KEY не установлен или используется значение по умолчанию. Укажите валидный ключ.")
        return
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    logger.info("--- Запуск RAG-пайплайна (улучшенный поиск) ---")
    global pipeline
    pipeline = rag_pipeline.RAGPipeline(config, logger)

    logger.info("Первоначальная инициализация/индексация базы знаний...")
    # Используем простой запрос для инициализации, не обязательно связанный с тестовыми данными напрямую
    init_query = "Общие принципы законодательства"
    init_result = await pipeline.run_pipeline(init_query, force_reindex=True)
    logger.info(f"Результат инициализационного запроса для '{init_query}' (начало): {init_result[:200]}...")

    if pipeline.vectorstore is None:
        logger.critical(
            "Не удалось инициализировать RAG пайплайн и его базу знаний. Проверьте логи Qdrant и настройки.")
        return

    yield
    print("Shutdown")

async def send_question(user_id, question: LLMQuestion, db: Session):
    answer = await pipeline.run_pipeline(question.question, force_reindex=False)
    answer.replace("\n","")
    if answer:
        llm_answer = Request(
            question = question.question,
            answer = answer,
            user_id = user_id
        )

        db.add(llm_answer)
        db.commit()
    else:
        raise HTTPException(status_code=501, detail="Smth went wrong with LLM")
    return answer

async def get_requests_history(user_id, session: Session):
    query = select(Request).where(Request.user_id == user_id).order_by(Request.id.desc()).limit(10)
    result = session.execute(query)
    history = result.scalars().all()

    final_result = []
    for request in history:
        final_result.append(HistoryItem(
            id=request.id,
            question = request.question,
            answer = request.answer.strip()
        ))

    return final_result
