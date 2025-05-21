import os
import yaml
import logging
import asyncio
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# LangChain imports
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredODTLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field as PydanticField
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Qdrant client
from qdrant_client import QdrantClient, models

# Progress bar
from tqdm.asyncio import tqdm

# Retry mechanism
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# --- Configuration ---
DEFAULT_CONFIG_PATH = "utils/config.yaml"

# --- Константы для метаданных ---
META_SOURCE_FILE_NAME = "source_file_name"
META_DOCUMENT_TITLE = "document_title"
META_PUBLICATION_DATE = "publication_date"
META_DOCUMENT_TYPE = "document_type"
META_LAW_ID = "law_id"
META_ARTICLE_NUMBER = "article_number"
META_SOURCE_URL = "source_url"


class AppConfig:
    def __init__(self, config_data: Dict[str, Any]):
        self.data_directory: str = config_data.get("data_directory", "rag_data_legislative_v1")
        self.chunk_size: int = config_data.get("chunk_size", 1500)
        self.chunk_overlap: int = config_data.get("chunk_overlap", 300)

        self.qdrant_location: str = config_data.get("qdrant_location", ":memory:")  # ":memory:" или URL
        self.qdrant_collection_name: str = config_data.get("qdrant_collection_name", "legislative_documents_rb_v2")
        self.qdrant_prefer_grpc: bool = config_data.get("qdrant_prefer_grpc",
                                                        True) if self.qdrant_location != ":memory:" else False
        self.qdrant_api_key: Optional[str] = config_data.get("qdrant_api_key", os.getenv("QDRANT_API_KEY"))
        self.vector_size: int = config_data.get("vector_size", 768)

        self.google_api_key: Optional[str] = config_data.get("google_api_key", os.getenv("GOOGLE_API_KEY"))
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY не найден в config.yaml или переменных окружения.")

        self.embedding_model_name: str = config_data.get("embedding_model_name", "models/embedding-004")
        self.chat_model_name: str = config_data.get("chat_model_name", "gemini-1.5-flash-latest")
        self.llm_temperature: float = config_data.get("llm_temperature", 0.05)

        self.search_k: int = config_data.get("search_k",
                                             5)  # Сколько документов В ИТОГЕ ПОПАДЕТ в LLM-ранжировщик или генератор ответа
        self.search_type: str = config_data.get("search_type", "mmr")
        # fetch_k: сколько документов извлекать из базы для MMR или MultiQuery ДО финального отбора k
        self.search_fetch_k: int = config_data.get("search_fetch_k", 20)
        self.relevance_threshold: float = config_data.get("relevance_threshold", 0.6)

        self.log_level: str = config_data.get("log_level", "INFO").upper()
        self.detailed_debug_logging: bool = config_data.get("detailed_debug_logging",
                                                            False)  # Для вывода контента чанков в лог

        self.enable_query_regeneration: bool = config_data.get("enable_query_regeneration", True)
        self.enable_llm_ranking: bool = config_data.get("enable_llm_ranking", True)
        # --- Новая опция для MultiQueryRetriever ---
        self.enable_multiquery_retriever: bool = config_data.get("enable_multiquery_retriever",
                                                                 True)  # Включим по умолчанию для демонстрации

        # Промпты (ваши версии из предыдущего сообщения)
        self.regeneration_prompt_template_str: str = config_data.get(
            "regeneration_prompt_template",
            """Ты — ассистент, помогающий сформулировать запрос для поиска в базе законодательных актов.
Перефразируй следующий вопрос пользователя так, чтобы он был максимально точным и охватывал ключевые термины и правовые концепции, относящиеся к сути вопроса в контексте законодательства Республики Беларусь.
Сохрани исходный смысл и язык (русский). Если вопрос касается конкретного аспекта законодательства или типа правоотношений, постарайся это отразить в перефразировке.
Так же при наличии в запросе конкретных предметов дай несколько синонимов по его характеристикам в перефразированный запрос (например по запросу "Украл ручку" ты даешь: мелкое хищение, хищение на малую стоимость и т.д).
Предоставь краткое объяснение, почему ты так перефразировал, указав, какие аспекты и термины ты уточнил или добавил для улучшения поиска.
{format_instructions}
Исходный вопрос: {question}"""
        )
        self.ranking_prompt_template_str: str = config_data.get(
            "ranking_prompt_template",
            """Ты — ассистент, оценивающий релевантность фрагментов законодательных документов по отношению к вопросу пользователя.
Оцени релевантность каждого из следующих фрагментов нормативно-правовых актов Республики Беларусь по отношению к вопросу пользователя.
Документы пронумерованы и содержат метаданные (название документа, возможно, статья, дата).
Присвой каждому документу оценку релевантности от 0.0 (совершенно нерелевантен) до 1.0 (полностью и непосредственно относится к сути вопроса или описывает применимую правовую норму).
Учитывай не только прямое упоминание ключевых слов, но и смысловое соответствие *правовой концепции* или *типу правоотношений*, описанных в документе, запросу.
Не старайся искать точных предметных совпадений, если таковые присутствуют в запросе, оценивай общие характеристики предмета и ищи совпадения по ним и давай оценку соответственно совпадениям по характеристикам предмета.
Предоставь краткое объяснение для каждой оценки, ссылаясь на содержание документа, метаданные и правовую суть вопроса.
{format_instructions}
Вопрос: {question}
Документы:
{documents}"""
        )
        self.qa_with_sources_prompt_template_str: str = config_data.get(
            "qa_with_sources_prompt_template",
            """Ты — официальный ИИ-ассистент, предоставляющий консультации по законодательству Республики Беларусь.
Твоя задача — точно и формально ответить на вопрос пользователя, основываясь на предоставленных ниже выдержках из нормативно-правовых актов (фрагменты документов).
В том случае, если в предоставленных фрагментах не найдено ничего конкретного по вопросу, о то ты должен вопервых - не использовать в ответе "я не нашел", "не знаю" и их производные, во вторых ответить основываясь на собственных знаниях и базовых понятиях.
Так же в случае, если конкретное наказание не найдено, то ты не должен отвечать "я не нашел конкретного наказания" и т.д, ты должен отвечать исходя из своих суждений по этому поводу и базовых принципов.
Отвечай на русском языке. Не используй никаких специальных символов которые поддерживает только python, твои ответы будут выведены на сайт без пост-обработки.
Так же не используй слова "предоставленные документы" и т.д, ты должен создавать ощущение того, что ты знал их заранее.
Проанализируй предоставленные фрагменты документов. Если они содержат законодательные нормы, статьи или положения, применимые к *типу* ситуации или правовому вопросу, описанному пользователем, изложи суть этих норм, ссылаясь на соответствующие статьи или документы, если они указаны в метаданных.
Если предоставленные фрагменты описывают применимые правовые нормы, но не содержат точного ответа на *конкретный пример* пользователя (например, общая статья о краже вместо примера с конфетой), объясни общую норму и укажи источник.
Если предоставленные фрагменты вообще не содержат информации по запросу, то отвечай основываясь на собсвтенных знаниях о законадательстве РБ, но не указывай конкретных наказаний (если не уверен, что знаешь конкретную статью), а только общие черты наказаний.
Не пробуй искать данные о конкретных предметах или типах предметов, которые указал пользователь в запросе, если конкретно таковых нету, оценивай общие характеристики предмета (примерную стоимость, государственную ценность, примерные размеры и т.д) и на основании этого ищи ответы.
В конце ответа всегда приводи точные "ИСТОЧНИКИ:", перечисляя для каждого использованного фрагмента:
- Название документа (из метаданных '{META_DOCUMENT_TITLE}' или '{META_SOURCE_FILE_NAME}').
- Номер документа, если известен (из метаданных '{META_LAW_ID}').
- Дату принятия/публикации, если известна (из метаданных '{META_PUBLICATION_DATE}').
- Номер статьи, если он релевантен для данного ответа и доступен в метаданных чанка (из метаданных '{META_ARTICLE_NUMBER}').
Если для ответа использовалось несколько фрагментов из одного документа, перечисли их все с указанием статьи, если применимо. Если источников несколько, перечисли каждый отдельно.

Вопрос: {question}
=========
ПРЕДОСТАВЛЕННЫЕ ДОКУМЕНТЫ:
{summaries}
=========
Ответ на русском:""".replace("{META_DOCUMENT_TITLE}", META_DOCUMENT_TITLE)
            .replace("{META_SOURCE_FILE_NAME}", META_SOURCE_FILE_NAME)
            .replace("{META_LAW_ID}", META_LAW_ID)
            .replace("{META_PUBLICATION_DATE}", META_PUBLICATION_DATE)
            .replace("{META_ARTICLE_NUMBER}", META_ARTICLE_NUMBER)
        )


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return AppConfig(config_data)
    except FileNotFoundError:
        default_conf_data = {
            "data_directory": "rag_data_legislative_v1",
            "qdrant_location": "http://localhost:6333",
            "qdrant_collection_name": "legislative_documents_rb_v2",
            "google_api_key": "YOUR_GOOGLE_API_KEY_HERE",
            "enable_multiquery_retriever": True,  # По умолчанию включим для демонстрации
            "detailed_debug_logging": False,
            # Остальные параметры возьмутся из AppConfig defaults
        }
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_conf_data, f, allow_unicode=True, sort_keys=False)
            logging.warning(
                f"Конфигурационный файл '{config_path}' не найден. Создан файл с настройками по умолчанию. Пожалуйста, проверьте и обновите его, особенно GOOGLE_API_KEY и qdrant_location.")
        except Exception as e:
            logging.error(f"Ошибка при создании дефолтного конфигурационного файла '{config_path}': {e}")
            logging.warning("Продолжение работы с настройками по умолчанию (Qdrant :memory:).")
            default_conf_data["qdrant_location"] = ":memory:"
        return AppConfig(default_conf_data)
    except yaml.YAMLError as e:
        logging.error(f"Ошибка при чтении конфигурационного файла '{config_path}': {e}")
        raise
    except ValueError as e:
        logging.error(f"Ошибка конфигурации: {e}")
        raise


def setup_logging(log_level: str) -> logging.Logger:
    level = getattr(logging, log_level, logging.INFO)
    # Устанавливаем уровень для root логгера, чтобы он не блокировал наши сообщения DEBUG
    logging.getLogger().setLevel(min(level, logging.WARNING if log_level != "DEBUG" else logging.DEBUG))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    )
    # Отключаем слишком подробные логи от некоторых библиотек, если наш уровень не DEBUG
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("qdrant_client").setLevel(logging.INFO)  # Qdrant может быть довольно "болтливым"
    return logging.getLogger(__name__)


class RegeneratedQuery(BaseModel):
    regenerated_question: str = PydanticField(
        description="Перефразированный вопрос пользователя на русском языке, оптимизированный для поиска по базе законодательных актов.")
    reasoning: str = PydanticField(
        description="Краткое объяснение, почему вопрос был перефразирован таким образом, и какие аспекты законодательства он затрагивает.")


class RankedDocumentInfo(BaseModel):
    document_index: int = PydanticField(description="Индекс документа в исходном списке (начиная с 0).")
    relevance_score: float = PydanticField(
        description="Оценка релевантности документа запросу (от 0.0 до 1.0) с точки зрения юридической значимости правовой нормы.")
    reasoning: str = PydanticField(
        description="Краткое объяснение, почему документ получил такую оценку релевантности, с учетом его нормативного содержания и применимости к правовой концепции запроса.")


class RankedDocumentsOutput(BaseModel):
    ranked_infos: List[RankedDocumentInfo] = PydanticField(
        description="Список ранжированных документов с их оценками и обоснованиями.")


def extract_basic_metadata_from_filename(filename: str) -> Dict[str, Any]:
    metadata = {}
    name_without_ext = Path(filename).stem.upper()  # Приводим к верхнему регистру для сопоставления
    parts = name_without_ext.split('_')

    if not parts: return metadata

    doc_type_map = {
        "ЗАКОН": "Закон", "УКАЗ": "Указ", "ПОСТАНОВЛЕНИЕ": "Постановление", "ДЕКРЕТ": "Декрет",
        "КОДЕКС": "Кодекс", "ПРИКАЗ": "Приказ", "ДИРЕКТИВА": "Директива",
        "КОАП": "Кодекс об административных правонарушениях", "УК": "Уголовный кодекс",
        "ГК": "Гражданский кодекс", "НК": "Налоговый кодекс", "ТК": "Трудовой кодекс",
        "СТАТЬЯ": "Статья"
    }
    # Ищем наиболее длинное совпадение типа документа из нескольких слов
    # Например "МЕТОДИЧЕСКИЕ РЕКОМЕНДАЦИИ"
    # Сначала пробуем ключи из нескольких слов
    for i in range(min(3, len(parts)), 0, -1):  # От 3 слов до 1
        potential_type_key = "_".join(parts[:i])
        if potential_type_key in doc_type_map:
            metadata[META_DOCUMENT_TYPE] = doc_type_map[potential_type_key]
            remaining_parts_str = "_".join(parts[i:])
            break
    else:  # Если не найдено многословного типа
        if parts[0] in doc_type_map:
            metadata[META_DOCUMENT_TYPE] = doc_type_map[parts[0]]
            remaining_parts_str = "_".join(parts[1:])
        else:
            metadata[META_DOCUMENT_TYPE] = "Документ"
            remaining_parts_str = "_".join(parts)

    # Извлечение даты (YYYY-MM-DD или YYYY) и номера из remaining_parts_str
    # Используем нежадный поиск для названия, чтобы отдать приоритет дате и номеру
    # Паттерн: (Опциональный_Номер)_?(Опциональная_Дата)_?(Название) или (Название)_?(Опциональный_Номер)_?(Опциональная_Дата)
    # Этот парсер можно значительно улучшить для специфичных форматов РБ

    date_match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{4})", remaining_parts_str)
    if date_match:
        metadata[META_PUBLICATION_DATE] = date_match.group(1)
        # Удаляем дату из строки, чтобы она не попала в номер или название
        remaining_parts_str = remaining_parts_str.replace(date_match.group(1), "").replace("__", "_").strip("_")

    # Поиск номера (например, 123-З, 42, VII)
    # Предположим, что номер - это короткая строка с цифрами, возможно, буквами и дефисом
    id_match = re.search(r"([\w\d]{1,10}-?[\w\d]{0,5})", remaining_parts_str)  # Более общий паттерн
    title_candidate = remaining_parts_str.replace('_', ' ').strip()

    if id_match and (len(id_match.group(1)) < 8 or any(c.isdigit() for c in id_match.group(1))):
        # Если найденный ID не слишком длинный или содержит цифры (чтобы не принять часть названия за ID)
        id_val = id_match.group(1)
        # Проверяем, не является ли ID частью более длинного слова (избегаем совпадений внутри слов)
        temp_title = remaining_parts_str
        # Удаляем ID из строки для получения названия
        # Аккуратное удаление, чтобы не испортить название, если ID - это часть слова
        # (например, если ID="VII" в "ГЛАВАVII")
        # Лучше использовать regex для удаления, если ID найден
        # Например, если remaining_parts_str = "123-Z_O_NALOGAH", id_val = "123-Z"
        # title_candidate должен стать "O_NALOGAH"
        pattern_id_in_str = r"(^|_)" + re.escape(id_val) + r"($|_)"
        if re.search(pattern_id_in_str, temp_title):
            metadata[META_LAW_ID] = id_val
            title_candidate = re.sub(pattern_id_in_str, "_", temp_title, 1).replace("__", "_").strip("_").replace('_',
                                                                                                                  ' ').strip()

    if title_candidate:
        metadata[META_DOCUMENT_TITLE] = title_candidate
    elif META_DOCUMENT_TITLE not in metadata and Path(filename).stem:  # Если после всего название не установлено
        metadata[META_DOCUMENT_TITLE] = Path(filename).stem.replace('_', ' ').strip()

    # Если файл называется "Статья_X..." или содержит "СТ_X"
    article_match_name = re.search(r"(?:СТАТЬЯ|СТ)_([\w\d\.]+)", name_without_ext)
    if article_match_name:
        metadata[META_ARTICLE_NUMBER] = article_match_name.group(1)
        if not metadata.get(META_DOCUMENT_TITLE) and metadata.get(META_DOCUMENT_TYPE) != "Статья":
            # Если это не файл типа "Статья", а просто в имени есть "Статья_X", то название может быть другое
            pass  # Оставляем предыдущее название или имя файла
        elif metadata.get(META_DOCUMENT_TYPE) == "Статья" and not metadata.get(META_DOCUMENT_TITLE):
            metadata[META_DOCUMENT_TITLE] = f"Статья {article_match_name.group(1)}"

    if not metadata.get(META_DOCUMENT_TITLE):  # Абсолютный fallback
        metadata[META_DOCUMENT_TITLE] = Path(filename).stem.replace("_", " ")

    return metadata


class RAGPipeline:
    def __init__(self, config: AppConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # --- Раздельные модели эмбеддингов для документов и запросов ---
        # ПРИМЕЧАНИЕ: Стандартные ретриверы LangChain (типа as_retriever()) обычно используют
        # эмбеддер, с которым был инициализирован VectorStore. Для использования отдельного
        # query_embeddings_model может потребоваться вызов vector_store.similarity_search_by_vector()
        # или кастомный ретривер. Пока что self.vectorstore будет использовать document_embeddings_model.
        self.document_embeddings_model = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model_name,
            google_api_key=self.config.google_api_key,
            task_type="retrieval_document"
        )
        self.query_embeddings_model = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model_name,  # Та же базовая модель
            google_api_key=self.config.google_api_key,
            task_type="retrieval_query"
        )
        # Для chat_llm используем основной API ключ
        self.chat_llm = ChatGoogleGenerativeAI(
            model=self.config.chat_model_name,
            temperature=self.config.llm_temperature,
            google_api_key=self.config.google_api_key,
            convert_system_message_to_human=True
        )

        if self.config.qdrant_location == ":memory:":
            self.logger.warning("Qdrant используется в режиме :memory:. Данные будут потеряны после перезапуска.")
            self.qdrant_client = QdrantClient(location=":memory:")
        else:
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_location,
                api_key=self.config.qdrant_api_key,
                prefer_grpc=self.config.qdrant_prefer_grpc
            )
            self.logger.info(f"Qdrant client инициализирован для URL: {self.config.qdrant_location}")

        self.vectorstore: Optional[Qdrant] = self._init_vectorstore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\nСтатья", "\nСтатья", "\n\nГлава", "\nГлава", "\n\nРаздел", "\nРаздел", "\n\nПункт",
                        "\nПункт", "\n\n§", "\n§", "\n\n", "\n", ". ", " ", ""]
        )
        self.base_retriever: Optional[Any] = None  # Для MultiQueryRetriever или стандартного
        self._init_chains_and_retrievers()

    def _init_vectorstore(self) -> Optional[Qdrant]:
        if self.config.qdrant_location == ":memory:":
            self.logger.info("Qdrant в режиме :memory:, проверка существующей коллекции пропускается.")
            # В режиме :memory: коллекция всегда создается заново при необходимости
            return None

        try:
            collection_info = self.qdrant_client.get_collection(collection_name=self.config.qdrant_collection_name)
            vectors_config = collection_info.config.params.vectors
            if isinstance(vectors_config,
                          models.NamedVectorParams):  # Если используется именованная конфигурация векторов
                actual_vector_size = vectors_config.size
            elif isinstance(vectors_config, models.VectorParams):  # Если используется стандартная конфигурация
                actual_vector_size = vectors_config.size
            elif isinstance(vectors_config, dict):  # Если это словарь конфигураций для нескольких векторов
                # Пытаемся найти конфигурацию по умолчанию или первую попавшуюся
                default_vector_config = vectors_config.get("")  # Qdrant <1.7 uses "" for default unnamed vector
                if default_vector_config:
                    actual_vector_size = default_vector_config.size
                else:  # Fallback if structure is unexpected or no default
                    first_key = next(iter(vectors_config))
                    actual_vector_size = vectors_config[first_key].size
                    self.logger.warning(
                        f"Не удалось определить размер вектора по умолчанию, используется первый найденный: '{first_key}' с размером {actual_vector_size}")
            else:
                self.logger.error(
                    f"Неизвестный формат конфигурации векторов: {vectors_config}. Невозможно проверить размер.")
                return None

            if actual_vector_size != self.config.vector_size:
                self.logger.error(
                    f"Размер векторов в существующей коллекции Qdrant ('{actual_vector_size}') "
                    f"не совпадает с конфигурацией ('{self.config.vector_size}'). "
                    f"Пересоздайте коллекцию с флагом force_recreate=True или измените vector_size в config.yaml."
                )
                return None

            self.logger.info(f"Использование существующей коллекции Qdrant '{self.config.qdrant_collection_name}'.")
            return Qdrant(
                client=self.qdrant_client,
                collection_name=self.config.qdrant_collection_name,
                embeddings=self.document_embeddings_model  # Используем document_embeddings_model для VectorStore
            )
        except Exception as e:
            self.logger.info(
                f"Коллекция '{self.config.qdrant_collection_name}' не найдена или недоступна ({type(e).__name__}: {e}). Потребуется создание.")
            return None

    def _init_chains_and_retrievers(self):  # Переименовано для ясности
        # --- Инициализация цепочек (LCEL) ---
        regenerate_parser = PydanticOutputParser(pydantic_object=RegeneratedQuery)
        regenerate_prompt = ChatPromptTemplate.from_template(
            template=self.config.regeneration_prompt_template_str,
            partial_variables={"format_instructions": regenerate_parser.get_format_instructions()}
        )
        self.query_regeneration_chain = (
                regenerate_prompt
                | self.chat_llm
                | regenerate_parser
        )

        rank_parser = PydanticOutputParser(pydantic_object=RankedDocumentsOutput)
        rank_prompt = ChatPromptTemplate.from_template(
            template=self.config.ranking_prompt_template_str,
            partial_variables={"format_instructions": rank_parser.get_format_instructions()}
        )
        self.document_ranking_chain = (
                rank_prompt
                | self.chat_llm
                | rank_parser
        )

        qa_prompt_lcel = ChatPromptTemplate.from_template(
            template=self.config.qa_with_sources_prompt_template_str
        )
        self.qa_generator_chain = (
                qa_prompt_lcel
                | self.chat_llm
                | StrOutputParser()
        )

        # --- Инициализация ретривера ---
        # Базовый ретривер создается здесь, но может быть переопределен, если vectorstore еще не готов.
        # Окончательная инициализация ретривера происходит в `run_pipeline` после проверки `self.vectorstore`.
        if self.vectorstore:
            standard_retriever = self.vectorstore.as_retriever(
                search_type=self.config.search_type,
                search_kwargs={
                    "k": self.config.search_fetch_k,  # Используем fetch_k для начальной выборки
                    # Для MMR можно добавить lambda_mult, если нужно кастомизировать
                    # "lambda_mult": 0.5,
                    # "fetch_k": self.config.search_fetch_k # Для MMR fetch_k здесь, а k - это финальное k
                }
            )
            if self.config.enable_multiquery_retriever:
                self.logger.info(
                    f"Включен MultiQueryRetriever (k={self.config.search_k}, LLM={self.config.chat_model_name}).")
                # Для MultiQueryRetriever, k у базового ретривера должен быть настроен так,
                # чтобы общее количество документов было разумным.
                # Например, если MultiQuery генерирует 3 запроса, и мы хотим в итоге ~fetch_k документов,
                # то k для каждого подзапроса = fetch_k / 3.
                # Или, если мы хотим получить search_k документов после MultiQuery, то можно
                # каждому подзапросу дать search_k, а потом уже отбирать.
                # Langchain по умолчанию объединяет результаты и удаляет дубликаты.
                # Мы хотим получить search_fetch_k документов ДО этапа LLM-ранжирования.
                # MultiQueryRetriever сам по себе не ограничивает финальное k, он собирает все результаты.
                # Ограничение k будет применяться позже, после этапа ретривинга.

                # Базовый ретривер для MultiQuery должен извлекать достаточно документов для каждого из сгенерированных запросов.
                # Если search_fetch_k - это общее желаемое количество перед ранжированием,
                # а MultiQuery генерирует, скажем, 3-5 запросов, то k для базового ретривера MultiQuery
                # может быть self.config.search_fetch_k // 3 (или другой делитель).
                # Однако, если мы хотим, чтобы каждый вариант запроса имел шанс найти хорошие доки,
                # можно поставить k = self.config.search_k для базового ретривера MultiQuery.
                # Затем мы будем отбирать self.config.search_k лучших из *объединенного* списка.

                mq_base_retriever_k = self.config.search_k  # Каждый вариант запроса ищет k документов
                self.logger.info(f"MultiQueryRetriever будет использовать базовый ретривер с k={mq_base_retriever_k}")

                mq_base_retriever = self.vectorstore.as_retriever(
                    search_type=self.config.search_type,  # Можно 'similarity' для скорости подзапросов
                    search_kwargs={"k": mq_base_retriever_k}
                )

                self.base_retriever = MultiQueryRetriever.from_llm(
                    retriever=mq_base_retriever,
                    llm=self.chat_llm
                )
            else:
                self.logger.info(
                    f"Используется стандартный ретривер (type={self.config.search_type}, fetch_k={self.config.search_fetch_k}).")
                self.base_retriever = standard_retriever
        else:
            self.logger.warning("Vectorstore не инициализирован, ретривер будет создан позже в run_pipeline.")

    def _get_document_loader(self, file_path: Path) -> Optional[Any]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            try:
                return PyPDFLoader(str(file_path))
            except ImportError:
                self.logger.warning(
                    "PyPDFLoader недоступен или не смог загрузить. Использую UnstructuredPDFLoader для %s.",
                    file_path.name)
                return UnstructuredPDFLoader(str(file_path), mode="elements", strategy="auto")
        elif ext in [".doc", ".docx"]:
            return UnstructuredWordDocumentLoader(str(file_path), mode="elements", strategy="auto")
        elif ext == ".odt":
            return UnstructuredODTLoader(str(file_path), mode="elements", strategy="auto")
        elif ext == ".txt":
            return TextLoader(str(file_path), encoding='utf-8')
        else:
            self.logger.warning(f"Неподдерживаемый тип файла: {file_path.name}")
            return None

    async def load_and_index_documents(self, force_recreate: bool = False):
        self.logger.info(f"Загрузка документов из: {self.config.data_directory}")
        source_dir = Path(self.config.data_directory)
        if not source_dir.is_dir():
            self.logger.error(f"Директория {source_dir} не найдена.")
            return

        all_docs_for_splitting: List[Document] = []
        failed_files: List[Path] = []
        file_paths = [p for p in source_dir.rglob("*.*") if
                      p.is_file() and not p.name.startswith('.') and not p.name.startswith('~')]

        for file_path in tqdm(file_paths, desc="Загрузка и предобработка файлов"):
            loader = self._get_document_loader(file_path)
            if loader:
                try:
                    loaded_file_parts = await asyncio.to_thread(loader.load)
                    for doc_part in loaded_file_parts:
                        initial_metadata = doc_part.metadata if isinstance(doc_part.metadata, dict) else {}
                        filename_metadata = extract_basic_metadata_from_filename(file_path.name)
                        final_metadata = {**initial_metadata, **filename_metadata}
                        try:
                            final_metadata['source_relative_path'] = str(file_path.relative_to(source_dir))
                        except ValueError:
                            final_metadata['source_relative_path'] = str(file_path)
                        final_metadata[META_SOURCE_FILE_NAME] = str(file_path.name)
                        if META_DOCUMENT_TITLE not in final_metadata or not final_metadata[META_DOCUMENT_TITLE]:
                            final_metadata[META_DOCUMENT_TITLE] = final_metadata.get(META_SOURCE_FILE_NAME,
                                                                                     "Неизвестный документ")
                        doc_part.metadata = final_metadata
                        all_docs_for_splitting.append(doc_part)
                    self.logger.debug(f"Загружено {len(loaded_file_parts)} частей из {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Ошибка загрузки {file_path.name}: {e}",
                                      exc_info=self.config.detailed_debug_logging)
                    failed_files.append(file_path)

        if failed_files: self.logger.warning(f"Не удалось загрузить: {', '.join(map(str, failed_files))}")
        if not all_docs_for_splitting:
            self.logger.warning("Документы для индексации не найдены.")
            return

        chunks: List[Document] = []
        self.logger.info(f"Разбиение {len(all_docs_for_splitting)} загруженных частей на чанки...")
        for doc_to_split in tqdm(all_docs_for_splitting, desc="Разбиение на чанки"):
            try:
                doc_chunks = self.text_splitter.split_documents([doc_to_split])  # split_documents ожидает список
                for chunk in doc_chunks:
                    # Наследуем метаданные и пытаемся найти номер статьи
                    chunk.metadata = doc_to_split.metadata.copy()  # Важно копировать
                    chunk_content_for_article_search = chunk.page_content.strip()
                    article_match = re.match(r"^\s*(?:Статья|Ст\.|Глава|Раздел|Пункт)\s+([\w\d\.\-]+)[\.\s]?",
                                             chunk_content_for_article_search, re.IGNORECASE | re.MULTILINE)
                    if article_match:
                        extracted_article_num = article_match.group(1).strip()
                        chunk.metadata[META_ARTICLE_NUMBER] = extracted_article_num
                        # Опционально: удалить "Статья X." из начала контента чанка, если это мешает
                        # chunk.page_content = re.sub(r"^\s*(?:Статья|Ст\.|Глава|Раздел|Пункт)\s+[\w\d\.\-]+[\.\s]?", "", chunk_content_for_article_search, 1, re.IGNORECASE | re.MULTILINE).strip()
                chunks.extend(doc_chunks)
            except Exception as e:
                self.logger.error(
                    f"Ошибка разбиения документа '{doc_to_split.metadata.get(META_SOURCE_FILE_NAME, 'Неизвестно')}': {e}",
                    exc_info=self.config.detailed_debug_logging)

        self.logger.info(f"Всего создано {len(chunks)} чанков.")
        if not chunks:
            self.logger.warning("После разбиения не осталось чанков для индексации.")
            return

        # --- Работа с Qdrant коллекцией ---
        collection_exists_on_init = self.vectorstore is not None

        if force_recreate and (collection_exists_on_init or self.config.qdrant_location != ":memory:"):
            self.logger.info(f"Принудительное удаление коллекции '{self.config.qdrant_collection_name}'.")
            try:
                # Проверяем, существует ли коллекция перед удалением, чтобы избежать ошибки
                self.qdrant_client.get_collection(
                    collection_name=self.config.qdrant_collection_name)  # Это вызовет исключение, если нет
                self.qdrant_client.delete_collection(collection_name=self.config.qdrant_collection_name)
                self.logger.info(f"Коллекция '{self.config.qdrant_collection_name}' успешно удалена.")
            except Exception as e:  # Exception, если коллекция не найдена (зависит от клиента Qdrant)
                self.logger.warning(
                    f"Не удалось удалить коллекцию '{self.config.qdrant_collection_name}' (возможно, ее не было): {type(e).__name__} - {e}")
            self.vectorstore = None  # Сбросить, чтобы была создана заново

        if self.vectorstore is None:  # Если не была инициализирована или была удалена
            try:
                self.logger.info(f"Создание новой коллекции Qdrant '{self.config.qdrant_collection_name}'.")
                self.qdrant_client.recreate_collection(  # recreate_collection удалит, если существует, и создаст
                    collection_name=self.config.qdrant_collection_name,
                    vectors_config=models.VectorParams(size=self.config.vector_size, distance=models.Distance.COSINE)
                )
                self.logger.info(f"Коллекция '{self.config.qdrant_collection_name}' успешно создана/пересоздана.")
                self.vectorstore = Qdrant(
                    client=self.qdrant_client,
                    collection_name=self.config.qdrant_collection_name,
                    embeddings=self.document_embeddings_model  # Используем document_embeddings_model
                )
            except Exception as e:
                self.logger.error(
                    f"Критическая ошибка: не удалось создать коллекцию '{self.config.qdrant_collection_name}': {e}",
                    exc_info=True)
                self.vectorstore = None
                return

        if self.vectorstore:
            self.logger.info(f"Индексация {len(chunks)} чанков в Qdrant...")
            try:
                await self.vectorstore.aadd_documents(documents=chunks, batch_size=128)  # Увеличен batch_size
                self.logger.info(f"Индексация {len(chunks)} чанков завершена.")
            except Exception as e:
                self.logger.error(f"Ошибка при добавлении документов в Qdrant: {e}", exc_info=True)
        else:
            self.logger.error("Векторное хранилище не было инициализировано. Индексация невозможна.")

        # После индексации переинициализируем ретриверы, так как vectorstore мог быть создан заново
        self._init_chains_and_retrievers()

    def process_user_query(self, query: str) -> str:
        cleaned_query = query.strip()
        self.logger.info(f"Очищенный запрос пользователя: {cleaned_query}")
        return cleaned_query

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
    async def _invoke_llm_chain_with_retry(self, chain, input_data: Dict):
        self.logger.debug(f"Вызов LLM цепочки ({type(chain).__name__}) с данными (ключи): {list(input_data.keys())}")
        if self.config.detailed_debug_logging:
            self.logger.debug(f"Полные данные для LLM цепочки: {input_data}")
        result = await chain.ainvoke(input_data)
        if self.config.detailed_debug_logging:
            self.logger.debug(f"Ответ LLM цепочки: {str(result)[:1000]}...")
        else:
            self.logger.debug(f"Ответ LLM цепочки (начало): {str(result)[:300]}...")
        return result

    async def regenerate_query_with_llm(self, user_query: str) -> RegeneratedQuery:
        self.logger.info(f"Регенерация запроса для: '{user_query}'")
        try:
            output: RegeneratedQuery = await self._invoke_llm_chain_with_retry(
                self.query_regeneration_chain, {"question": user_query}
            )
            self.logger.info(f"Регенерированный запрос: '{output.regenerated_question}', Причина: {output.reasoning}")
            return output
        except RetryError as re_err:
            self.logger.error(
                f"Не удалось регенерировать запрос '{user_query}' после нескольких попыток: {re_err.last_attempt.exception()}",
                exc_info=self.config.detailed_debug_logging)
        except Exception as e:
            self.logger.error(f"Ошибка при регенерации запроса '{user_query}': {e}",
                              exc_info=self.config.detailed_debug_logging)
        return RegeneratedQuery(regenerated_question=user_query,
                                reasoning="Регенерация запроса не удалась, используется исходный вопрос.")

    async def retrieve_relevant_documents(self, query_for_retrieval: str) -> List[Document]:
        if not self.vectorstore:
            self.logger.error("Векторное хранилище не инициализировано. Поиск невозможен.")
            return []
        if not self.base_retriever:
            self.logger.error("Базовый ретривер не инициализирован. Поиск невозможен.")
            # Попытка инициализировать на лету, если vectorstore уже есть
            if self.vectorstore:
                self._init_chains_and_retrievers()
                if not self.base_retriever:
                    return []  # Если все равно не удалось
            else:
                return []

        self.logger.info(f"Получение релевантных документов для запроса: '{query_for_retrieval}'")
        self.logger.info(
            f"Тип ретривера: {'MultiQueryRetriever' if self.config.enable_multiquery_retriever else self.config.search_type}")
        self.logger.info(
            f"Параметры поиска: k(fetch)={self.config.search_fetch_k}, k(final_to_rank_or_qa)={self.config.search_k}")

        # --- Диагностический "сырой" поиск (до MMR/MultiQuery) ---
        # Этот блок полезен для понимания, что находит самый базовый семантический поиск
        if self.config.detailed_debug_logging:
            try:
                self.logger.debug(
                    f"ДИАГНОСТИКА: Выполняется сырой similarity_search для: '{query_for_retrieval}' с k={self.config.search_fetch_k}")
                # ПРИМЕЧАНИЕ: Используется self.document_embeddings_model, т.к. он привязан к self.vectorstore
                # Для использования self.query_embeddings_model здесь, нужен similarity_search_by_vector
                # query_vector = await self.query_embeddings_model.aembed_query(query_for_retrieval)
                # raw_search_results = await self.vectorstore.asimilarity_search_by_vector_with_relevance_scores(
                #     embedding=query_vector, k=self.config.search_fetch_k
                # )
                # Пока что используем стандартный search, который применит эмбеддер хранилища
                raw_search_results = await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query_for_retrieval, k=self.config.search_fetch_k
                )

                self.logger.debug(f"ДИАГНОСТИКА: Найдено {len(raw_search_results)} сырых результатов:")
                for i, (doc, score) in enumerate(raw_search_results[:5]):  # Логируем только первые 5 для краткости
                    self.logger.debug(
                        f"  ДИАГН. {i + 1}: Score={score:.4f}, "
                        f"Title='{doc.metadata.get(META_DOCUMENT_TITLE, 'N/A')}', "
                        f"Article='{doc.metadata.get(META_ARTICLE_NUMBER, 'N/A')}', "
                        f"Content[:100]='{doc.page_content[:100].strip()}'"
                    )
            except Exception as e_diag:
                self.logger.error(f"ДИАГНОСТИКА: Ошибка сырого поиска: {e_diag}",
                                  exc_info=self.config.detailed_debug_logging)
        # --- Конец диагностического блока ---

        try:
            # Используем self.base_retriever, который может быть MultiQuery или стандартным
            retrieved_documents = await self.base_retriever.aget_relevant_documents(query_for_retrieval)

            # MultiQueryRetriever может вернуть больше документов, чем search_k, т.к. он объединяет результаты
            # Стандартный ретривер с search_type=mmr уже учтет search_fetch_k (в search_kwargs) и вернет k
            # Если это НЕ MMR, то стандартный ретривер вернет search_fetch_k документов

            self.logger.info(
                f"Ретривером ({type(self.base_retriever).__name__}) найдено {len(retrieved_documents)} документов до финального отбора.")

            # Если это был MultiQuery или простой similarity, и мы получили больше, чем search_k,
            # а MMR не применялся на уровне ретривера, то здесь можно отобрать топ-k по схожести
            # (если бы у нас были скоры) или просто первые k.
            # Однако, если дальше идет LLM-ранжирование, лучше передать ему больше документов (search_fetch_k).
            # Финальный отбор до self.config.search_k произойдет после LLM-ранжирования или перед QA.

            # Если это НЕ MMR и НЕ MultiQuery, то стандартный ретривер вернул self.config.search_fetch_k документов.
            # Если это MMR, он уже должен был вернуть self.config.search_k (но это зависит от того, как k и fetch_k интерпретируются MMR)
            # Лучше передавать search_fetch_k на ранжирование, а потом отбирать search_k.

            if self.config.detailed_debug_logging and retrieved_documents:
                self.logger.debug(
                    f"Первые {min(3, len(retrieved_documents))} извлеченных документов (до LLM-ранжирования):")
                for i, doc in enumerate(retrieved_documents[:3]):
                    self.logger.debug(
                        f"  РЕТР. {i + 1}: Title='{doc.metadata.get(META_DOCUMENT_TITLE, 'N/A')}', "
                        f"Article='{doc.metadata.get(META_ARTICLE_NUMBER, 'N/A')}', "
                        f"Content[:100]='{doc.page_content[:100].strip()}'"
                    )
            return retrieved_documents  # Возвращаем все, что извлек ретривер (до search_fetch_k)

        except Exception as e:
            self.logger.error(f"Ошибка при получении релевантных документов: {e}", exc_info=True)
            return []

    def _format_docs_for_prompt(self, documents: List[Document], for_ranking: bool = False) -> str:
        formatted_parts = []
        for i, doc_chunk in enumerate(documents):
            title = doc_chunk.metadata.get(META_DOCUMENT_TITLE,
                                           doc_chunk.metadata.get(META_SOURCE_FILE_NAME, f"Документ {i + 1}"))
            law_id_str = f" (Документ №: {doc_chunk.metadata[META_LAW_ID]})" if META_LAW_ID in doc_chunk.metadata else ""
            date_str = f" (Дата: {doc_chunk.metadata[META_PUBLICATION_DATE]})" if META_PUBLICATION_DATE in doc_chunk.metadata else ""

            article_parts = []
            if META_ARTICLE_NUMBER in doc_chunk.metadata: article_parts.append(doc_chunk.metadata[META_ARTICLE_NUMBER])
            # Article title не используется в этом форматировании, но может быть добавлено
            article_str = f" ({', '.join(article_parts)})" if article_parts else ""

            header_prefix = f"--- Документ {i} ---" if for_ranking else f"--- Фрагмент из: {title}{law_id_str}{article_str}{date_str} ---"
            meta_for_ranking = ""
            if for_ranking:  # Для ранжирования можно добавить больше метаданных отдельно
                meta_parts_ranking = []
                if title: meta_parts_ranking.append(f"Название: {title}")
                if law_id_str: meta_parts_ranking.append(law_id_str.strip(" ()"))  # убираем скобки
                if article_str: meta_parts_ranking.append(article_str.strip(" ()"))
                if date_str: meta_parts_ranking.append(date_str.strip(" ()"))
                if meta_parts_ranking: meta_for_ranking = f"\nМетаданные: {'; '.join(meta_parts_ranking)}"

            formatted_parts.append(f"{header_prefix}{meta_for_ranking}\nСодержание: {doc_chunk.page_content}")
        return "\n\n".join(formatted_parts)

    async def rank_documents_with_llm(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents: return []
        # Ограничиваем количество документов для ранжирования, если их слишком много
        # search_fetch_k - это сколько мы получили от ретривера
        docs_to_rank = documents[:self.config.search_fetch_k]  # Берем не более search_fetch_k

        self.logger.info(f"Ранжирование {len(docs_to_rank)} документов с помощью LLM для запроса: '{query}'...")
        if not docs_to_rank: return []

        formatted_documents_for_prompt = self._format_docs_for_prompt(docs_to_rank, for_ranking=True)

        try:
            ranked_output_obj: RankedDocumentsOutput = await self._invoke_llm_chain_with_retry(
                self.document_ranking_chain,
                {"question": query, "documents": formatted_documents_for_prompt}
            )

            relevant_docs_after_ranking: List[Tuple[Document, float, str]] = []  # (doc, score, reasoning)
            initial_doc_map = {i: doc for i, doc in enumerate(docs_to_rank)}

            for ranked_info in ranked_output_obj.ranked_infos:
                if 0 <= ranked_info.document_index < len(docs_to_rank):
                    doc_to_consider = initial_doc_map[ranked_info.document_index]
                    self.logger.debug(
                        f"LLM оценил документ {ranked_info.document_index} "
                        f"(Файл: {doc_to_consider.metadata.get(META_SOURCE_FILE_NAME, 'N/A')}, "
                        f"Статья: {doc_to_consider.metadata.get(META_ARTICLE_NUMBER, 'N/A')}) "
                        f"с оценкой: {ranked_info.relevance_score:.2f}. Причина: {ranked_info.reasoning}"
                    )
                    if ranked_info.relevance_score >= self.config.relevance_threshold:
                        relevant_docs_after_ranking.append(
                            (doc_to_consider, ranked_info.relevance_score, ranked_info.reasoning))
                else:
                    self.logger.warning(
                        f"LLM-ранжировщик вернул некорректный индекс документа: {ranked_info.document_index}.")

            relevant_docs_after_ranking.sort(key=lambda item: item[1], reverse=True)

            # Оставляем только self.config.search_k лучших документов ПОСЛЕ ранжирования
            final_ranked_documents = [doc for doc, score, reasoning in
                                      relevant_docs_after_ranking[:self.config.search_k]]

            self.logger.info(
                f"Отобрано {len(final_ranked_documents)} документов из {len(docs_to_rank)} после LLM ранжирования (порог={self.config.relevance_threshold}, топ k={self.config.search_k}).")
            return final_ranked_documents
        except RetryError as re_err:
            self.logger.error(
                f"Не удалось ранжировать документы после нескольких попыток: {re_err.last_attempt.exception()}",
                exc_info=self.config.detailed_debug_logging)
        except Exception as e:
            self.logger.error(f"Ошибка при LLM-ранжировании документов: {e}.",
                              exc_info=self.config.detailed_debug_logging)

        self.logger.warning(
            "Возвращаем первые {self.config.search_k} документов из исходного набора из-за ошибки/отсутствия результатов LLM-ранжирования.")
        return docs_to_rank[:self.config.search_k]

    async def generate_answer(self, query: str, context_documents: List[Document]) -> str:
        if not context_documents:
            self.logger.warning(f"Нет контекстных документов для генерации ответа на запрос: '{query}'.")
            return "К сожалению, по вашему запросу не найдено достаточно релевантных документов в базе знаний для формирования ответа."

        self.logger.info(
            f"Генерация ответа с LLM для запроса: '{query}' и {len(context_documents)} документами в контексте...")
        summaries_for_prompt = self._format_docs_for_prompt(context_documents, for_ranking=False)

        # --- Логирование контекста для QA LLM ---
        if self.config.detailed_debug_logging:
            self.logger.debug(f"ЗАПРОС ДЛЯ QA LLM: {query}")
            self.logger.debug(f"ПОЛНЫЙ КОНТЕКСТ (SUMMARIES) ДЛЯ QA LLM:\n{summaries_for_prompt}")
        else:
            self.logger.info(
                f"QA LLM: Запрос='{query}', Контекст (начало)='{summaries_for_prompt[:300].replace(chr(10), ' ')}...'")
        # --- Конец логирования ---
        try:
            answer = await self._invoke_llm_chain_with_retry(
                self.qa_generator_chain,
                {"question": query, "summaries": summaries_for_prompt}
            )
            # Логика добавления источников, если LLM не справилась (ваша предыдущая версия была хороша)
            if not re.search(r"ИСТОЧНИКИ:", answer, re.IGNORECASE) and context_documents:
                self.logger.warning("LLM не включила блок 'ИСТОЧНИКИ:'. Формируем вручную.")
                sources_data_strings = []
                processed_source_identifiers = set()
                for doc_chunk in context_documents:
                    display_name = doc_chunk.metadata.get(META_DOCUMENT_TITLE)
                    if not display_name: display_name = doc_chunk.metadata.get(META_SOURCE_FILE_NAME,
                                                                               "Неизвестный источник")

                    law_id = doc_chunk.metadata.get(META_LAW_ID)
                    article_info = doc_chunk.metadata.get(META_ARTICLE_NUMBER)
                    pub_date = doc_chunk.metadata.get(META_PUBLICATION_DATE)
                    source_identifier = (display_name, article_info if article_info else "Весь документ")

                    if display_name != "Неизвестный источник" and source_identifier not in processed_source_identifiers:
                        parts = [f"- {display_name}"]
                        if law_id: parts.append(f"(№ {law_id})")
                        if article_info: parts.append(f", {article_info}")
                        if pub_date: parts.append(f", от {pub_date}")
                        sources_data_strings.append(" ".join(parts))
                        processed_source_identifiers.add(source_identifier)
                if sources_data_strings:
                    answer_main_part = re.split(r"\n\s*ИСТОЧНИКИ:", answer, flags=re.IGNORECASE | re.DOTALL)[0]
                    answer = f"{answer_main_part.strip()}\n\nИСТОЧНИКИ:\n" + "\n".join(sources_data_strings)
                else:
                    answer = f"{answer.strip()}\n\nИСТОЧНИКИ: информация предоставлена на основе внутренней базы знаний."
            return answer.strip()
        except RetryError as re_err:
            self.logger.error(f"Не удалось сгенерировать ответ '{query}': {re_err.last_attempt.exception()}",
                              exc_info=self.config.detailed_debug_logging)
        except Exception as e:
            self.logger.error(f"Ошибка при генерации ответа '{query}': {e}",
                              exc_info=self.config.detailed_debug_logging)
        return "При генерации ответа произошла системная ошибка. Пожалуйста, попробуйте позже."

    def format_response(self, answer: str) -> str:
        return answer.strip()

    async def run_pipeline(self, user_query_raw: str, force_reindex: bool = False) -> str:
        self.logger.info(f"Старт обработки запроса: '{user_query_raw}' (Переиндексация: {force_reindex})")

        if force_reindex or self.vectorstore is None:
            self.logger.info("Требуется инициализация/переиндексация базы знаний...")
            await self.load_and_index_documents(force_recreate=force_reindex)
            if self.vectorstore is None:
                self.logger.critical("База знаний не была инициализирована. Обработка запроса невозможна.")
                return "Ошибка: База знаний недоступна. Пожалуйста, обратитесь к администратору."
            # После успешной индексации/переиндексации, ретриверы должны быть переинициализированы
            self._init_chains_and_retrievers()

        cleaned_query = self.process_user_query(user_query_raw)
        query_for_retrieval = cleaned_query
        query_for_llms = cleaned_query  # Исходный вопрос для передачи в LLM ранжировщик и генератор ответа

        if self.config.enable_query_regeneration:
            regenerated_output = await self.regenerate_query_with_llm(cleaned_query)
            query_for_retrieval = regenerated_output.regenerated_question  # Используем регенерированный для поиска
            # query_for_llms можно оставить оригинальным или тоже использовать регенерированный.
            # Часто для LLM-ранжирования и ответа лучше оригинальный вопрос пользователя, т.к. он отражает его намерение.
            # query_for_llms = regenerated_output.regenerated_question # Если хотите и для LLM использовать регенерированный
            self.logger.info(f"Запрос для ретривера после регенерации: '{query_for_retrieval}'")
            self.logger.info(f"Запрос для LLM (ранжирование/ответ): '{query_for_llms}'")

        # Этап 1: Ретривинг документов
        # retrieved_documents будет содержать до search_fetch_k документов
        retrieved_documents = await self.retrieve_relevant_documents(query_for_retrieval)

        if not retrieved_documents:
            self.logger.warning(f"Не найдено документов на этапе ретривинга для запроса: '{query_for_retrieval}'.")
            return f"К сожалению, по вашему запросу '{user_query_raw}' не найдено информации в базе законодательных актов."

        documents_for_processing = retrieved_documents  # Это до search_fetch_k документов

        # Этап 2: LLM-Ранжирование (опционально)
        if self.config.enable_llm_ranking and len(documents_for_processing) > 0:
            # LLM-ранжировщик получит documents_for_processing (до search_fetch_k)
            # и вернет до self.config.search_k лучших
            ranked_documents = await self.rank_documents_with_llm(query_for_llms, documents_for_processing)
            if not ranked_documents:
                self.logger.warning(
                    "После LLM-ранжирования не осталось релевантных документов. Используем топ-{self.config.search_k} из результатов ретривера.")
                # Берем лучшие из того, что вернул ретривер, если ранжировщик все отбросил
                # (предполагаем, что retrieve_relevant_documents вернул их уже в каком-то порядке)
                documents_for_qa = documents_for_processing[:self.config.search_k]
            else:
                documents_for_qa = ranked_documents  # Это уже топ self.config.search_k
        else:
            self.logger.info("LLM-ранжирование документов отключено или нет документов для ранжирования.")
            # Если ранжирование отключено, берем топ-k из результатов ретривера
            documents_for_qa = documents_for_processing[:self.config.search_k]

        if not documents_for_qa:
            self.logger.warning("Не осталось документов для генерации ответа после всех этапов.")
            return f"Не удалось найти достаточно релевантные документы для вашего запроса '{user_query_raw}'."

        # Этап 3: Генерация ответа
        answer = await self.generate_answer(query_for_llms, documents_for_qa)
        formatted_answer = self.format_response(answer)
        self.logger.info(f"Финальный ответ (длина: {len(formatted_answer)} симв.): {formatted_answer[:300]}...")
        return formatted_answer


# --- Функции для тестового окружения (без изменений, кроме возможных улучшений в extract_basic_metadata_from_filename)



async def main():
    config = None
    try:
        config = load_config()
    except Exception as e:
        print(f"Критическая ошибка: не удалось загрузить конфигурацию: {e}. Завершение работы.")
        return

    logger = setup_logging(config.log_level)

    if not config.google_api_key or "YOUR_GOOGLE_API_KEY_HERE" in config.google_api_key:
        logger.critical("GOOGLE_API_KEY не установлен или используется значение по умолчанию. Укажите валидный ключ.")
        return
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    logger.info("--- Запуск RAG-пайплайна (улучшенный поиск) ---")



    pipeline = RAGPipeline(config, logger)

    # --- Тестовый запуск для индексации ---
    logger.info("Первоначальная инициализация/индексация базы знаний...")
    # Используем простой запрос для инициализации, не обязательно связанный с тестовыми данными напрямую
    init_query = "Общие принципы законодательства"
    init_result = await pipeline.run_pipeline(init_query, force_reindex=True)
    logger.info(f"Результат инициализационного запроса для '{init_query}' (начало): {init_result[:200]}...")

    if pipeline.vectorstore is None:
        logger.critical(
            "Не удалось инициализировать RAG пайплайн и его базу знаний. Проверьте логи Qdrant и настройки.")
        return

    # --- Тестовые запросы ---
    queries = [
        "Какое наказание предусмотрено за мелкое хищение имущества?",
        "Что такое преступление по УК РБ?",
        "Что будет если я украду ядерное топливо",
        "Разрешено ли плавать за буйками"
    ]

    for i, user_query in enumerate(queries):
        print(f"\n================ ЗАПРОС ПОЛЬЗОВАТЕЛЯ ({i + 1}) ================\n{user_query}")
        logger.info(f"Обработка запроса пользователя ({i + 1}): {user_query}")
        # Включаем детальное логирование для первого тестового запроса, чтобы посмотреть на контекст
        # config.detailed_debug_logging = (i == 0) # или всегда True/False в config.yaml
        answer = await pipeline.run_pipeline(user_query, force_reindex=False)
        print(f"\n---------------- ОТВЕТ КОНСУЛЬТАНТА ----------------\n{answer}")
        print("======================================================")
        if i < len(queries) - 1: await asyncio.sleep(1)  # Пауза между запросами

    logger.info("--- RAG-пайплайн завершил работу ---")


if __name__ == "__main__":
    asyncio.run(main())