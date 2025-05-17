from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


DATABASE_URL = "sqlite:///gos.sqlite3"


engine = create_engine(DATABASE_URL, echo=False)
created_session = sessionmaker(engine,expire_on_commit=False)


def get_session() -> Session:
    with created_session() as session:
        yield session

def create_section() -> Session:
    session_instance = created_session()
    return session_instance