import os
from datetime import datetime
from typing import Optional, List

from sqlmodel import Field, SQLModel, create_engine, Session, select

def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _normalize_database_url(url: str) -> str:
    # Heroku-style URLs sometimes use "postgres://", which SQLAlchemy rejects.
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


_load_dotenv_if_available()

# Database URL from environment or default to local docker-postgres settings.
DATABASE_URL = _normalize_database_url(
    os.getenv(
        "DATABASE_URL",
        "postgresql://vnrate_user:vnrate_password@localhost:5432/vnrate_db",
    )
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

class RatingHistory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    company_name: str
    ticker: str
    sector: str = "Unknown"
    rating_detail: str
    rating_date: str
    rating_agency: str = "VN-Rating AI"
    source: str = "VN-Rating Analyze"
    confidence: float = 0.0
    risk_score: float = 0.0
    risk_level: str = ""
    created_at_utc: datetime = Field(default_factory=datetime.utcnow)
    
    # Financial features
    current_ratio: Optional[float] = None
    debt_equity_ratio: Optional[float] = None
    gross_profit_margin: Optional[float] = None
    operating_profit_margin: Optional[float] = None
    ebit_margin: Optional[float] = None
    pretax_profit_margin: Optional[float] = None
    net_profit_margin: Optional[float] = None
    asset_turnover: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    operating_cashflow_ps: Optional[float] = None
    free_cashflow_ps: Optional[float] = None

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

def save_rating_history(record: dict, session: Optional[Session] = None):
    """Save a rating record to the database."""
    # Convert created_at_utc string to datetime object if needed
    if isinstance(record.get("created_at_utc"), str):
        try:
            record["created_at_utc"] = datetime.strptime(record["created_at_utc"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            record["created_at_utc"] = datetime.utcnow()
    
    history_item = RatingHistory(**record)
    
    if session is not None:
        session.add(history_item)
        session.commit()
        session.refresh(history_item)
        return history_item
    else:
        with Session(engine) as local_session:
            local_session.add(history_item)
            local_session.commit()
            local_session.refresh(history_item)
            return history_item

def get_all_rating_history() -> List[RatingHistory]:
    """Fetch all rating history records from the database."""
    with Session(engine) as session:
        statement = select(RatingHistory).order_by(RatingHistory.created_at_utc.desc())
        results = session.exec(statement)
        return list(results.all())
