import pandas as pd
from pathlib import Path
from database import create_db_and_tables, save_rating_history, get_all_rating_history
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    # 1. Ensure tables exist
    logger.info("Ensuring database tables exist...")
    create_db_and_tables()

    # 2. Check if CSV exists
    csv_path = Path(__file__).parent / "data" / "prediction_history.csv"
    if not csv_path.exists():
        logger.warning(f"CSV file not found at {csv_path}. Skipping migration.")
        return

    # 3. Read CSV
    logger.info(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        if df.empty:
            logger.info("CSV is empty. Nothing to migrate.")
            return
        
        # 4. Check if DB is already populated to make migration idempotent
        existing_records = get_all_rating_history()
        if len(existing_records) > 0:
            logger.info("Database already contains history records. Skipping migration.")
            return
        
        # 4. Save to DB
        logger.info(f"Migrating {len(df)} records to database...")
        records = df.to_dict(orient="records")
        for record in records:
            # Handle NaN values (SQLModel doesn't like them for float fields)
            clean_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            save_rating_history(clean_record)
            
        logger.info("Migration completed successfully.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
