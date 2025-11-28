import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

import json

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None


def get_table_name(date=None):
    """Generate table name based on date (YYYYMMDD format)."""
    if date is None:
        date = datetime.now()
    return f"events_{date.strftime('%Y%m%d')}"


def init_db(table_name=None):
    """Initializes the database by creating a daily event table if it doesn't exist."""
    if table_name is None:
        table_name = get_table_name()

    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()

            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                am_pm VARCHAR(2) CHECK (am_pm IN ('AM', 'PM')),
                start_time TIME,
                end_time TIME,
                peak TEXT,
                file_path TEXT,
                metadata JSONB
            );
            """
            cur.execute(create_table_query)

            conn.commit()
            cur.close()
            conn.close()
            print(f"Database table '{table_name}' initialized successfully.")
        except psycopg2.Error as e:
            print(f"Error initializing database: {e}")


def log_event(peak, file_path, start_time=None, end_time=None, metadata=None):
    """Logs an event to the database."""
    # Determine the table name based on start_time or current time
    if start_time:
        table_name = get_table_name(start_time)
        event_time = start_time
    else:
        now = datetime.now()
        table_name = get_table_name(now)
        event_time = now

    # Extract year, month, day
    year = event_time.year
    month = event_time.month
    day = event_time.day

    # Determine AM/PM based on hour
    am_pm = "AM" if event_time.hour < 12 else "PM"

    # Extract time only (HH:MM:SS) from datetime
    start_time_only = start_time.time() if start_time else None
    end_time_only = end_time.time() if end_time else None

    # Ensure the table exists for this date
    init_db(table_name)

    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            insert_query = f"""
            INSERT INTO {table_name} (year, month, day, am_pm, start_time, end_time, peak, file_path, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            # Ensure metadata is a JSON string if it's a dict
            if metadata and isinstance(metadata, dict):
                metadata_json = json.dumps(metadata)
            else:
                metadata_json = metadata

            cur.execute(
                insert_query,
                (
                    year,
                    month,
                    day,
                    am_pm,
                    start_time_only,
                    end_time_only,
                    peak,
                    file_path,
                    metadata_json,
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
            print(f"Event logged to '{table_name}': {peak} ({am_pm})")
        except psycopg2.Error as e:
            print(f"Error logging event: {e}")


if __name__ == "__main__":
    init_db()
