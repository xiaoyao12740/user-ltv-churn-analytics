import os, time
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pymysql
from src.paths import ROOT


def get_engine():
    load_dotenv()
    url = (f"mysql+pymysql://{os.getenv('MYSQL_USER','ltv_user')}:{os.getenv('MYSQL_PASSWORD','change_me')}@"
           f"{os.getenv('MYSQL_HOST','localhost')}:{os.getenv('MYSQL_PORT','3306')}/"
           f"{os.getenv('MYSQL_DATABASE','user_ltv_analytics')}?charset=utf8mb4")
    return create_engine(url, pool_pre_ping=True)


def connection_available():
    try:
        with get_engine().connect() as conn: conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def wait_for_mysql(timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if connection_available(): return True
        time.sleep(2)
    return False


def initialize_schema():
    """Create database and tables from version-controlled MySQL SQL files."""
    load_dotenv()
    conn = pymysql.connect(host=os.getenv("MYSQL_HOST", "localhost"), port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "ltv_user"), password=os.getenv("MYSQL_PASSWORD", "change_me"),
        database=os.getenv("MYSQL_DATABASE", "user_ltv_analytics"), charset="utf8mb4", autocommit=True)
    try:
        with conn.cursor() as cursor:
            for name in ("02_create_tables.sql",):
                sql = (ROOT / "sql" / name).read_text(encoding="utf-8")
                for statement in (part.strip() for part in sql.split(";") if part.strip()): cursor.execute(statement)
    finally:
        conn.close()
