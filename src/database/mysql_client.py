import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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

