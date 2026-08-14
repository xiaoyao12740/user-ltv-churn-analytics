import os
import pytest
from sqlalchemy import text
from src.database.mysql_client import get_engine


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_MYSQL_TESTS") != "1", reason="set RUN_MYSQL_TESTS=1 to run MySQL integration checks")
def test_mysql_loaded_tables_and_foreign_keys():
    with get_engine().connect() as conn:
        counts = {table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
                  for table in ("users", "events", "transactions", "user_snapshots", "customer_predictions")}
        orphan_events = conn.execute(text("SELECT COUNT(*) FROM events e LEFT JOIN users u USING(user_id) WHERE u.user_id IS NULL")).scalar_one()
        orphan_orders = conn.execute(text("SELECT COUNT(*) FROM transactions t LEFT JOIN users u USING(user_id) WHERE u.user_id IS NULL")).scalar_one()
    assert all(value > 0 for value in counts.values())
    assert orphan_events == orphan_orders == 0

