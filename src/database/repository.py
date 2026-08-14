import json
import pandas as pd
from sqlalchemy import text
from src.paths import METRICS


def write_frames(engine, users, events, transactions, snapshots=None, predictions=None):
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for table in ("customer_predictions","user_snapshots","transactions","events","users"):
            conn.execute(text(f"TRUNCATE TABLE {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    users.to_sql("users",engine,if_exists="append",index=False,chunksize=5000,method="multi")
    events.to_sql("events",engine,if_exists="append",index=False,chunksize=5000,method="multi")
    transactions.to_sql("transactions",engine,if_exists="append",index=False,chunksize=5000,method="multi")
    if snapshots is not None: snapshots.to_sql("user_snapshots",engine,if_exists="append",index=False,chunksize=3000,method="multi")
    if predictions is not None: predictions.to_sql("customer_predictions",engine,if_exists="append",index=False,chunksize=3000,method="multi")


def verify_mysql(engine, users, events, transactions, snapshots, monthly_kpis):
    expected={"users":len(users),"events":len(events),"transactions":len(transactions),"user_snapshots":len(snapshots)}
    with engine.connect() as conn:
        actual={table:int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()) for table in expected}
        sql_revenue=float(conn.execute(text("SELECT COALESCE(SUM(amount-refund_amount),0) FROM transactions")).scalar_one())
        orphan_events=int(conn.execute(text("SELECT COUNT(*) FROM events e LEFT JOIN users u USING(user_id) WHERE u.user_id IS NULL")).scalar_one())
        orphan_orders=int(conn.execute(text("SELECT COUNT(*) FROM transactions t LEFT JOIN users u USING(user_id) WHERE u.user_id IS NULL")).scalar_one())
    pandas_revenue=float(monthly_kpis.revenue.sum())
    result={"row_counts_expected":expected,"row_counts_actual":actual,"row_counts_match":expected==actual,
            "orphan_events":orphan_events,"orphan_transactions":orphan_orders,"sql_net_revenue":sql_revenue,
            "pandas_net_revenue":pandas_revenue,"revenue_absolute_difference":abs(sql_revenue-pandas_revenue),
            "revenue_matches":abs(sql_revenue-pandas_revenue)<.01}
    METRICS.mkdir(parents=True,exist_ok=True)
    (METRICS/"mysql_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not result["row_counts_match"] or orphan_events or orphan_orders or not result["revenue_matches"]:
        raise ValueError(f"MySQL reconciliation failed: {result}")
    return result
