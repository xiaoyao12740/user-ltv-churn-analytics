import numpy as np
import pandas as pd
from src.paths import INTERIM

TARGETS = ["churn_30d", "future_90d_revenue"]
ID_COLS = ["user_id", "snapshot_date"]


def feature_columns(frame):
    return [c for c in frame.columns if c not in ID_COLS + TARGETS + ["split"]]


def build_snapshots(users, events, transactions, output_dir=INTERIM,
                    snapshot_dates=None):
    if snapshot_dates is None:
        snapshot_dates = pd.date_range("2024-07-01", "2025-03-01", freq="MS")
    users = users.copy(); users.signup_date = pd.to_datetime(users.signup_date)
    events = events.copy(); events.event_time = pd.to_datetime(events.event_time)
    transactions = transactions.copy(); transactions.order_time = pd.to_datetime(transactions.order_time)
    transactions["net_revenue"] = transactions.amount - transactions.refund_amount
    rows = []
    event_types = ["login","browse","search","add_to_cart"]
    for snap in snapshot_dates:
        eligible = users[users.signup_date < snap].copy().set_index("user_id")
        hist_e = events[events.event_time < snap].copy()
        hist_t = transactions[transactions.order_time < snap].copy()
        base = eligible.copy()
        base["snapshot_date"] = snap
        base["tenure_days"] = (snap - base.signup_date).dt.days
        for days in (7,30,90):
            recent = hist_e[hist_e.event_time >= snap - pd.Timedelta(days=days)]
            base[f"active_days_{days}d"] = recent.assign(d=recent.event_time.dt.date).groupby("user_id").d.nunique()
        recent30 = hist_e[hist_e.event_time >= snap - pd.Timedelta(days=30)]
        counts = recent30.groupby(["user_id","event_type"]).size().unstack(fill_value=0)
        for typ in event_types:
            base[f"{typ.replace('add_to_cart','cart')}_count_30d"] = counts.get(typ, pd.Series(dtype=float))
        for days in (30,90):
            rt = hist_t[hist_t.order_time >= snap - pd.Timedelta(days=days)]
            base[f"purchase_count_{days}d"] = rt.groupby("user_id").order_id.nunique()
            base[f"revenue_{days}d"] = rt.groupby("user_id").net_revenue.sum()
        last_active = hist_e.groupby("user_id").event_time.max()
        last_purchase = hist_t.groupby("user_id").order_time.max()
        base["days_since_last_active"] = (snap - last_active).dt.days
        base["days_since_last_purchase"] = (snap - last_purchase).dt.days
        base["avg_session_duration"] = hist_e.groupby("user_id").session_duration.mean()
        base["avg_order_value"] = hist_t.groupby("user_id").net_revenue.mean()
        base["purchase_frequency"] = hist_t.groupby("user_id").order_id.nunique() / np.maximum(base.tenure_days / 30, 1)
        base["historical_ltv"] = hist_t.groupby("user_id").net_revenue.sum()
        future30 = events[(events.event_time >= snap) & (events.event_time < snap + pd.Timedelta(days=30))]
        active_future = set(future30.user_id)
        base["churn_30d"] = (~base.index.isin(active_future)).astype(int)
        future90 = transactions[(transactions.order_time >= snap) & (transactions.order_time < snap + pd.Timedelta(days=90))]
        base["future_90d_revenue"] = future90.groupby("user_id").net_revenue.sum()
        base = base.reset_index()
        keep = ID_COLS + ["tenure_days","active_days_7d","active_days_30d","active_days_90d",
             "login_count_30d","browse_count_30d","search_count_30d","cart_count_30d",
             "purchase_count_30d","purchase_count_90d","revenue_30d","revenue_90d",
             "days_since_last_active","days_since_last_purchase","avg_session_duration","avg_order_value",
             "purchase_frequency","historical_ltv"] + TARGETS
        rows.append(base[keep].fillna({c: 0 for c in keep if c not in ID_COLS}))
    snapshots = pd.concat(rows, ignore_index=True)
    unique_dates = sorted(snapshots.snapshot_date.unique())
    if len(unique_dates) >= 3:
        train_end, val_end = unique_dates[-3], unique_dates[-2]
        snapshots["split"] = np.where(snapshots.snapshot_date < train_end, "train", np.where(snapshots.snapshot_date < val_end, "validation", "test"))
    else:
        snapshots["split"] = "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(output_dir / "user_snapshots.csv", index=False)
    return snapshots
