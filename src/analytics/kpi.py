import pandas as pd
import numpy as np
from src.paths import PROCESSED


def calculate_kpis(users, events, transactions, output_dir=PROCESSED):
    e = events.copy()
    e["date"] = pd.to_datetime(e.event_time).dt.normalize()
    daily = e.groupby("date").user_id.nunique().rename("dau").to_frame()
    dates = pd.date_range(e.date.min(), e.date.max(), freq="D")
    daily = daily.reindex(dates, fill_value=0).rename_axis("date")
    indexed = e.set_index("event_time")
    wau = indexed.groupby(pd.Grouper(freq="W-MON"))["user_id"].nunique().rename("wau")
    monthly = indexed.groupby(pd.Grouper(freq="MS"))["user_id"].nunique().rename("mau").to_frame()
    t = transactions.copy()
    t["net_revenue"] = t.amount - t.refund_amount
    t["month"] = pd.to_datetime(t.order_time).dt.to_period("M").dt.to_timestamp()
    revenue = t.groupby("month").net_revenue.sum()
    payers = t.groupby("month").user_id.nunique()
    orders = t.groupby("month").order_id.nunique()
    monthly = monthly.join(revenue.rename("revenue"), how="left").join(payers.rename("paying_users"), how="left").join(orders.rename("orders"), how="left").fillna(0)
    monthly["arpu"] = monthly.revenue / monthly.mau.replace(0, np.nan)
    monthly["arppu"] = monthly.revenue / monthly.paying_users.replace(0, np.nan)
    monthly["aov"] = monthly.revenue / monthly.orders.replace(0, np.nan)
    monthly["paying_rate"] = monthly.paying_users / monthly.mau.replace(0, np.nan)
    daily_mau = daily.index.to_period("M").map(monthly.mau)
    daily["dau_mau"] = daily.dau.to_numpy() / np.maximum(1, np.asarray(daily_mau, dtype=float))
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_csv(output_dir / "daily_kpis.csv", index=False)
    wau.reset_index().to_csv(output_dir / "weekly_kpis.csv", index=False)
    monthly.reset_index(names="month").to_csv(output_dir / "monthly_kpis.csv", index=False)
    channel = t.merge(users[["user_id","channel"]], on="user_id").groupby("channel").agg(revenue=("net_revenue","sum"), paying_users=("user_id","nunique"), orders=("order_id","nunique")).reset_index()
    channel.to_csv(output_dir / "channel_revenue.csv", index=False)
    return daily.reset_index(), monthly.reset_index(names="month")

