import pandas as pd
from src.paths import PROCESSED


def calculate_retention(users, events, output_dir=PROCESSED):
    u = users[["user_id","signup_date","channel","device"]].copy()
    u.signup_date = pd.to_datetime(u.signup_date).dt.normalize()
    e = events[["user_id","event_time"]].copy()
    e["event_date"] = pd.to_datetime(e.event_time).dt.normalize()
    x = e.merge(u, on="user_id")
    x["day"] = (x.event_date - x.signup_date).dt.days
    base = len(u)
    summary = {f"d{d}_retention": x.loc[x.day == d, "user_id"].nunique() / base for d in (1,7,30)}
    cohort_size = u.assign(cohort=u.signup_date.dt.to_period("M")).groupby("cohort").user_id.nunique()
    x["cohort"] = x.signup_date.dt.to_period("M")
    x["activity_month"] = x.event_date.dt.to_period("M")
    x["period"] = (x.activity_month.astype(int) - x.cohort.astype(int)).astype(int)
    counts = x.groupby(["cohort","period"]).user_id.nunique().unstack(fill_value=0)
    cohort = counts.div(cohort_size, axis=0)
    cohort.index = cohort.index.astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_dir / "retention_summary.csv", index=False)
    cohort.to_csv(output_dir / "cohort_retention.csv")
    for dimension in ("channel", "device"):
        merged = x[x.day.isin([1,7,30])].groupby([dimension,"day"]).user_id.nunique().unstack(fill_value=0)
        denom = u.groupby(dimension).user_id.nunique()
        merged.div(denom, axis=0).to_csv(output_dir / f"retention_by_{dimension}.csv")
    return summary, cohort

