import argparse
import numpy as np
import pandas as pd
from src.paths import RAW, ensure_directories

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-06-30 23:59:59")


def generate(users_count: int = 30_000, seed: int = 42, output_dir=RAW):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    user_ids = np.arange(1, users_count + 1)
    types = rng.choice(["high_value", "regular", "casual", "price_sensitive", "at_risk"], users_count,
                       p=[.09, .35, .27, .18, .11])
    channels = rng.choice(["organic", "search", "social", "referral", "display", "affiliate"], users_count,
                          p=[.24, .22, .18, .14, .10, .12])
    # Leave at least 120 days for labels; older users are slightly more common.
    signup_offsets = (rng.beta(1.15, 1.8, users_count) * 420).astype(int)
    signup = START + pd.to_timedelta(signup_offsets, unit="D")
    channel_cost = {"organic": 4, "search": 26, "social": 18, "referral": 9, "display": 34, "affiliate": 29}
    users = pd.DataFrame({
        "user_id": user_ids, "signup_date": signup.normalize(), "channel": channels,
        "region": rng.choice(["East", "South", "North", "West", "Central"], users_count, p=[.28,.24,.19,.13,.16]),
        "device": rng.choice(["Android", "iOS", "Web"], users_count, p=[.49,.32,.19]),
        "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], users_count, p=[.18,.34,.25,.15,.08]),
        "acquisition_cost": [max(0, rng.normal(channel_cost[c], channel_cost[c] * .22 + 2)) for c in channels]
    })
    users["acquisition_cost"] = users["acquisition_cost"].round(2)

    type_activity = {"high_value": 44, "regular": 25, "casual": 13, "price_sensitive": 18, "at_risk": 16}
    type_decay = {"high_value": .0010, "regular": .0018, "casual": .0030, "price_sensitive": .0023, "at_risk": .0065}
    channel_mult = {"organic": 1.08, "search": 1.02, "social": .92, "referral": 1.12, "display": .83, "affiliate": .96}
    event_types = np.array(["login", "browse", "search", "favorite", "add_to_cart", "purchase"])
    type_probs = {
        "high_value": [.18,.31,.16,.08,.13,.14], "regular": [.22,.35,.17,.07,.12,.07],
        "casual": [.31,.40,.16,.05,.06,.02], "price_sensitive": [.24,.34,.22,.07,.10,.03],
        "at_risk": [.29,.38,.17,.06,.07,.03]
    }
    event_frames = []
    order_frames = []
    event_id = order_id = 1
    cat = np.array(["Electronics", "Home", "Beauty", "Sports", "Apparel", "Grocery"])
    aov = {"high_value": 155, "regular": 78, "casual": 49, "price_sensitive": 42, "at_risk": 61}
    for i, (uid, s, typ, channel) in enumerate(zip(user_ids, signup, types, channels)):
        available = max(1, (END.normalize() - s).days + 1)
        expected = type_activity[typ] * channel_mult[channel] * (available / 365)
        n = max(1, rng.poisson(expected))
        # Exponential decay plus strong onboarding activity and some uniform visits.
        candidates = rng.integers(0, available, size=n * 3)
        weights = np.exp(-type_decay[typ] * candidates) * (1 + 1.5 * (candidates < 14))
        keep = rng.random(len(candidates)) < (weights / weights.max())
        days = candidates[keep][:n]
        if len(days) < n:
            days = np.r_[days, rng.integers(0, available, n - len(days))]
        hours = rng.integers(0, 24, n)
        minutes = rng.integers(0, 60, n)
        times = s + pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
        weekend_boost = np.where(pd.DatetimeIndex(times).dayofweek >= 5, 1.12, 1.0)
        selected_types = rng.choice(event_types, n, p=type_probs[typ])
        # Small weekend shift from browsing to purchase.
        flip = (selected_types == "browse") & (rng.random(n) < (weekend_boost - 1) * .12)
        selected_types[flip] = "purchase"
        durations = np.maximum(0, rng.lognormal(3.25, .75, n) - 8).round(1)
        event_frames.append(pd.DataFrame({"event_id": np.arange(event_id, event_id+n), "user_id": uid,
                                          "event_time": times, "event_type": selected_types,
                                          "session_duration": durations}))
        event_id += n
        purchase_times = times[selected_types == "purchase"]
        if len(purchase_times):
            amounts = rng.lognormal(np.log(aov[typ]) - .35, .72, len(purchase_times))
            month_factor = np.where(pd.DatetimeIndex(purchase_times).month.isin([11,12]), 1.10, 1.0)
            amounts = np.maximum(5, amounts * month_factor).round(2)
            refunded = rng.random(len(amounts)) < (.10 if typ == "price_sensitive" else .055)
            refund = np.where(refunded, amounts * rng.uniform(.25, 1, len(amounts)), 0).round(2)
            order_frames.append(pd.DataFrame({"order_id": np.arange(order_id, order_id+len(amounts)), "user_id": uid,
                                              "order_time": purchase_times, "amount": amounts,
                                              "refund_amount": np.minimum(refund, amounts),
                                              "product_category": rng.choice(cat, len(amounts))}))
            order_id += len(amounts)
    events = pd.concat(event_frames, ignore_index=True).sort_values("event_time")
    transactions = (pd.concat(order_frames, ignore_index=True).sort_values("order_time") if order_frames else
                    pd.DataFrame(columns=["order_id","user_id","order_time","amount","refund_amount","product_category"]))
    users.to_csv(output_dir / "users.csv", index=False, date_format="%Y-%m-%d")
    events.to_csv(output_dir / "events.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    transactions.to_csv(output_dir / "transactions.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    return users, events, transactions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_directories()
    users, events, orders = generate(args.users, args.seed)
    print(f"Generated {len(users):,} users, {len(events):,} events, {len(orders):,} orders")


if __name__ == "__main__":
    main()
