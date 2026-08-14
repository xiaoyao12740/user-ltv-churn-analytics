import pandas as pd


def validate_data(users: pd.DataFrame, events: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    errors: list[str] = []
    if users["user_id"].isna().any() or not users["user_id"].is_unique:
        errors.append("users.user_id must be unique and non-null")
        raise ValueError("Data validation failed:\n- " + "\n- ".join(errors))
    known = set(users["user_id"])
    unknown_events = set(events["user_id"]) - known
    unknown_orders = set(transactions["user_id"]) - known
    if unknown_events:
        errors.append(f"events contain {len(unknown_events)} unknown user_id values")
    if unknown_orders:
        errors.append(f"transactions contain {len(unknown_orders)} unknown user_id values")
    signup = users.set_index("user_id")["signup_date"]
    if not events.empty:
        event_signup = events["user_id"].map(signup)
        if (pd.to_datetime(events["event_time"]) < pd.to_datetime(event_signup)).any():
            errors.append("event_time must be on or after signup_date")
        if events["session_duration"].isna().any() or (events["session_duration"] < 0).any():
            errors.append("session_duration must be non-negative")
    if not transactions.empty:
        order_signup = transactions["user_id"].map(signup)
        if (pd.to_datetime(transactions["order_time"]) < pd.to_datetime(order_signup)).any():
            errors.append("order_time must be on or after signup_date")
        if transactions["amount"].isna().any() or (transactions["amount"] <= 0).any():
            errors.append("amount must be positive")
        if transactions["refund_amount"].isna().any() or (transactions["refund_amount"] < 0).any() or (transactions["refund_amount"] > transactions["amount"]).any():
            errors.append("refund_amount must be between zero and amount")
    if errors:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(errors))
    return {"users": len(users), "events": len(events), "transactions": len(transactions), "status": "valid"}


def load_and_validate(raw_dir):
    raw_dir = pd.io.common.stringify_path(raw_dir)
    users = pd.read_csv(f"{raw_dir}/users.csv", parse_dates=["signup_date"])
    events = pd.read_csv(f"{raw_dir}/events.csv", parse_dates=["event_time"])
    transactions = pd.read_csv(f"{raw_dir}/transactions.csv", parse_dates=["order_time"])
    return validate_data(users, events, transactions)
