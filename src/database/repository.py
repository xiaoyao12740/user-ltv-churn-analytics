def write_frames(engine, users, events, transactions, snapshots=None, predictions=None):
    users.to_sql("users",engine,if_exists="append",index=False,chunksize=5000,method="multi")
    events.to_sql("events",engine,if_exists="append",index=False,chunksize=5000,method="multi")
    transactions.to_sql("transactions",engine,if_exists="append",index=False,chunksize=5000,method="multi")
    if snapshots is not None: snapshots.to_sql("user_snapshots",engine,if_exists="append",index=False,chunksize=3000,method="multi")
    if predictions is not None: predictions.to_sql("customer_predictions",engine,if_exists="append",index=False,chunksize=3000,method="multi")

