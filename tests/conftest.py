import pandas as pd
import pytest

@pytest.fixture
def sample_data():
    users=pd.DataFrame({"user_id":[1,2],"signup_date":pd.to_datetime(["2024-01-01","2024-01-02"]),"channel":["organic","search"],"device":["Web","iOS"]})
    events=pd.DataFrame({"event_id":[1,2,3,4],"user_id":[1,1,2,1],"event_time":pd.to_datetime(["2024-01-01","2024-01-08","2024-01-03","2024-02-01"]),"event_type":["login","browse","login","purchase"],"session_duration":[10,20,5,30]})
    orders=pd.DataFrame({"order_id":[1],"user_id":[1],"order_time":pd.to_datetime(["2024-02-01"]),"amount":[100.],"refund_amount":[10.],"product_category":["Home"]})
    return users,events,orders

