import pandas as pd
import pytest
from src.data.validate_data import validate_data

def test_valid_data(sample_data): assert validate_data(*sample_data)["status"]=="valid"
@pytest.mark.parametrize("mutation",["duplicate","unknown","event_before","negative_session","bad_amount","bad_refund"])
def test_integrity_rules(sample_data,mutation):
    u,e,t=[x.copy() for x in sample_data]
    if mutation=="duplicate": u.loc[1,"user_id"]=1
    elif mutation=="unknown": e.loc[0,"user_id"]=999
    elif mutation=="event_before": e.loc[0,"event_time"]=pd.Timestamp("2023-12-31")
    elif mutation=="negative_session": e.loc[0,"session_duration"]=-1
    elif mutation=="bad_amount": t.loc[0,"amount"]=0
    elif mutation=="bad_refund": t.loc[0,"refund_amount"]=101
    with pytest.raises(ValueError): validate_data(u,e,t)

