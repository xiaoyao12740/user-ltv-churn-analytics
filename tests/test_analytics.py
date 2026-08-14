import pandas as pd
from src.analytics.kpi import calculate_kpis
from src.analytics.retention import calculate_retention

def test_kpi_calculation(sample_data,tmp_path):
    _,monthly=calculate_kpis(*sample_data,output_dir=tmp_path)
    feb=monthly.loc[monthly.month==pd.Timestamp("2024-02-01")].iloc[0]
    assert feb.revenue==90 and feb.aov==90
def test_retention(sample_data,tmp_path):
    summary,_=calculate_retention(sample_data[0],sample_data[1],tmp_path)
    assert summary["d7_retention"]==.5

def censoring_data():
    users=pd.DataFrame({"user_id":[1,2,3],"signup_date":pd.to_datetime(["2024-01-01","2024-01-01","2024-03-01"]),"channel":["a"]*3,"device":["Web"]*3})
    events=pd.DataFrame({"user_id":[1,2,3],"event_time":pd.to_datetime(["2024-01-01","2024-03-15","2024-03-15"])})
    return users,events

def test_cohort_unmatured_period_is_nan(tmp_path):
    _,cohort=calculate_retention(*censoring_data(),tmp_path)
    assert pd.isna(cohort.loc["2024-03",1])

def test_matured_zero_retention_remains_zero(tmp_path):
    _,cohort=calculate_retention(*censoring_data(),tmp_path)
    assert cohort.loc["2024-01",1]==0
