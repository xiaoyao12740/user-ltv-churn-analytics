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

