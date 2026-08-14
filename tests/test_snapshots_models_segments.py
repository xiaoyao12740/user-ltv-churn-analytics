import numpy as np
import pandas as pd
from src.features.build_snapshots import build_snapshots,feature_columns,TARGETS
from src.models.train_ltv import train_ltv
from src.models.train_churn import train_churn
from src.segmentation.customer_segments import segment_customers

def test_snapshot_uses_only_past(sample_data,tmp_path):
    u,e,t=sample_data
    snap=build_snapshots(u,e,t,tmp_path,[pd.Timestamp("2024-02-01")])
    row=snap[snap.user_id==1].iloc[0]
    assert row.historical_ltv==0 and row.future_90d_revenue==90
    assert not set(TARGETS)&set(feature_columns(snap))

def synthetic_snapshots(n=180):
    rng=np.random.default_rng(42); dates=np.repeat(pd.date_range("2024-01-01",periods=6,freq="MS"),n//6)
    x=rng.poisson(5,n); churn=(rng.random(n)<(.65-.05*np.minimum(x,8))).astype(int); y=np.maximum(0,x*8+rng.normal(0,15,n))
    return pd.DataFrame({"user_id":np.arange(n),"snapshot_date":dates,"tenure_days":rng.integers(30,400,n),"active_days_7d":x,"active_days_30d":x*2,"historical_ltv":x*12,"churn_30d":churn,"future_90d_revenue":y,"split":np.where(dates<dates[-60],"train",np.where(dates<dates[-30],"validation","test"))})

def test_training_and_segments(tmp_path,monkeypatch):
    import src.models.train_ltv as ltv_module
    import src.models.train_churn as churn_module
    monkeypatch.setattr(ltv_module, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(ltv_module, "MODELS", tmp_path / "models")
    monkeypatch.setattr(ltv_module, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(churn_module, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(churn_module, "MODELS", tmp_path / "models")
    monkeypatch.setattr(churn_module, "PROCESSED", tmp_path / "processed")
    s=synthetic_snapshots()
    lm,lp,_,_=train_ltv(s); cm,cp,_,_,_,_=train_churn(s)
    out=segment_customers(lp,cp,s[s.split=="train"],tmp_path)
    assert {"linear_regression","random_forest"}<=lm.keys()
    assert {"logistic_regression","random_forest"}<=cm.keys()
    assert set(out.customer_segment)<= {"Priority Retention","VIP Maintenance","Automated Reactivation","Growth/Nurture"}
