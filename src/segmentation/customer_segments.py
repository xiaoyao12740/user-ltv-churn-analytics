import pandas as pd
from src.paths import PROCESSED, METRICS
import json


def segment_customers(ltv_predictions, churn_predictions, train_snapshots, output_dir=PROCESSED, value_threshold=None):
    positive_revenue = train_snapshots.loc[train_snapshots.future_90d_revenue > 0, "future_90d_revenue"]
    value_threshold = (float(value_threshold) if value_threshold is not None else
                       (float(positive_revenue.median()) if not positive_revenue.empty else 1.0))
    # A fixed operational probability cutoff is selected before test evaluation.
    risk_threshold = 0.5
    frame = ltv_predictions.merge(churn_predictions[["user_id","snapshot_date","churn_probability"]],on=["user_id","snapshot_date"])
    frame["value_segment"] = frame.predicted_ltv.ge(value_threshold).map({True:"High Value",False:"Low Value"})
    frame["risk_segment"] = frame.churn_probability.ge(risk_threshold).map({True:"High Risk",False:"Low Risk"})
    mapping = {
        ("High Value","High Risk"):("Priority Retention","Personal outreach and retention offer"),
        ("High Value","Low Risk"):("VIP Maintenance","VIP benefits and cross-sell"),
        ("Low Value","High Risk"):("Automated Reactivation","Automated win-back campaign"),
        ("Low Value","Low Risk"):("Growth/Nurture","Education and next-best-action nurture")}
    pairs=frame.apply(lambda r:mapping[(r.value_segment,r.risk_segment)],axis=1)
    frame["customer_segment"]=[x[0] for x in pairs]; frame["recommended_action"]=[x[1] for x in pairs]
    keep=["user_id","predicted_ltv","churn_probability","value_segment","risk_segment","customer_segment","recommended_action"]
    # One latest test snapshot row per user for activation use.
    result=frame.sort_values("snapshot_date").drop_duplicates("user_id",keep="last")[keep]
    output_dir.mkdir(parents=True,exist_ok=True); result.to_csv(output_dir/"customer_segments.csv",index=False)
    counts=result.customer_segment.value_counts().to_dict()
    (METRICS/"segmentation_metrics.json").write_text(json.dumps({"value_threshold":value_threshold,"risk_threshold":risk_threshold,"counts":counts},indent=2),encoding="utf-8")
    return result
