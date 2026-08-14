import argparse, json, time
import pandas as pd
from src.paths import RAW, PROCESSED, METRICS, ensure_directories
from src.data.generate_data import generate
from src.data.validate_data import validate_data
from src.analytics.kpi import calculate_kpis
from src.analytics.retention import calculate_retention
from src.features.build_snapshots import build_snapshots
from src.models.train_ltv import train_ltv
from src.models.train_churn import train_churn
from src.segmentation.customer_segments import segment_customers
from src.visualization.make_figures import make_figures
from src.database.mysql_client import connection_available


def run(users_count=30_000,seed=42,skip_mysql=False):
    ensure_directories(); started=time.perf_counter(); timings={}
    def stage(name,fn):
        print(f"[{name}] starting...",flush=True); t=time.perf_counter(); value=fn(); timings[name]=round(time.perf_counter()-t,2); print(f"[{name}] done in {timings[name]:.2f}s",flush=True); return value
    users,events,orders=stage("generate",lambda:generate(users_count,seed))
    stage("validate",lambda:validate_data(users,events,orders))
    mysql_status="skipped by flag" if skip_mysql else ("connection available (load requires initialized schema)" if connection_available() else "unavailable; SQL/code artifacts retained")
    print(f"[mysql] {mysql_status}")
    daily,monthly=stage("kpi",lambda:calculate_kpis(users,events,orders))
    retention,cohort=stage("retention",lambda:calculate_retention(users,events))
    snapshots=stage("snapshots",lambda:build_snapshots(users,events,orders))
    ltv_metrics,ltv,_,_=stage("ltv",lambda:train_ltv(snapshots,seed))
    churn_metrics,churn,importance,features,probs,preds=stage("churn",lambda:train_churn(snapshots,seed))
    segments=stage("segmentation",lambda:segment_customers(ltv,churn,snapshots[snapshots.split=="train"],value_threshold=ltv_metrics["segmentation_value_threshold"]))
    # Consolidated fact exports for Power BI.
    latest=snapshots[snapshots.split=="test"].sort_values("snapshot_date").drop_duplicates("user_id",keep="last")
    latest.to_csv(PROCESSED/"powerbi_customer_features.csv",index=False)
    users.to_csv(PROCESSED/"powerbi_users.csv",index=False)
    stage("figures",lambda:make_figures(daily,monthly,cohort,ltv,churn,segments,importance,features,probs,preds))
    total=round(time.perf_counter()-started,2)
    run_info={"users":len(users),"events":len(events),"orders":len(orders),"date_start":str(min(users.signup_date.min(),events.event_time.min())),"date_end":str(max(events.event_time.max(),orders.order_time.max())),"mysql_status":mysql_status,"timings_seconds":timings,"total_seconds":total,"seed":seed}
    (METRICS/"pipeline_run.json").write_text(json.dumps(run_info,indent=2),encoding="utf-8")
    print(f"Pipeline complete in {total:.2f}s")
    return run_info


def main():
    p=argparse.ArgumentParser(); p.add_argument("--users",type=int,default=30_000); p.add_argument("--seed",type=int,default=42); p.add_argument("--skip-mysql",action="store_true"); a=p.parse_args(); run(a.users,a.seed,a.skip_mysql)
if __name__=="__main__": main()
