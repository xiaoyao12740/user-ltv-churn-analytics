import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from src.paths import FIGURES


def _save(name):
    plt.tight_layout(); plt.savefig(FIGURES/name,dpi=180,bbox_inches="tight"); plt.close()


def make_figures(daily,monthly,cohort,ltv,churn,segments,importance,features,probs,preds):
    FIGURES.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,3)); ax.axis("off")
    boxes=["Synthetic Data","Validation + MySQL","KPI + Cohorts","Leakage-safe Snapshots","LTV + Churn Models","Segments + Power BI"]
    for i,b in enumerate(boxes):
        x=.02+i*.16; ax.text(x,.5,b,ha="center",va="center",fontsize=9,bbox=dict(boxstyle="round,pad=.5",fc="#e8f1fb",ec="#2878b5"),transform=ax.transAxes)
        if i<len(boxes)-1: ax.annotate("",xy=(x+.115,.5),xytext=(x+.075,.5),arrowprops=dict(arrowstyle="->"),xycoords=ax.transAxes)
    ax.set_title("User LTV & Churn Analytics Pipeline",fontweight="bold"); _save("01_project_pipeline.png")
    fig,axs=plt.subplots(2,2,figsize=(12,7)); daily.plot(x="date",y="dau",ax=axs[0,0],legend=False,title="Daily Active Users"); monthly.plot(x="month",y="mau",ax=axs[0,1],marker="o",legend=False,title="Monthly Active Users"); monthly.plot(x="month",y="revenue",ax=axs[1,0],marker="o",legend=False,title="Monthly Net Revenue"); daily.plot(x="date",y="dau_mau",ax=axs[1,1],legend=False,title="DAU / MAU Stickiness"); _save("02_kpi_trends.png")
    plt.figure(figsize=(12,6)); plt.imshow(cohort.iloc[:,:12],aspect="auto",cmap="Blues",vmin=0,vmax=1); plt.colorbar(label="Retention"); plt.yticks(range(len(cohort.index)),cohort.index); plt.xticks(range(min(12,len(cohort.columns))),cohort.columns[:12]); plt.xlabel("Months Since Signup"); plt.ylabel("Signup Cohort"); plt.title("Monthly Cohort Retention"); _save("03_retention_cohort.png")
    plt.figure(figsize=(9,5)); plt.hist(ltv.future_90d_revenue,bins=60,color="#2878b5"); plt.xlabel("Future 90-day Net Revenue"); plt.ylabel("Snapshots"); plt.title("LTV Target Distribution"); _save("04_ltv_distribution.png")
    plt.figure(figsize=(7,6)); plt.scatter(ltv.future_90d_revenue,ltv.predicted_ltv,s=8,alpha=.25); lim=max(ltv.future_90d_revenue.max(),ltv.predicted_ltv.max()); plt.plot([0,lim],[0,lim],"--",color="black"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("LTV: Actual vs Predicted"); _save("05_ltv_actual_vs_predicted.png")
    y=churn.churn_30d
    fig,ax=plt.subplots(figsize=(7,6));
    for name,p in probs.items(): RocCurveDisplay.from_predictions(y,p,name=name.replace("_"," ").title(),ax=ax)
    ax.set_title("Churn ROC Curves"); _save("06_churn_roc.png")
    fig,ax=plt.subplots(figsize=(7,6));
    for name,p in probs.items(): PrecisionRecallDisplay.from_predictions(y,p,name=name.replace("_"," ").title(),ax=ax)
    ax.set_title("Churn Precision-Recall Curves"); _save("07_churn_pr_curve.png")
    selected=next(iter(preds)); fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(y,preds[selected],ax=ax,cmap="Blues"); ax.set_title(f"Confusion Matrix: {selected.replace('_',' ').title()}"); _save("08_confusion_matrix.png")
    imp=pd.Series(importance,index=features).nlargest(12).sort_values(); plt.figure(figsize=(9,6)); imp.plot.barh(color="#2878b5"); plt.title("Random Forest Feature Importance"); plt.xlabel("Importance"); _save("09_feature_importance.png")
    plt.figure(figsize=(9,6)); colors=segments.customer_segment.map({"Priority Retention":"#d62728","VIP Maintenance":"#2ca02c","Automated Reactivation":"#ff7f0e","Growth/Nurture":"#1f77b4"}); plt.scatter(segments.predicted_ltv,segments.churn_probability,c=colors,s=12,alpha=.45); plt.xlabel("Predicted LTV"); plt.ylabel("Churn Probability"); plt.title("Customer Value × Churn Risk Matrix"); _save("10_value_risk_matrix.png")

