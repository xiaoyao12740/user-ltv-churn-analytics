import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features.build_snapshots import feature_columns
from src.paths import METRICS, MODELS, PROCESSED


def _metrics(y, pred, prob):
    return {"accuracy":float(accuracy_score(y,pred)), "precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)), "f1":float(f1_score(y,pred,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,prob)), "pr_auc":float(average_precision_score(y,prob))}


def train_churn(snapshots, seed=42):
    features = feature_columns(snapshots)
    train = snapshots[snapshots.split == "train"]; val = snapshots[snapshots.split == "validation"]; test = snapshots[snapshots.split == "test"]
    logistic = Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=seed))])
    forest = Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=160,min_samples_leaf=8,max_features="sqrt",class_weight="balanced",n_jobs=-1,random_state=seed))])
    models = {"logistic_regression":logistic,"random_forest":forest}
    for model in models.values(): model.fit(train[features],train.churn_30d)
    val_scores = {name: average_precision_score(val.churn_30d,m.predict_proba(val[features])[:,1]) for name,m in models.items()}
    selected = max(val_scores,key=val_scores.get)
    results={}; probs={}; predictions={}
    for name,model in models.items():
        probs[name]=model.predict_proba(test[features])[:,1]; predictions[name]=(probs[name]>=.5).astype(int)
        results[name]=_metrics(test.churn_30d,predictions[name],probs[name])
    results["selected_model"]=selected
    METRICS.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True); PROCESSED.mkdir(parents=True,exist_ok=True)
    (METRICS/"churn_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    for name,model in models.items(): joblib.dump(model,MODELS/f"churn_{name}.joblib")
    out=test[["user_id","snapshot_date","churn_30d"]].copy(); out["churn_probability"]=probs[selected]; out["churn_prediction"] = predictions[selected]
    out.to_csv(PROCESSED/"churn_predictions.csv",index=False)
    importance = forest.named_steps["model"].feature_importances_
    return results,out,importance,features,probs,predictions

