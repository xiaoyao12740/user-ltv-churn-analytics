import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features.build_snapshots import feature_columns
from src.paths import METRICS, MODELS, PROCESSED


def _logit(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _metrics(y, pred, prob):
    prevalence = float(np.mean(y)); k = max(1, int(np.ceil(len(y) * .10)))
    top = np.argsort(prob)[-k:]; top_precision = float(np.mean(np.asarray(y)[top]))
    top_recall = float(np.asarray(y)[top].sum() / max(1, np.asarray(y).sum()))
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, prob)), "pr_auc": float(average_precision_score(y, prob)),
            "brier_score": float(brier_score_loss(y, prob)), "prevalence": prevalence,
            "no_skill_pr_auc": prevalence, "pr_auc_lift": float(average_precision_score(y, prob) / prevalence),
            "precision_at_top_10pct": top_precision, "recall_at_top_10pct": top_recall,
            "lift_at_top_10pct": float(top_precision / prevalence)}


def _choose_threshold(y, probability):
    candidates = np.linspace(.10, .90, 161)
    scores = [f1_score(y, probability >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def train_churn(snapshots, seed=42):
    features = feature_columns(snapshots)
    split_col = "churn_split" if "churn_split" in snapshots else "split"
    train = snapshots[snapshots[split_col] == "train"]
    val = snapshots[snapshots[split_col] == "validation"]
    test = snapshots[snapshots[split_col] == "test"]
    if test.snapshot_date.nunique() != 1:
        raise ValueError("Churn test must contain exactly one evaluation snapshot")
    models = {
        "logistic_regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(
            n_estimators=160, min_samples_leaf=8, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed))])}
    calibrated_val = {}; calibrated_test = {}; calibrators = {}; thresholds = {}
    for name, model in models.items():
        model.fit(train[features], train.churn_30d)
        raw_val = model.predict_proba(val[features])[:, 1]
        calibrator = LogisticRegression(random_state=seed).fit(_logit(raw_val), val.churn_30d)
        calibrators[name] = calibrator
        calibrated_val[name] = calibrator.predict_proba(_logit(raw_val))[:, 1]
        thresholds[name] = _choose_threshold(val.churn_30d, calibrated_val[name])
        raw_test = model.predict_proba(test[features])[:, 1]
        calibrated_test[name] = calibrator.predict_proba(_logit(raw_test))[:, 1]
    selected = max(models, key=lambda name: average_precision_score(val.churn_30d, calibrated_val[name]))
    results = {}
    predictions = {}
    for name, probability in calibrated_test.items():
        predictions[name] = (probability >= thresholds[name]).astype(int)
        results[name] = _metrics(test.churn_30d, predictions[name], probability)
        results[name]["validation_selected_threshold"] = thresholds[name]
        results[name]["fixed_0_5"] = _metrics(test.churn_30d, probability >= .5, probability)
    results["selected_model"] = selected
    results["calibration_method"] = "Platt scaling fitted on validation snapshot only"
    results["temporal_design"] = {"split_column": split_col,
        "validation_snapshot": str(val.snapshot_date.iloc[0]), "test_snapshot": str(test.snapshot_date.iloc[0]),
        "label_horizon_days": 30}

    METRICS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
    (METRICS / "churn_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, model in models.items():
        joblib.dump({"model": model, "calibrator": calibrators[name], "threshold": thresholds[name]}, MODELS / f"churn_{name}.joblib")
    out = test[["user_id", "snapshot_date", "churn_30d"]].copy()
    out["churn_probability"] = calibrated_test[selected]
    out["churn_prediction"] = predictions[selected]
    out.to_csv(PROCESSED / "churn_predictions.csv", index=False)
    importance = models["random_forest"].named_steps["model"].feature_importances_
    return results, out, importance, features, calibrated_test, predictions
