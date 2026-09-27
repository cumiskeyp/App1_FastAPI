"""
Phase 6 – App 1 (FastAPI): Model Validation Tool
S&P 500 Tech-5 Open-Gap Direction Predictor
================================================
Run locally:  uvicorn main:app --reload --port 8001
Deploy:       Render (see render.yaml)
"""

import io, base64, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
CONTINUOUS_COLS = [
    "daily_return","rsi_14","adx_14","macd","macd_hist","obv_trend",
    "spy_daily_return","vix_close","rsi_delta_3d","adx_delta_3d",
    "atr_14","bb_width","bb_pctb","realized_vol_20d",
    "stoch_k","stoch_d","roc_10","mom_10","cci_14",
    "vol_rel_20d","ad_line","adosc",
    "hl_range_pct","dist_from_20d_high","dist_from_20d_low",
    "gap_pct_lag1","vix_change",
]
CAT_COLS   = ["vix_bin","rsi_bin","adx_bin"]
TARGET_COL = "open_dir"

app = FastAPI(title="Phase 6 – App 1: Model Validation Tool")
templates = Jinja2Templates(directory="templates")

# Custom Jinja2 filters
templates.env.filters["enumerate"] = enumerate


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_models():
    models = {
        "Naive Bayes":       GaussianNB(),
        "Decision Tree":     DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=6,
                                                    random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,
                                                         learning_rate=0.05,
                                                         max_depth=4, random_state=42),
        "SVM":               SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
    }
    if XGBOOST_OK:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    return models


def prepare_data(df, feature_mode):
    y = df[TARGET_COL].astype(int)
    if feature_mode == "groomed":
        if "adosc_t1" not in df.columns:
            df = df.copy()
            df["adosc_t1"] = df["adosc"].shift(1)
        X = df[["adosc_t1"]].dropna()
        y = y.loc[X.index]
    else:
        available = [c for c in CONTINUOUS_COLS if c in df.columns]
        if "adosc_t1" not in df.columns and "adosc" in df.columns:
            df = df.copy()
            df["adosc_t1"] = df["adosc"].shift(1)
        if "adosc_t1" in df.columns:
            available.append("adosc_t1")
        parts = [df[available]]
        cats = [c for c in CAT_COLS if c in df.columns]
        if cats:
            parts.append(pd.get_dummies(df[cats], drop_first=True))
        X = pd.concat(parts, axis=1).dropna()
        y = y.loc[X.index]
    return X, y, list(X.columns)


def train_evaluate(model, X, y, train_frac=0.80):
    n = int(len(X) * train_frac)
    X_tr, X_te = X.iloc[:n], X.iloc[n:]
    y_tr, y_te = y.iloc[:n], y.iloc[n:]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model.fit(X_tr_s, y_tr)
    p_tr = model.predict_proba(X_tr_s)[:, 1]
    p_te = model.predict_proba(X_te_s)[:, 1]
    pred_te = (p_te >= 0.5).astype(int)
    return {
        "auc_tr": float(roc_auc_score(y_tr, p_tr)),
        "auc_te": float(roc_auc_score(y_te, p_te)),
        "fpr_tr": roc_curve(y_tr, p_tr)[0],
        "tpr_tr": roc_curve(y_tr, p_tr)[1],
        "fpr_te": roc_curve(y_te, p_te)[0],
        "tpr_te": roc_curve(y_te, p_te)[1],
        "cm": confusion_matrix(y_te, pred_te),
        "report": classification_report(y_te, pred_te, output_dict=True),
        "n_train": len(X_tr), "n_test": len(X_te),
        "pred_df": pd.DataFrame({
            "actual": y_te.values,
            "prob_up": np.round(p_te, 4),
            "predicted": pred_te,
            "correct": (pred_te == y_te.values).astype(int),
        }),
    }


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_roc(res, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(res["fpr_tr"], res["tpr_tr"],
            label=f"Train AUC = {res['auc_tr']:.4f}", linestyle="--", alpha=0.7)
    ax.plot(res["fpr_te"], res["tpr_te"],
            label=f"Test  AUC = {res['auc_te']:.4f}", linewidth=2, color="orange")
    ax.plot([0,1],[0,1],"k:",alpha=0.4,label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_cm(cm, model_name):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=["Down (0)","Up (1)"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    return fig_to_b64(fig)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    models = list(build_models().keys())
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "result": None,
    })


@app.post("/validate", response_class=HTMLResponse)
async def validate(
    request: Request,
    file: UploadFile = File(...),
    model_choice: str = Form(...),
    feature_mode: str = Form(...),
    train_frac: float = Form(...),
):
    models_avail = list(build_models().keys())
    error = None
    result = None

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        rows, cols = df.shape

        # Normalise ticker column
        if "Ticker" in df.columns:
            df = df.rename(columns={"Ticker": "ticker"})

        if TARGET_COL not in df.columns:
            raise ValueError(f"Column '{TARGET_COL}' not found in CSV.")

        X, y, feat_names = prepare_data(df, feature_mode)
        model = build_models()[model_choice]
        res = train_evaluate(model, X, y, train_frac)

        # Build classification report table
        rep = res["report"]
        rep_rows = []
        for k, label in [("0","Down (0)"),("1","Up (1)"),
                          ("macro avg","Macro Avg"),("weighted avg","Weighted Avg")]:
            if k in rep:
                r = rep[k]
                rep_rows.append({
                    "label": label,
                    "precision": f"{r.get('precision',0):.3f}",
                    "recall":    f"{r.get('recall',0):.3f}",
                    "f1":        f"{r.get('f1-score',0):.3f}",
                    "support":   f"{r.get('support',0):.0f}",
                })

        delta = res["auc_te"] - res["auc_tr"]
        result = {
            "model": model_choice,
            "feature_mode": feature_mode,
            "rows": rows, "cols": cols,
            "auc_tr": f"{res['auc_tr']:.4f}",
            "auc_te": f"{res['auc_te']:.4f}",
            "delta": f"{delta:+.4f}",
            "delta_class": "positive" if delta >= 0 else "negative",
            "n_train": f"{res['n_train']:,}",
            "n_test":  f"{res['n_test']:,}",
            "generalises": res["auc_te"] >= res["auc_tr"],
            "roc_img": make_roc(res, model_choice),
            "cm_img":  make_cm(res["cm"], model_choice),
            "rep_rows": rep_rows,
            "feat_names": feat_names,
            "pred_table": res["pred_df"].head(50).to_dict(orient="records"),
        }

    except Exception as e:
        error = str(e)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models_avail,
        "selected_model": model_choice,
        "selected_feature": feature_mode,
        "selected_frac": train_frac,
        "result": result,
        "error": error,
    })
