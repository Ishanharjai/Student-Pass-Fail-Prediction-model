import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, roc_auc_score)
from sklearn.base import clone

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Student Performance Analysis & Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# STYLING
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .main {
        background-color: #F7F9FC;
        background-image: radial-gradient(circle at 20% 50%, rgba(59,110,245,0.03) 0%, transparent 50%),
                          radial-gradient(circle at 80% 20%, rgba(109,140,247,0.03) 0%, transparent 50%);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------- APP HEADER ---------- */
    .app-header {
        padding: 1.6rem 2rem;
        background: linear-gradient(135deg, #1A1F4B 0%, #3B6EF5 50%, #6D8CF7 100%);
        border-radius: 18px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(59,110,245,0.3);
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: ''; position: absolute; top: -50%; right: -20%;
        width: 300px; height: 300px; background: rgba(255,255,255,0.05); border-radius: 50%;
    }
    .app-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
    .app-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

    /* ---------- CARDS ---------- */
    .card {
        background: white; padding: 1.3rem 1.5rem; border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #EEF1F6; margin-bottom: 1.1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 36px rgba(59,110,245,0.12), 0 4px 12px rgba(0,0,0,0.06);
        border-color: #C5D1F7;
    }

    /* ---------- GLASS INFO CARD ---------- */
    .info-glass {
        background: linear-gradient(135deg, rgba(59,110,245,0.06) 0%, rgba(109,140,247,0.04) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59,110,245,0.12);
        border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.1rem;
        transition: all 0.3s ease;
    }
    .info-glass:hover { box-shadow: 0 8px 32px rgba(59,110,245,0.1); border-color: rgba(59,110,245,0.25); }

    /* ---------- STAT BADGE ---------- */
    .stat-badge {
        background: white; padding: 1rem 1.3rem; border-radius: 12px;
        border-left: 4px solid #3B6EF5; box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        margin-bottom: 0.8rem; transition: all 0.3s ease;
    }
    .stat-badge:hover { transform: translateX(4px); box-shadow: 0 6px 20px rgba(59,110,245,0.12); }
    .stat-badge.green  { border-left-color: #1FAA59; }
    .stat-badge.red    { border-left-color: #E23744; }
    .stat-badge.purple { border-left-color: #9B5EF5; }
    .stat-badge.orange { border-left-color: #F59B3B; }
    .stat-badge .stat-label {
        font-size: 0.78rem; color: #7A8399; text-transform: uppercase;
        letter-spacing: 0.04em; font-weight: 500; margin: 0;
    }
    .stat-badge .stat-value { font-size: 1.35rem; font-weight: 700; color: #1E2749; margin: 0.15rem 0 0 0; }

    /* ---------- METRIC CARDS ---------- */
    .metric-card {
        background: white; padding: 1rem 1.2rem; border-radius: 14px;
        border: 1px solid #EEF1F6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 4px 14px rgba(0,0,0,0.05);
        text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        position: relative; overflow: hidden;
    }
    .metric-card::after {
        content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3B6EF5, #6D8CF7, #9B5EF5); opacity: 0; transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 32px rgba(59,110,245,0.15), 0 4px 10px rgba(0,0,0,0.05);
        border-color: #8FA0FF;
    }
    .metric-card:hover::after { opacity: 1; }
    .metric-card .label { font-size: 0.8rem; color: #7A8399; font-weight: 500; text-transform: uppercase; letter-spacing: 0.03em; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1E2749; }
    .metric-card .icon { font-size: 1.6rem; margin-bottom: 0.3rem; }

    /* ---------- RESULT BADGES ---------- */
    .result-pass, .result-fail {
        color: white; padding: 1.8rem; border-radius: 18px; text-align: center;
        animation: fadeSlideIn 0.5s ease; position: relative; overflow: hidden;
    }
    .result-pass {
        background: linear-gradient(135deg, #0D8B44 0%, #1FAA59 40%, #35C77E 100%);
        box-shadow: 0 10px 30px rgba(31,170,89,0.3), 0 4px 12px rgba(31,170,89,0.15);
    }
    .result-fail {
        background: linear-gradient(135deg, #B91C2E 0%, #E23744 40%, #F1585F 100%);
        box-shadow: 0 10px 30px rgba(226,55,68,0.3), 0 4px 12px rgba(226,55,68,0.15);
    }
    .result-pass::before, .result-fail::before {
        content: ''; position: absolute; top: -40%; right: -30%;
        width: 200px; height: 200px; background: rgba(255,255,255,0.08); border-radius: 50%;
    }
    @keyframes fadeSlideIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
    .result-pass h2, .result-fail h2 { margin: 0; font-size: 2rem; position: relative; }
    .result-pass p, .result-fail p { margin: 0.3rem 0 0 0; opacity: 0.95; position: relative; }

    /* ---------- RISK BADGE ---------- */
    .risk-badge {
        display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem; margin-top: 0.6rem;
    }
    .risk-low    { background: rgba(31,170,89,0.15); color: #0D8B44; border: 1px solid rgba(31,170,89,0.35); }
    .risk-medium { background: rgba(245,155,59,0.18); color: #B9700F; border: 1px solid rgba(245,155,59,0.4); }
    .risk-high   { background: rgba(226,55,68,0.15);  color: #B91C2E; border: 1px solid rgba(226,55,68,0.35); }

    /* ---------- PULSING DOT ---------- */
    .pulse-dot {
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        background: #1FAA59; margin-right: 8px; animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(31,170,89,0.5); }
        70%  { box-shadow: 0 0 0 8px rgba(31,170,89,0); }
        100% { box-shadow: 0 0 0 0 rgba(31,170,89,0); }
    }

    /* ---------- CONFIDENCE BAR ---------- */
    .confidence-bar-bg {
        background: rgba(255,255,255,0.2); border-radius: 20px; height: 12px;
        margin: 0.8rem auto 0 auto; max-width: 280px; overflow: hidden; position: relative;
    }
    .confidence-bar-fill { height: 100%; border-radius: 20px; background: rgba(255,255,255,0.85); transition: width 1s ease; }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #141937 0%, #1E2749 40%, #253060 100%); }
    section[data-testid="stSidebar"] * { color: #F7F9FC !important; }
    .sidebar-brand { text-align: center; padding: 0.6rem 0 0.2rem 0; }
    .sidebar-brand h2 {
        margin: 0; font-size: 1.45rem; font-weight: 700; letter-spacing: 0.02em;
        background: linear-gradient(90deg, #8FA0FF, #E0E6FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p { margin: 0.15rem 0 0 0; font-size: 0.78rem; color: #8FA0FF !important; letter-spacing: 0.04em; font-weight: 500; }
    .sidebar-divider { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 0.8rem 0; }

    /* ---------- BUTTONS ---------- */
    .stButton>button {
        background: linear-gradient(135deg, #3B6EF5 0%, #6D8CF7 100%);
        color: white; border: none; border-radius: 12px; padding: 0.7rem 1.6rem;
        font-weight: 600; width: 100%; transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(59,110,245,0.25); position: relative; overflow: hidden;
    }
    .stButton>button::after {
        content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(59,110,245,0.4); color: white; }
    .stButton>button:hover::after { left: 100%; }
    .stButton>button:active { transform: translateY(0); box-shadow: 0 2px 8px rgba(59,110,245,0.25); }

    .stDownloadButton>button {
        background: linear-gradient(135deg, #6D8CF7 0%, #9B5EF5 100%);
        color: white; border: none; border-radius: 12px; padding: 0.6rem 1.4rem;
        font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(155,94,245,0.2);
    }
    .stDownloadButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(155,94,245,0.35); color: white; }

    /* ---------- SECTION TITLE ---------- */
    .section-title {
        font-size: 1.2rem; font-weight: 600; color: #1E2749; margin-bottom: 0.6rem;
        padding-bottom: 0.3rem; border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, #3B6EF5, transparent) 1;
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: white; border-radius: 10px 10px 0 0; padding: 10px 18px;
        font-weight: 600; color: #5A6378; border: 1px solid #EEF1F6; border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B6EF5, #6D8CF7) !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# DATA LOADING
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("student_cleaned.csv")
    df = df.drop(columns=[c for c in ["Result", "Performance_Band"] if c in df.columns])
    return df

RAW_DF = load_data()

FEATURES = [
    "Hours_Studied", "attendance", "Sleep_hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity",
    "Parental Involvement", "Access_to_Resources", "ExtracurricularActivities",
    "motivation_level", "Internet_Access", "Family Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
]
CATEGORICAL = [
    "Parental Involvement", "Access_to_Resources", "ExtracurricularActivities",
    "motivation_level", "Internet_Access", "Family Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
]
NUMERIC = ["Hours_Studied", "attendance", "Sleep_hours", "Previous_Scores",
           "Tutoring_Sessions", "Physical_Activity"]
RADAR_FEATURES = NUMERIC

def label_results(df, threshold):
    out = df.copy()
    out["Result"] = np.where(out["Exam_Score"] >= threshold, "Pass", "Fail")
    return out

def group_by_feature(series_encoded, encoded_cols):
    grouped = {}
    for feat in FEATURES:
        matched = [c for c in encoded_cols if c == feat or c.startswith(feat + "_")]
        grouped[feat] = series_encoded[matched].sum() if matched else 0.0
    return pd.Series(grouped)

# =====================================================================
# MODEL TRAINING (cached per threshold)
# =====================================================================
@st.cache_resource(show_spinner="Training models…")
def train_all_models(threshold):
    data = label_results(RAW_DF, threshold)
    X = data[FEATURES]
    y = data["Result"]
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def metrics_for(model, X_te, y_te):
        y_pred = model.predict(X_te)
        fail_idx = list(model.classes_).index("Fail")
        y_proba_fail = model.predict_proba(X_te)[:, fail_idx]
        return {
            "y_pred": y_pred,
            "y_proba_fail": y_proba_fail,
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, pos_label="Fail", zero_division=0),
            "recall": recall_score(y_te, y_pred, pos_label="Fail", zero_division=0),
            "f1": f1_score(y_te, y_pred, pos_label="Fail", zero_division=0),
        }

    log_model = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=6).fit(X_train, y_train)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10).fit(X_train, y_train)

    results = {
        "Logistic Regression": {"model": log_model, "uses_scaling": True,
                                 **metrics_for(log_model, X_test_scaled, y_test)},
        "Decision Tree": {"model": dt_model, "uses_scaling": False,
                           **metrics_for(dt_model, X_test, y_test)},
        "Random Forest": {"model": rf_model, "uses_scaling": False,
                           **metrics_for(rf_model, X_test, y_test)},
    }

    comparison = pd.DataFrame([
        {"Model": name, "Accuracy": r["accuracy"], "Fail Precision": r["precision"],
         "Fail Recall": r["recall"], "Fail F1-Score": r["f1"]}
        for name, r in results.items()
    ]).sort_values(by=["Fail F1-Score", "Fail Recall", "Accuracy"], ascending=False).reset_index(drop=True)

    best_name = comparison.iloc[0]["Model"]

    # Refit each model on the FULL dataset so any of the 3 can be used for live prediction
    final_models = {}
    for name, info in results.items():
        final_model = clone(info["model"])
        if info["uses_scaling"]:
            final_scaler = StandardScaler().fit(X_encoded)
            final_model.fit(final_scaler.transform(X_encoded), y)
        else:
            final_scaler = None
            final_model.fit(X_encoded, y)
        final_models[name] = {"model": final_model, "scaler": final_scaler}

    return {
        "data": data,
        "results": results,
        "comparison": comparison,
        "best_name": best_name,
        "final_models": final_models,
        "encoded_columns": X_encoded.columns.tolist(),
        "y_test": y_test,
    }

def predict_with(model_name, input_dict, bundle):
    row = pd.DataFrame([input_dict])
    row_encoded = pd.get_dummies(row, drop_first=True).reindex(columns=bundle["encoded_columns"], fill_value=0)
    fm = bundle["final_models"][model_name]
    model = fm["model"]
    row_final = fm["scaler"].transform(row_encoded) if fm["scaler"] is not None else row_encoded
    pred = model.predict(row_final)[0]
    classes = list(model.classes_)
    proba = dict(zip(classes, model.predict_proba(row_final)[0])) if hasattr(model, "predict_proba") else None
    return pred, proba, row_encoded.iloc[0]

def explain_prediction(model_name, bundle, row_encoded):
    """Local explanation for Logistic Regression, global importance for tree models."""
    fm = bundle["final_models"][model_name]
    model = fm["model"]
    cols = bundle["encoded_columns"]
    if model_name == "Logistic Regression":
        scaled_row = fm["scaler"].transform(pd.DataFrame([row_encoded]))[0]
        classes = list(model.classes_)
        sign = 1 if classes[1] == "Fail" else -1  # coef_ points toward classes_[1]
        contributions = pd.Series(model.coef_[0] * scaled_row * sign, index=cols)
        grouped = group_by_feature(contributions, cols)
        grouped = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(6)
        is_local = True
    else:
        importances = pd.Series(model.feature_importances_, index=cols)
        grouped = group_by_feature(importances, cols).sort_values(ascending=False).head(6)
        is_local = False
    return grouped, is_local

def risk_band(fail_prob):
    if fail_prob < 0.30:
        return "Low Risk", "risk-low", "#1FAA59"
    elif fail_prob < 0.60:
        return "Medium Risk", "risk-medium", "#F59B3B"
    else:
        return "High Risk", "risk-high", "#E23744"

def radar_chart(input_dict, df, name="This Student", color="#3B6EF5"):
    student_vals, avg_vals = [], []
    for f in RADAR_FEATURES:
        lo, hi = df[f].min(), df[f].max()
        rng = (hi - lo) or 1
        student_vals.append((input_dict[f] - lo) / rng * 100)
        avg_vals.append((df[f].mean() - lo) / rng * 100)
    categories = RADAR_FEATURES + [RADAR_FEATURES[0]]
    student_vals += [student_vals[0]]
    avg_vals += [avg_vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='Class Average',
                                   line_color="#9B5EF5", opacity=0.55))
    fig.add_trace(go.Scatterpolar(r=student_vals, theta=categories, fill='toself', name=name,
                                   line_color=color))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=True, height=340, margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(orientation="h", y=-0.1),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def student_input_form(suffix=""):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📚 Study Habits**")
        hours_studied = st.slider("Hours Studied/week", 0, 50, 20, key=f"hs{suffix}")
        attendance = st.slider("Attendance (%)", 0, 100, 80, key=f"att{suffix}")
        sleep_hours = st.slider("Sleep Hours/night", 0, 12, 7, key=f"sl{suffix}")
        previous_scores = st.slider("Previous Scores", 0, 100, 75, key=f"ps{suffix}")
        tutoring_sessions = st.slider("Tutoring Sessions/month", 0, 10, 1, key=f"ts{suffix}")
        physical_activity = st.slider("Physical Activity (hrs/wk)", 0, 10, 3, key=f"pa{suffix}")
    with c2:
        st.markdown("**🏠 Support & Environment**")
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1, key=f"pi{suffix}")
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1, key=f"ar{suffix}")
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1, key=f"ml{suffix}")
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1, key=f"fi{suffix}")
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1, key=f"tq{suffix}")
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=1, key=f"pf{suffix}")
    with c3:
        st.markdown("**🧑 Personal Details**")
        distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"], index=0, key=f"df{suffix}")
        parental_education = st.selectbox("Parental Education", ["High School", "College", "Postgraduate"], index=0, key=f"pe{suffix}")
        extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"], index=1, key=f"ec{suffix}")
        internet_access = st.selectbox("Internet Access", ["No", "Yes"], index=1, key=f"ia{suffix}")
        learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"], index=0, key=f"ld{suffix}")
        school_type = st.selectbox("School Type", ["Public", "Private"], index=0, key=f"st{suffix}")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0, key=f"gd{suffix}")

    return {
        "Hours_Studied": hours_studied, "attendance": attendance, "Sleep_hours": sleep_hours,
        "Previous_Scores": previous_scores, "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,
        "Parental Involvement": parental_involvement, "Access_to_Resources": access_to_resources,
        "ExtracurricularActivities": extracurricular, "motivation_level": motivation_level,
        "Internet_Access": internet_access, "Family Income": family_income,
        "Teacher_Quality": teacher_quality, "School_Type": school_type,
        "Peer_Influence": peer_influence, "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education, "Distance_from_Home": distance_from_home,
        "Gender": gender,
    }

def result_card(pred, proba, key_prefix=""):
    css_class = "result-pass" if pred == "Pass" else "result-fail"
    icon = "✅" if pred == "Pass" else "⚠️"
    confidence = proba[pred] * 100 if proba else None
    fail_prob = proba.get("Fail", 0) if proba else 0
    risk_label, risk_css, risk_color = risk_band(fail_prob)
    conf_bar_html = f'<div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:{confidence:.0f}%"></div></div>' if confidence else ""
    st.markdown(f"""
    <div class="{css_class}">
        <h2>{icon} Predicted: {pred}</h2>
        <p>{'This student is expected to pass.' if pred == 'Pass' else 'This student is at risk of failing.'}</p>
        {f'<p style="font-size:1.4rem;font-weight:700;margin-top:0.5rem;position:relative;">{confidence:.1f}% confidence</p>' if confidence else ''}
        {conf_bar_html}
        <div class="risk-badge {risk_css}">⚡ {risk_label} of failing ({fail_prob*100:.0f}%)</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# SIDEBAR — GLOBAL CONTROLS
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>Student Performance</h2>
        <p>Analysis &amp; Pass/Fail Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    threshold = st.slider(
        "🎯 Pass/Fail threshold (Exam Score)", min_value=55, max_value=90, value=70, step=1,
        help="Students scoring at or above this value are labeled 'Pass'. All charts, models, and stats update live.",
    )

    model_options = ["Auto (Best Model)", "Logistic Regression", "Decision Tree", "Random Forest"]
    selected_model_choice = st.selectbox("🤖 Model used for prediction", model_options, index=0)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    bundle = train_all_models(threshold)
    df = bundle["data"]
    active_model = bundle["best_name"] if selected_model_choice == "Auto (Best Model)" else selected_model_choice

    pass_rate = (df["Result"] == "Pass").mean() * 100
    st.caption(f"Dataset: **{df.shape[0]:,}** students · **{df.shape[1]}** columns")
    st.caption(f"Current pass rate: **{pass_rate:.1f}%** (threshold = {threshold})")
    st.caption(f"Active model: **{active_model}**")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download cleaned dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_cleaned.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption("Built with Streamlit · Logistic Regression, Decision Tree & Random Forest")

# =====================================================================
# TOP-LEVEL TABS
# =====================================================================
tab_predict, tab_data, tab_insights, tab_lab, tab_findings = st.tabs(
    ["🔮 Predictor", "📋 Data Explorer", "📈 Insights", "🧪 Model Lab", "📝 Findings"]
)

# =====================================================================
# TAB 1 — PREDICTOR
# =====================================================================
with tab_predict:
    st.markdown("""
    <div class="app-header">
        <h1>🎓 Student Pass / Fail Predictor</h1>
        <p>Enter a student's profile to predict Pass/Fail, see why, and compare against the class average.</p>
    </div>
    """, unsafe_allow_html=True)

    fail_count = int((df["Result"] == "Fail").sum())
    pass_count = int((df["Result"] == "Pass").sum())
    avg_score = df["Exam_Score"].mean()

    qs1, qs2, qs3, qs4 = st.columns(4)
    for col_qs, icon_qs, label_qs, value_qs in zip(
        [qs1, qs2, qs3, qs4],
        ["🧑‍🎓", "✅", "⚠️", "📊"],
        ["Total Students", "Passing", "Failing", "Avg Exam Score"],
        [f"{len(df):,}", f"{pass_count:,}", f"{fail_count:,}", f"{avg_score:,.1f}"],
    ):
        col_qs.markdown(f"""<div class="metric-card">
            <div class="icon">{icon_qs}</div><div class="label">{label_qs}</div><div class="value">{value_qs}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    compare_mode = st.checkbox("🆚 Compare two students side-by-side", value=False)

    if not compare_mode:
        st.markdown('<div class="section-title">🧑‍🎓 Student Profile</div>', unsafe_allow_html=True)
        input_dict = student_input_form(suffix="_single")
        predict_btn = st.button("🔮 Predict Pass / Fail", key="predict_single")

        if predict_btn:
            pred, proba, row_encoded = predict_with(active_model, input_dict, bundle)
            st.session_state["last_pred"] = (pred, proba, row_encoded, input_dict)

        st.markdown("<br>", unsafe_allow_html=True)

        if "last_pred" in st.session_state:
            pred, proba, row_encoded, saved_input = st.session_state["last_pred"]
            res_col, exp_col = st.columns([1, 1.2])
            with res_col:
                result_card(pred, proba)
                if proba:
                    fig = go.Figure(go.Bar(
                        x=list(proba.values()), y=list(proba.keys()), orientation="h",
                        marker_color=["#1FAA59" if k == "Pass" else "#E23744" for k in proba.keys()],
                        text=[f"{v*100:.1f}%" for v in proba.values()], textposition="outside",
                        marker=dict(line=dict(width=0), cornerradius=6),
                    ))
                    fig.update_layout(height=130, margin=dict(l=10, r=40, t=10, b=10),
                                       xaxis=dict(range=[0, 1], title="Probability", showgrid=False),
                                       yaxis=dict(showgrid=False),
                                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width="stretch")
            with exp_col:
                grouped, is_local = explain_prediction(active_model, bundle, row_encoded)
                title = "🔍 Why this prediction" if is_local else "🔍 What this model weighs most heavily"
                st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
                colors = ["#E23744" if v > 0 else "#1FAA59" for v in grouped.values] if is_local else ["#3B6EF5"] * len(grouped)
                fig_exp = go.Figure(go.Bar(
                    x=grouped.values, y=grouped.index, orientation="h",
                    marker_color=colors, marker=dict(cornerradius=4),
                ))
                fig_exp.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                       xaxis=dict(title="Contribution toward Fail" if is_local else "Importance", showgrid=False),
                                       yaxis=dict(autorange="reversed"),
                                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_exp, width="stretch")
                if not is_local:
                    st.caption("Tree-based models don't give a simple per-student breakdown, so this shows the factors the model relies on overall.")

            st.markdown('<div class="section-title">📡 This Student vs. Class Average</div>', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(saved_input, df), width="stretch")
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 1.5rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🔮</div>
                <h3 style="color:#3B6EF5; margin:0;">Ready to Predict</h3>
                <p style="color:#7A8399; margin:0.5rem 0 0 0;">Fill in the student profile above and click <b>Predict</b> to see the result, an explanation, and a class comparison.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        colA, colB = st.columns(2)
        with colA:
            st.markdown('<div class="section-title">🧑‍🎓 Student A</div>', unsafe_allow_html=True)
            input_a = student_input_form(suffix="_a")
        with colB:
            st.markdown('<div class="section-title">🧑‍🎓 Student B</div>', unsafe_allow_html=True)
            input_b = student_input_form(suffix="_b")

        if st.button("🆚 Compare Students", key="predict_compare"):
            pred_a, proba_a, _ = predict_with(active_model, input_a, bundle)
            pred_b, proba_b, _ = predict_with(active_model, input_b, bundle)
            st.session_state["last_compare"] = (pred_a, proba_a, input_a, pred_b, proba_b, input_b)

        if "last_compare" in st.session_state:
            pred_a, proba_a, input_a, pred_b, proba_b, input_b = st.session_state["last_compare"]
            rc1, rc2 = st.columns(2)
            with rc1:
                result_card(pred_a, proba_a)
            with rc2:
                result_card(pred_b, proba_b)

            st.markdown('<div class="section-title">📊 Pass Probability Comparison</div>', unsafe_allow_html=True)
            fig_cmp = go.Figure(go.Bar(
                x=["Student A", "Student B"],
                y=[proba_a.get("Pass", 0) * 100, proba_b.get("Pass", 0) * 100],
                marker_color=["#3B6EF5", "#9B5EF5"],
                text=[f"{proba_a.get('Pass', 0)*100:.1f}%", f"{proba_b.get('Pass', 0)*100:.1f}%"],
                textposition="outside",
            ))
            fig_cmp.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                   yaxis=dict(title="Pass Probability (%)", range=[0, 105]),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cmp, width="stretch")

            st.markdown('<div class="section-title">📡 Profile Comparison</div>', unsafe_allow_html=True)
            fig_radar = go.Figure()
            for label, vals, color in [("Student A", input_a, "#3B6EF5"), ("Student B", input_b, "#9B5EF5")]:
                r = []
                for f in RADAR_FEATURES:
                    lo, hi = df[f].min(), df[f].max()
                    rng = (hi - lo) or 1
                    r.append((vals[f] - lo) / rng * 100)
                r += [r[0]]
                fig_radar.add_trace(go.Scatterpolar(r=r, theta=RADAR_FEATURES + [RADAR_FEATURES[0]],
                                                      fill='toself', name=label, line_color=color, opacity=0.7))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
                                     height=360, margin=dict(l=30, r=30, t=30, b=30),
                                     legend=dict(orientation="h", y=-0.1), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_radar, width="stretch")
        else:
            st.info("Fill in both student profiles above and click **Compare Students**.")

    st.markdown("<br>", unsafe_allow_html=True)
    best_acc = bundle["comparison"].iloc[0]["Accuracy"]
    st.markdown(f"""
    <div class="card">
        <div class="section-title"><span class="pulse-dot"></span> Model in Use</div>
        <p style="color:#555; margin:0;">Predicting with: <b style="color:#3B6EF5">{active_model}</b>
        {' <span style="color:#7A8399;">(auto-selected as best performer)</span>' if selected_model_choice == 'Auto (Best Model)' else ''}</p>
        <p style="color:#7A8399; font-size:0.85rem; margin:0.3rem 0 0 0;">Best model overall: <b>{bundle['best_name']}</b> · Accuracy: <b>{best_acc*100:.1f}%</b> · Change the model or threshold anytime from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# TAB 2 — DATA EXPLORER
# =====================================================================
with tab_data:
    st.markdown("""
    <div class="app-header">
        <h1>🗂️ Dataset Explorer</h1>
        <p>Browse, filter, and download the cleaned Student Performance dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, icon_m, label, value in zip(
        [m1, m2, m3, m4], ["🧑‍🎓", "📈", "📉", "⏱️"],
        ["Total Students", "Avg Exam Score", "Fail Rate",
         "Avg Hours Studied"],
        [f"{len(df):,}", f"{df['Exam_Score'].mean():.1f}",
         f"{(df['Result']=='Fail').mean()*100:.1f}%", f"{df['Hours_Studied'].mean():.1f} hrs/wk"],
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="icon">{icon_m}</div><div class="label">{label}</div><div class="value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Filters</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        involvement_f = st.multiselect("Parental Involvement", sorted(df["Parental Involvement"].unique()))
    with f2:
        school_f = st.multiselect("School Type", sorted(df["School_Type"].unique()))
    with f3:
        status_f = st.multiselect("Result", sorted(df["Result"].unique()))

    f4, f5 = st.columns(2)
    with f4:
        hours_range = st.slider("Hours Studied range", 0, int(df["Hours_Studied"].max()) or 1,
                                 (0, int(df["Hours_Studied"].max()) or 1))
    with f5:
        att_range = st.slider("Attendance range (%)", 0, 100, (0, 100))

    filtered = df.copy()
    if involvement_f:
        filtered = filtered[filtered["Parental Involvement"].isin(involvement_f)]
    if school_f:
        filtered = filtered[filtered["School_Type"].isin(school_f)]
    if status_f:
        filtered = filtered[filtered["Result"].isin(status_f)]
    filtered = filtered[filtered["Hours_Studied"].between(*hours_range)]
    filtered = filtered[filtered["attendance"].between(*att_range)]

    st.markdown(f'<div class="section-title">📋 Showing {len(filtered):,} of {len(df):,} records</div>', unsafe_allow_html=True)
    st.dataframe(filtered, width="stretch", height=380)

    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="student_filtered.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Filtered Data Overview</div>', unsafe_allow_html=True)
    dv1, dv2 = st.columns(2)
    with dv1:
        filt_status = filtered["Result"].value_counts().reset_index()
        filt_status.columns = ["Result", "Count"]
        fig_dv1 = px.pie(filt_status, names="Result", values="Count", hole=0.55,
                          color="Result", color_discrete_map={"Pass": "#1FAA59", "Fail": "#E23744"})
        fig_dv1.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                               title=dict(text="Pass vs Fail Split", font=dict(size=13, color="#1E2749")),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dv1, width="stretch")
    with dv2:
        fig_dv2 = px.histogram(filtered, x="Exam_Score", nbins=30, color_discrete_sequence=["#3B6EF5"], opacity=0.8)
        fig_dv2.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                               title=dict(text="Exam Score Distribution", font=dict(size=13, color="#1E2749")),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dv2, width="stretch")

# =====================================================================
# TAB 3 — INSIGHTS (VISUALIZATIONS)
# =====================================================================
with tab_insights:
    st.markdown("""
    <div class="app-header">
        <h1>📈 Visual Insights</h1>
        <p>Explore what drives exam performance across study habits and support factors.</p>
    </div>
    """, unsafe_allow_html=True)

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        st.markdown('<div class="section-title">Pass/Fail Distribution</div>', unsafe_allow_html=True)
        pie_data = df["Result"].value_counts().reset_index()
        pie_data.columns = ["Result", "Count"]
        fig = px.pie(pie_data, names="Result", values="Count", hole=0.5,
                     color="Result", color_discrete_map={"Pass": "#1FAA59", "Fail": "#E23744"})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, width="stretch")
    with row1c2:
        st.markdown('<div class="section-title">Hours Studied vs Exam Score</div>', unsafe_allow_html=True)
        sample = df.sample(min(1500, len(df)), random_state=1)
        fig = px.scatter(sample, x="Hours_Studied", y="Exam_Score", color="Result",
                          color_discrete_map={"Pass": "#1FAA59", "Fail": "#E23744"}, opacity=0.6)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, width="stretch")

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        st.markdown('<div class="section-title">Attendance vs Exam Score</div>', unsafe_allow_html=True)
        sample2 = df.sample(min(1500, len(df)), random_state=2)
        fig = px.scatter(sample2, x="attendance", y="Exam_Score", color="Result",
                          color_discrete_map={"Pass": "#1FAA59", "Fail": "#E23744"}, opacity=0.6)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, width="stretch")
    with row2c2:
        st.markdown('<div class="section-title">Fail Rate by Parental Involvement</div>', unsafe_allow_html=True)
        fail_rate = df.groupby("Parental Involvement")["Result"].apply(
            lambda s: (s == "Fail").mean() * 100
        ).reset_index(name="Fail Rate (%)").sort_values("Fail Rate (%)", ascending=False)
        fig = px.bar(fail_rate, x="Parental Involvement", y="Fail Rate (%)",
                     color="Fail Rate (%)", color_continuous_scale=["#35C77E", "#E23744"])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-title">🔗 Correlation Heatmap (Numeric Features)</div>', unsafe_allow_html=True)
    corr = df[NUMERIC + ["Exam_Score"]].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig_corr.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_corr, width="stretch")

    st.markdown('<div class="section-title">🧭 Build Your Own Scatter Plot</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        x_axis = st.selectbox("X-axis", NUMERIC, index=0)
    with sc2:
        y_axis = st.selectbox("Y-axis", NUMERIC + ["Exam_Score"], index=len(NUMERIC))
    sample3 = df.sample(min(2000, len(df)), random_state=3)
    fig_custom = px.scatter(sample3, x=x_axis, y=y_axis, color="Result",
                             color_discrete_map={"Pass": "#1FAA59", "Fail": "#E23744"}, opacity=0.6)
    fig_custom.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_custom, width="stretch")

    row3c1, row3c2 = st.columns(2)
    with row3c1:
        st.markdown('<div class="section-title">🏆 Top 5 Performance-Boosting Factors</div>', unsafe_allow_html=True)
        factor_scores = {}
        for col in ["Parental Involvement", "Access_to_Resources", "Teacher_Quality", "Family Income", "Peer_Influence"]:
            factor_scores[col] = df.groupby(col)["Exam_Score"].mean().max() - df.groupby(col)["Exam_Score"].mean().min()
        top5 = pd.Series(factor_scores).sort_values(ascending=False).head(5).reset_index()
        top5.columns = ["Factor", "Score Spread"]
        fig_top = px.bar(top5, x="Score Spread", y="Factor", orientation="h", color_discrete_sequence=["#1FAA59"])
        fig_top.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_top, width="stretch")
    with row3c2:
        st.markdown('<div class="section-title">📉 Fail Rate by Distance from Home</div>', unsafe_allow_html=True)
        dist_fail = df.groupby("Distance_from_Home")["Result"].apply(lambda s: (s == "Fail").mean() * 100).reset_index(name="Fail Rate (%)")
        fig_bot = px.bar(dist_fail, x="Distance_from_Home", y="Fail Rate (%)", color_discrete_sequence=["#E23744"])
        fig_bot.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bot, width="stretch")

# =====================================================================
# TAB 4 — MODEL LAB
# =====================================================================
with tab_lab:
    st.markdown("""
    <div class="app-header">
        <h1>🧪 Model Lab</h1>
        <p>Compare Logistic Regression, Decision Tree, and Random Forest side by side.</p>
    </div>
    """, unsafe_allow_html=True)

    comp_display = bundle["comparison"].copy()
    for c in ["Accuracy", "Fail Precision", "Fail Recall", "Fail F1-Score"]:
        comp_display[c] = (comp_display[c] * 100).round(2).astype(str) + "%"
    st.markdown('<div class="section-title">🤖 Metrics Comparison</div>', unsafe_allow_html=True)
    st.dataframe(comp_display, width="stretch", hide_index=True)
    st.success(f"🏆 Best model: **{bundle['best_name']}** (ranked by Fail F1-Score → Fail Recall → Accuracy)")

    comp_raw = bundle["comparison"].copy()
    fig_models = go.Figure()
    colors = ["#3B6EF5", "#1FAA59", "#F59B3B", "#9B5EF5"]
    for i, metric in enumerate(["Accuracy", "Fail Precision", "Fail Recall", "Fail F1-Score"]):
        fig_models.add_trace(go.Bar(name=metric, x=comp_raw["Model"], y=comp_raw[metric] * 100,
                                     marker_color=colors[i],
                                     text=[f"{v:.1f}%" for v in comp_raw[metric] * 100], textposition="outside"))
    fig_models.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(title="Score (%)", range=[0, 105]), legend=dict(orientation="h", y=1.15),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_models, width="stretch")

    y_test = bundle["y_test"]
    st.markdown('<div class="section-title">🧮 Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for cm_col, (name, info) in zip(cm_cols, bundle["results"].items()):
        cm = confusion_matrix(y_test, info["y_pred"], labels=["Pass", "Fail"])
        with cm_col:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                x=["Pred Pass", "Pred Fail"], y=["Actual Pass", "Actual Fail"])
            fig_cm.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                                  title=dict(text=name, font=dict(size=13, color="#1E2749")),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_cm, width="stretch")

    st.markdown('<div class="section-title">📉 ROC Curves</div>', unsafe_allow_html=True)
    y_test_bin = (y_test == "Fail").astype(int)
    fig_roc = go.Figure()
    for name, info in bundle["results"].items():
        fpr, tpr, _ = roc_curve(y_test_bin, info["y_proba_fail"])
        auc = roc_auc_score(y_test_bin, info["y_proba_fail"])
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                  line=dict(dash="dash", color="gray")))
    fig_roc.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           legend=dict(orientation="h", y=-0.2),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_roc, width="stretch")

    st.markdown('<div class="section-title">🌟 Feature Importance</div>', unsafe_allow_html=True)
    fi_model = st.radio("Model", ["Decision Tree", "Random Forest"], horizontal=True)
    model = bundle["final_models"][fi_model]["model"]
    importances = pd.Series(model.feature_importances_, index=bundle["encoded_columns"])
    grouped_fi = group_by_feature(importances, bundle["encoded_columns"]).sort_values(ascending=True)
    fig_fi = go.Figure(go.Bar(x=grouped_fi.values, y=grouped_fi.index, orientation="h", marker_color="#3B6EF5"))
    fig_fi.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_fi, width="stretch")

# =====================================================================
# TAB 5 — FINDINGS & CONCLUSION
# =====================================================================
with tab_findings:
    st.markdown("""
    <div class="app-header">
        <h1>📝 Findings & Conclusion</h1>
        <p>Key academic takeaways at the current pass/fail threshold.</p>
    </div>
    """, unsafe_allow_html=True)

    fail_rate_overall = (df["Result"] == "Fail").mean() * 100
    involvement_fail = df.groupby("Parental Involvement")["Result"].apply(lambda s: (s == "Fail").mean() * 100)
    worst_involvement = involvement_fail.idxmax()
    worst_involvement_rate = involvement_fail.max()
    best_school = df.groupby("School_Type")["Exam_Score"].mean().idxmax()
    worst_school = df.groupby("School_Type")["Exam_Score"].mean().idxmin()
    low_hours_df = df[df["Hours_Studied"] < 10]
    low_hours_fail_rate = (low_hours_df["Result"] == "Fail").mean() * 100 if len(low_hours_df) else 0
    high_hours_df = df[df["Hours_Studied"] >= 20]
    high_hours_fail_rate = (high_hours_df["Result"] == "Fail").mean() * 100 if len(high_hours_df) else 0
    best_teacher_quality = df.groupby("Teacher_Quality")["Exam_Score"].mean().idxmax()

    st.markdown('<div class="section-title">🔍 Key Findings</div>', unsafe_allow_html=True)
    kf1, kf2, kf3, kf4 = st.columns(4)
    kf1.markdown(f"""<div class="stat-badge red"><p class="stat-label">Overall Fail Rate</p><p class="stat-value">{fail_rate_overall:.1f}%</p></div>""", unsafe_allow_html=True)
    kf2.markdown(f"""<div class="stat-badge orange"><p class="stat-label">Highest-Risk Involvement Level</p><p class="stat-value">{worst_involvement}</p></div>""", unsafe_allow_html=True)
    kf3.markdown(f"""<div class="stat-badge green"><p class="stat-label">Best School Type</p><p class="stat-value">{best_school}</p></div>""", unsafe_allow_html=True)
    kf4.markdown(f"""<div class="stat-badge purple"><p class="stat-label">Best Teacher Quality</p><p class="stat-value">{best_teacher_quality}</p></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    - At a pass threshold of **{threshold}**, **{fail_rate_overall:.1f}%** of students in the dataset fail.
    - Students with **{worst_involvement}** parental involvement have the highest fail rate among involvement levels, at **{worst_involvement_rate:.1f}%**.
    - The **{best_school}** school type has the highest average exam score, while **{worst_school}** has the lowest.
    - Students studying **under 10 hours/week** fail at **{low_hours_fail_rate:.1f}%**, compared to **{high_hours_fail_rate:.1f}%** for those studying **20+ hours/week** — study time is strongly linked to passing.
    - **{best_teacher_quality}** teacher quality is associated with the highest average exam scores.
    """)

    disc1, disc2 = st.columns(2)
    with disc1:
        st.markdown('<div class="section-title">⚡ Study Time Impact on Fail Rate</div>', unsafe_allow_html=True)
        disc_compare = pd.DataFrame({"Study Time": ["< 10 hrs/week", "≥ 20 hrs/week"],
                                      "Fail Rate (%)": [low_hours_fail_rate, high_hours_fail_rate]})
        fig_disc = px.bar(disc_compare, x="Study Time", y="Fail Rate (%)", color="Study Time",
                           color_discrete_map={"< 10 hrs/week": "#E23744", "≥ 20 hrs/week": "#1FAA59"},
                           text="Fail Rate (%)")
        fig_disc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_disc.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_disc, width="stretch")
    with disc2:
        st.markdown('<div class="section-title">🏫 Average Score by School Type</div>', unsafe_allow_html=True)
        school_score = df.groupby("School_Type")["Exam_Score"].mean().reset_index().sort_values("Exam_Score", ascending=False)
        fig_reg = px.bar(school_score, x="School_Type", y="Exam_Score", color="Exam_Score",
                          color_continuous_scale=["#E23744", "#35C77E"], text="Exam_Score")
        fig_reg.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_reg.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_reg, width="stretch")

    st.markdown('<div class="section-title">📌 Conclusion</div>', unsafe_allow_html=True)
    st.markdown(f"""
    The analysis shows that **hours studied** and **attendance** are among the strongest drivers of exam
    performance — students who study more consistently and attend more regularly are considerably more
    likely to pass. Support factors like **parental involvement**, **access to resources**, and
    **teacher quality** also matter: students lacking these support structures fail at a noticeably higher rate.

    Among the three models tested, **{bundle['best_name']}** was selected as the top performer because it gives
    the best balance of correctly catching at-risk (Fail) students (recall) while keeping overall accuracy high.
    Use the **Model Lab** tab to inspect confusion matrices, ROC curves, and feature importance, and the
    **Predictor** tab to test individual students, see why the model made its call, and compare two profiles
    side by side.
    """)
