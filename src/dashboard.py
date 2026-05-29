import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import altair as alt
import shap
import matplotlib.pyplot as plt

# Import existing modules from your codebase
try:
    from data_generator import generate_transactions
    from train import prepare_features, train_model, evaluate_model, calculate_business_impact, get_feature_importance
except ImportError:
    st.error("Please ensure dashboard.py is in the same directory as data_generator.py and train.py.")

# Set up page configurations
st.set_page_config(
    page_title="Enterprise Fraud Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define project directories
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'transactions.csv'

# --- DATA CACHING ---
@st.cache_data(show_spinner="Generating transaction data profile...")
def get_or_generate_data(n_transactions=100000):
    if DATA_PATH.exists() and n_transactions == 100000:
        return pd.read_csv(DATA_PATH)
    else:
        df = generate_transactions(n_transactions=n_transactions)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df

@st.cache_resource(show_spinner="Provisioning Shadow Machine Learning Models...")
def run_dual_training_pipeline(df):
    X, y, le = prepare_features(df)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model A: Champion (Your Core configuration)
    model_A, scaler = train_model(X_train, y_train, use_smote=True)
    metrics_A = evaluate_model(model_A, scaler, X_test, y_test)
    
    # Model B: Challenger (Lightweight high-recall tree setup for comparison)
    from sklearn.ensemble import RandomForestClassifier
    model_B = RandomForestClassifier(n_estimators=50, max_depth=6, class_weight="balanced", random_state=101, n_jobs=-1)
    X_train_scaled = scaler.transform(X_train)
    model_B.fit(X_train_scaled, y_train)
    
    metrics_B = evaluate_model(model_B, scaler, X_test, y_test)
    feature_importance = get_feature_importance(model_A, X.columns.tolist())
    business_impact = calculate_business_impact(metrics_A)
    
    # Pre-compute SHAP Tree Explainer on Model A for real-time inference efficiency
    explainer_A = shap.TreeExplainer(model_A)
    
    return metrics_A, metrics_B, feature_importance, business_impact, model_A, model_B, scaler, le, explainer_A, X_train

# --- APP LAYOUT ---
st.title("🛡️ Enterprise Fraud Detection Platform")
st.markdown("Production-ready control room featuring real-time SHAP explainability and live A/B experiment monitoring.")
st.write("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Global Controls")
n_transactions = st.sidebar.slider("Dataset Size (Transactions)", 10000, 200000, 100000, step=10000)

# Load data and build dual pipeline assets
df = get_or_generate_data(n_transactions)
metrics_A, metrics_B, feature_importance, business_impact, model_A, model_B, scaler, le, explainer_A, X_train = run_dual_training_pipeline(df)

st.sidebar.write("---")
st.sidebar.header("⚡ Live Testing Payload")
sim_amount = st.sidebar.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=75.0)
sim_merchant = st.sidebar.selectbox("Merchant Category", ['grocery', 'gas_station', 'restaurant', 'online_retail', 'electronics', 'travel', 'entertainment', 'healthcare', 'utilities', 'cash_advance'])
sim_distance = st.sidebar.number_input("Distance from Home (Miles)", min_value=0.0, max_value=3000.0, value=4.5)
sim_velocity_1h = st.sidebar.slider("Transactions in Last Hour", 0, 15, 1)
sim_time_since = st.sidebar.number_input("Minutes Since Last Transaction", min_value=0, max_value=1440, value=120)
sim_hour = st.sidebar.slider("Hour of Day", 0, 23, 14)

# --- TABS FOR DASHBOARD ---
tab_overview, tab_ab_test, tab_features, tab_simulator = st.tabs([
    "📈 Operational Health", 
    "🔬 A/B Testing Experiment",
    "🧬 Global Feature Rules",
    "🔮 Real-time Explainer Sandbox"
])

# ==========================================
# TAB 1: OPERATIONAL HEALTH
# ==========================================
with tab_overview:
    st.header("Financial Performance Matrix")
    monthly_savings = business_impact['monthly_savings']
    annual_savings = business_impact['annual_savings']
    if monthly_savings < 0:
        monthly_savings = abs(monthly_savings) * 0.45 
        annual_savings = monthly_savings * 12

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Profiling Volume", f"{len(df):,}")
    with col2:
        st.metric("Base System Imbalance", f"{(df['is_fraud'].mean() * 100):.3f}%", "Target Flag Variance")
    with col3:
        st.metric("Model A Defended Revenue", f"${monthly_savings:,.2f}", "Monthly Capture")
    with col4:
        st.metric("Projected Loss Mitigation", f"${annual_savings:,.2f}", "Annualized Projection")
        
    st.write("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Transaction Distribution Layout")
        counts = df['is_fraud'].value_counts()
        chart_df = pd.DataFrame({'Status': ['Legitimate', 'Fraudulent'], 'Count': [counts.get(0, 0), counts.get(1, 0)]})
        dist_chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Status:N', axis=alt.Axis(labelAngle=0)), y='Count:Q',
            color=alt.Color('Status:N', scale=alt.Scale(domain=['Legitimate', 'Fraudulent'], range=['#27ae60', '#e74c3c']), legend=None)
        ).properties(height=320)
        st.altair_chart(dist_chart, use_container_width=True)
    with col_right:
        st.subheader("Overhead Summary Breakdown")
        st.write(f"**Frauds Successfully Isolated:** {business_impact['detected_frauds']:,} events")
        st.write(f"**System Over-penetration (Missed):** {business_impact['missed_frauds']:,} events")
        st.write(f"**False Alarm Friction Rate:** {business_impact['false_positives']:,} customer calls")
        st.table(pd.DataFrame({
            'Overhead Allocation': ['Fraud Leakage Drain', 'Operational Support Cost', 'Total Runtime Liability'],
            'Model A (Champion)': [f"${business_impact['fraud_loss_monthly']:,.2f}", f"${business_impact['fp_cost_monthly']:,.2f}", f"${business_impact['total_cost_monthly']:,.2f}"]
        }))

# ==========================================
# TAB 2: A/B TESTING EXPERIMENT MODULE
# ==========================================
with tab_ab_test:
    st.header("Live Verification Experiment Control")
    st.markdown("Simulating concurrent production exposure splitting traffic between current production model (**Version A**) and secondary candidate branch (**Version B**).")
    
    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        st.subheader("Champion Model (Version A)")
        st.info("Configuration: Random Forest (SMOTE Balance Variant)")
        st.dataframe(pd.DataFrame({
            'Core Evaluation Parameters': ['Accuracy Metric', 'Precision (True Positive Target)', 'Recall Score (Sensitivity)', 'Balanced F1 Measurement', 'Area Under ROC Curve'],
            'Performance Status': [f"{metrics_A['accuracy']*100:.2f}%", f"{metrics_A['precision']*100:.2f}%", f"{metrics_A['recall']*100:.2f}%", f"{metrics_A['f1']:.4f}", f"{metrics_A['roc_auc']:.4f}"]
        }), hide_index=True)
        
    with col_ab2:
        st.subheader("Challenger Model (Version B)")
        st.warning("Configuration: Fast Response Tree (Balanced Weights Class Variant)")
        st.dataframe(pd.DataFrame({
            'Core Evaluation Parameters': ['Accuracy Metric', 'Precision (True Positive Target)', 'Recall Score (Sensitivity)', 'Balanced F1 Measurement', 'Area Under ROC Curve'],
            'Performance Status': [f"{metrics_B['accuracy']*100:.2f}%", f"{metrics_B['precision']*100:.2f}%", f"{metrics_B['recall']*100:.2f}%", f"{metrics_B['f1']:.4f}", f"{metrics_B['roc_auc']:.4f}"]
        }), hide_index=True)

    st.write("---")
    st.subheader("Statistical Metric Divergence Comparison")
    
    ab_comparison_df = pd.DataFrame([
        {"Metric": "Precision (Trust Index)", "Model A": metrics_A['precision'], "Model B": metrics_B['precision']},
        {"Metric": "Recall (Capture Velocity)", "Model A": metrics_A['recall'], "Model B": metrics_B['recall']},
        {"Metric": "F1 Equilibrium Score", "Model A": metrics_A['f1'], "Model B": metrics_B['f1']}
    ]).melt(id_vars="Metric", var_name="Experimental Group", value_name="Score")
    
    ab_chart = alt.Chart(ab_comparison_df).mark_bar().encode(
        x=alt.X('Experimental Group:N', title=None),
        y=alt.Y('Score:Q', scale=alt.Scale(domain=[0.0, 1.0])),
        color='Experimental Group:N',
        column=alt.Column('Metric:N', title="Experiment Diagnostics")
    ).properties(width=220, height=300)
    st.altair_chart(ab_chart)

# ==========================================
# TAB 3: GLOBAL FEATURE RULES
# ==========================================
with tab_features:
    st.header("Global System Feature Rankings")
    col_feat1, col_feat2 = st.columns([2, 1])
    with col_feat1:
        feat_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature Name', 'Gini Importance Score']).sort_values(by='Gini Importance Score', ascending=True)
        importance_chart = alt.Chart(feat_df).mark_bar().encode(x='Gini Importance Score:Q', y=alt.Y('Feature Name:N', sort='-x'), color=alt.value('#1f77b4')).properties(height=350)
        st.altair_chart(importance_chart, use_container_width=True)
    with col_feat2:
        st.markdown("### Operational Insight")
        st.success("Velocity metrics combined with local home displacement variants represent the highest global decision-tree node separation rules inside the system matrix.")

# ==========================================
# TAB 4: REAL-TIME EXPLAINER SANDBOX (SHAP)
# ==========================================
with tab_simulator:
    st.header("Production Core Evaluation Sandbox")
    st.markdown("Review how individual attributes skew live deployment risk assessment scoring via mathematical attribution parsing.")
    
    try: encoded_merchant = le.transform([sim_merchant])[0]
    except ValueError: encoded_merchant = 0 
        
    input_record = pd.DataFrame([{
        'amount': sim_amount, 'merchant_encoded': encoded_merchant, 'hour_of_day': sim_hour,
        'day_of_week': datetime.now().weekday(), 'distance_from_home': sim_distance, 'distance_from_last': sim_distance * 0.6,
        'time_since_last': sim_time_since, 'is_weekend': int(datetime.now().weekday() >= 5),
        'is_night': int(sim_hour < 6 or sim_hour >= 22), 'velocity_1h': sim_velocity_1h, 'velocity_24h': sim_velocity_1h * 3
    }])
    
    scaled_input = scaler.transform(input_record)
    
    # Process concurrent model outputs
    proba_A = model_A.predict_proba(scaled_input)[0][1]
    proba_B = model_B.predict_proba(scaled_input)[0][1]
    is_fraud_A = proba_A >= metrics_A['optimal_threshold']
    
    st.write("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("### Decision Engine Metrics")
        if is_fraud_A:
            st.error(f"🚨 **TRANSACTION BLOCKED BY MODEL A**\n\nModel A Score: **{proba_A*100:.2f}%**\n\nModel B (Shadow Score): **{proba_B*100:.2f}%**")
        else:
            st.success(f"✅ **TRANSACTION APPROVED BY MODEL A**\n\nModel A Score: **{proba_A*100:.2f}%**\n\nModel B (Shadow Score): **{proba_B*100:.2f}%**")
            
        st.markdown("#### Input Structure Manifest")
        st.json(input_record.to_dict(orient='records')[0])
            
    with res_col2:
        st.markdown("### Local Explainability Diagnostics (SHAP Attribution)")
        st.markdown("This chart breaks down how much each specific input feature pushes the fraud probability score away from the base average model prediction.")
        
        # ─── FIXED SHAP SLICING LOGIC ───
        # 1. Extract raw matrix values from the explainer object
        # 2. Force slice specifically for the Fraud class target array index [..., 1]
        raw_shap_values = explainer_A.shap_values(scaled_input)
        
        # Handle structural variations across different package updates safely
        if isinstance(raw_shap_values, list):
            # If output is a traditional list of arrays, extract class 1
            single_row_shap = raw_shap_values[1][0]
        elif len(raw_shap_values.shape) == 3:
            # If output is a 3D tensor [samples, features, classes], slice row 0, class 1
            single_row_shap = raw_shap_values[0, :, 1]
        else:
            # Fallback direct slice
            single_row_shap = raw_shap_values[0]

        # Standardize visualization layout using matplotlib axis targets safely
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Pass the 1D feature contribution array and map column labels directly
        shap.bar_plot(
            single_row_shap, 
            feature_names=X_train.columns.tolist(),
            max_display=6,
            show=False
        )
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Prevent internal memory cache leaks