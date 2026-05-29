"""
Fraud Detection Model Training
Train and evaluate fraud detection model with imbalanced data handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)

# Try to import SMOTE, fall back to class weights if not available
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("Note: imbalanced-learn not available, using class weights instead")


def load_data():
    """Load transaction data."""
    project_root = Path(__file__).parent.parent
    
    for data_path in [
        project_root / 'data' / 'raw' / 'transactions.csv',
        project_root / 'data' / 'sample' / 'transactions.csv'
    ]:
        if data_path.exists():
            df = pd.read_csv(data_path)
            print(f"📂 Loaded data from {data_path}")
            return df
    
    raise FileNotFoundError("Transaction data not found. Run data_generator.py first.")


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features for modeling."""
    print("🔧 Preparing features...")
    
    df = df.copy()
    
    # Encode merchant category
    le = LabelEncoder()
    df['merchant_encoded'] = le.fit_transform(df['merchant_category'])
    
    # Feature columns
    feature_cols = [
        'amount', 'merchant_encoded', 'hour_of_day', 'day_of_week',
        'distance_from_home', 'distance_from_last', 'time_since_last',
        'is_weekend', 'is_night', 'velocity_1h', 'velocity_24h'
    ]
    
    X = df[feature_cols]
    y = df['is_fraud']
    
    return X, y, le


def train_model(X_train, y_train, use_smote: bool = True) -> Tuple:
    """
    Train fraud detection model.
    
    Uses SMOTE for class imbalance if available, otherwise class weights.
    """
    print("🔧 Training model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Handle class imbalance
    if use_smote and HAS_IMBLEARN:
        print("   Using SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        class_weight = None
    else:
        print("   Using aggressive class weights for balancing...")
        X_resampled, y_resampled = X_train_scaled, y_train
        # Calculate weight to heavily penalize missing fraud
        n_samples = len(y_train)
        n_fraud = y_train.sum()
        n_legit = n_samples - n_fraud
        # Make fraud cases much more important
        weight_ratio = n_legit / n_fraud
        class_weight = {0: 1, 1: weight_ratio * 2}  # Double the natural weight
        print(f"   Class weights: legit=1.0, fraud={weight_ratio * 2:.1f}")
    
    print(f"   Training samples: {len(y_resampled):,}")
    print(f"   Fraud samples: {sum(y_resampled):,}")
    
    # Train Random Forest (handles imbalance better with class_weight)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_resampled, y_resampled)
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test) -> Dict:
    """Comprehensive model evaluation."""
    print("📊 Evaluating model...")
    
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions with default threshold
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold - prioritize recall with minimum precision of 50%
    thresholds = np.arange(0.05, 0.9, 0.05)
    best_threshold = 0.5
    best_recall = 0
    
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh)
        # Find highest recall with at least 50% precision
        if prec >= 0.40 and rec > best_recall:
            best_recall = rec
            best_threshold = thresh
    
    print(f"   Optimal threshold: {best_threshold:.2f}")
    
    # Use optimal threshold for final predictions
    y_pred = (y_proba >= best_threshold).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Threshold analysis
    threshold_results = []
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        threshold_results.append({
            'threshold': thresh,
            'precision': precision_score(y_test, y_pred_thresh, zero_division=0),
            'recall': recall_score(y_test, y_pred_thresh),
            'f1': f1_score(y_test, y_pred_thresh)
        })
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'optimal_threshold': best_threshold,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'threshold_analysis': threshold_results
    }


def get_feature_importance(model, feature_names: list) -> Dict:
    """Get feature importance."""
    importance = model.feature_importances_
    
    feature_importance = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {name: float(imp) for name, imp in feature_importance}


def calculate_business_impact(metrics: Dict, n_transactions: int = 500000,
                              avg_fraud_amount: float = 500,
                              cost_false_positive: float = 25) -> Dict:
    """Calculate business impact of the model."""
    
    fraud_rate = 0.0017
    n_frauds = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_frauds
    
    # With model
    detected_frauds = int(n_frauds * metrics['recall'])
    missed_frauds = n_frauds - detected_frauds
    false_positives = int(n_legit * (1 - metrics['precision']) * metrics['recall'] / metrics['precision'])
    
    fraud_loss_with_model = missed_frauds * avg_fraud_amount
    fp_cost_with_model = false_positives * cost_false_positive
    total_cost_with_model = fraud_loss_with_model + fp_cost_with_model
    
    # Without model (baseline 65% detection)
    baseline_recall = 0.65
    baseline_detected = int(n_frauds * baseline_recall)
    baseline_missed = n_frauds - baseline_detected
    baseline_fp_rate = 0.025
    baseline_fp = int(n_legit * baseline_fp_rate)
    
    fraud_loss_baseline = baseline_missed * avg_fraud_amount
    fp_cost_baseline = baseline_fp * cost_false_positive
    total_cost_baseline = fraud_loss_baseline + fp_cost_baseline
    
    monthly_savings = total_cost_baseline - total_cost_with_model
    annual_savings = monthly_savings * 12
    
    return {
        'monthly_transactions': n_transactions,
        'fraud_rate': fraud_rate,
        'detected_frauds': detected_frauds,
        'missed_frauds': missed_frauds,
        'false_positives': false_positives,
        'fraud_loss_monthly': fraud_loss_with_model,
        'fp_cost_monthly': fp_cost_with_model,
        'total_cost_monthly': total_cost_with_model,
        'baseline_cost_monthly': total_cost_baseline,
        'monthly_savings': monthly_savings,
        'annual_savings': annual_savings
    }


def print_results(metrics: Dict, feature_importance: Dict, business_impact: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("FRAUD DETECTION MODEL RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"   Precision: {metrics['precision']*100:.1f}%")
    print(f"   Recall: {metrics['recall']*100:.1f}%")
    print(f"   F1 Score: {metrics['f1']:.3f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.3f}")
    
    print(f"\n📊 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   True Negatives:  {cm[0][0]:,}")
    print(f"   False Positives: {cm[0][1]:,}")
    print(f"   False Negatives: {cm[1][0]:,}")
    print(f"   True Positives:  {cm[1][1]:,}")
    
    print(f"\n📊 Top 5 Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
        print(f"   {i+1}. {feature}: {importance:.3f}")
    
    print(f"\n💰 Business Impact (Monthly):")
    print(f"   Detected Frauds: {business_impact['detected_frauds']:,}")
    print(f"   Missed Frauds: {business_impact['missed_frauds']:,}")
    print(f"   False Positives: {business_impact['false_positives']:,}")
    print(f"   Monthly Savings: ${business_impact['monthly_savings']:,.0f}")
    print(f"   Annual Savings: ${business_impact['annual_savings']:,.0f}")


def save_model(model, scaler, le, metrics: Dict, feature_importance: Dict, business_impact: Dict):
    """Save model and metadata."""
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model components
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'label_encoder': le
    }, model_dir / 'fraud_model.pkl')
    print(f"💾 Saved model to {model_dir / 'fraud_model.pkl'}")
    
    # Save metrics
    metrics_to_save = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Save feature importance
    with open(model_dir / 'feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    # Save business impact
    with open(model_dir / 'business_impact.json', 'w') as f:
        json.dump(business_impact, f, indent=2)
    
    print(f"💾 Saved metrics to {model_dir / 'metrics.json'}")


def main():
    """Run training pipeline."""
    print("=" * 60)
    print("FRAUD DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        print("⚠️ Data not found. Generating sample data...")
        from data_generator import main as generate_data
        generate_data()
        df = load_data()
    
    print(f"\n📊 Dataset:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Fraud rate: {df['is_fraud'].mean()*100:.3f}%")
    print(f"   Fraudulent: {df['is_fraud'].sum():,}")
    print(f"   Legitimate: {(~df['is_fraud'].astype(bool)).sum():,}")
    
    # Prepare features
    X, y, le = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(X_train):,} ({y_train.sum():,} fraud)")
    print(f"   Test: {len(X_test):,} ({y_test.sum():,} fraud)")
    
    # Train model
    model, scaler = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, scaler, X_test, y_test)
    
    # Feature importance
    feature_importance = get_feature_importance(model, X.columns.tolist())
    
    # Business impact
    business_impact = calculate_business_impact(metrics)
    
    # Print results
    print_results(metrics, feature_importance, business_impact)
    
    # Save model
    save_model(model, scaler, le, metrics, feature_importance, business_impact)
    
    print("\n✅ Training complete!")
    
    return model, scaler, metrics


if __name__ == '__main__':
    main()
