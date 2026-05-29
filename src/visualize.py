"""
Fraud Detection Visualization
Generate plots for model analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

plt.style.use('seaborn-v0_8-whitegrid')


def plot_class_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot class distribution (fraud vs legitimate)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    fraud_counts = df['is_fraud'].value_counts()
    colors = ['#27ae60', '#e74c3c']
    labels = ['Legitimate', 'Fraud']
    
    axes[0].bar(labels, [fraud_counts[0], fraud_counts[1]], color=colors)
    axes[0].set_ylabel('Number of Transactions')
    axes[0].set_title('Transaction Distribution', fontweight='bold')
    
    for i, count in enumerate([fraud_counts[0], fraud_counts[1]]):
        axes[0].text(i, count + 500, f'{count:,}', ha='center', fontsize=10)
    
    # Pie chart (log scale representation)
    axes[1].pie([fraud_counts[0], fraud_counts[1]], labels=labels, colors=colors,
                autopct='%1.2f%%', explode=[0, 0.1], startangle=90)
    axes[1].set_title('Class Imbalance', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, save_path: str = None):
    """Plot feature distributions by fraud status."""
    features = ['amount', 'distance_from_home', 'velocity_1h', 'hour_of_day']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        for fraud_val, color, label in [(0, '#27ae60', 'Legitimate'), (1, '#e74c3c', 'Fraud')]:
            data = df[df['is_fraud'] == fraud_val][feature]
            axes[i].hist(data, bins=30, alpha=0.6, color=color, label=label, density=True)
        
        axes[i].set_xlabel(feature.replace('_', ' ').title())
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution', fontweight='bold')
        axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm: list, save_path: str = None):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm_array = np.array(cm)
    
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'], ax=ax)
    
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def plot_feature_importance(feature_importance: dict, save_path: str = None):
    """Plot feature importance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, color=colors[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(importance):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def plot_fraud_by_hour(df: pd.DataFrame, save_path: str = None):
    """Plot fraud rate by hour of day."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    hourly_stats = df.groupby('hour_of_day').agg({
        'is_fraud': ['sum', 'count']
    })
    hourly_stats.columns = ['fraud_count', 'total']
    hourly_stats['fraud_rate'] = hourly_stats['fraud_count'] / hourly_stats['total'] * 100
    
    hours = hourly_stats.index
    ax.bar(hours, hourly_stats['fraud_rate'], color='#e74c3c', alpha=0.7)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Fraud Rate (%)')
    ax.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    
    # Highlight night hours
    ax.axvspan(22, 24, alpha=0.2, color='gray', label='Night hours')
    ax.axvspan(0, 6, alpha=0.2, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()


def generate_all_plots():
    """Generate all visualization plots."""
    print("\n📊 Generating visualizations...")
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'docs' / 'img'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(project_root / 'data' / 'raw' / 'transactions.csv')
    
    # Load metrics
    model_dir = project_root / 'models'
    
    with open(model_dir / 'metrics.json', 'r') as f:
        metrics = json.load(f)
    
    with open(model_dir / 'feature_importance.json', 'r') as f:
        feature_importance = json.load(f)
    
    # 1. Class distribution
    plot_class_distribution(df, output_dir / 'class_distribution.png')
    
    # 2. Feature distributions
    plot_feature_distributions(df, output_dir / 'feature_distributions.png')
    
    # 3. Confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], output_dir / 'confusion_matrix.png')
    
    # 4. Feature importance
    plot_feature_importance(feature_importance, output_dir / 'feature_importance.png')
    
    # 5. Fraud by hour
    plot_fraud_by_hour(df, output_dir / 'fraud_by_hour.png')
    
    print(f"\n✅ All visualizations saved to {output_dir}")


def main():
    generate_all_plots()


if __name__ == "__main__":
    main()
