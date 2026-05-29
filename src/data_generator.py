"""
Fraud Detection Data Generator
Creates realistic financial transaction data with fraud patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

# Configuration
N_TRANSACTIONS = 100000
FRAUD_RATE = 0.0017  # 0.17% fraud rate (realistic)

MERCHANT_CATEGORIES = [
    'grocery', 'gas_station', 'restaurant', 'online_retail',
    'electronics', 'travel', 'entertainment', 'healthcare',
    'utilities', 'cash_advance'
]

# Fraud patterns
FRAUD_PATTERNS = {
    'high_amount': 0.3,      # High value transaction
    'unusual_location': 0.25, # Far from home
    'rapid_succession': 0.25, # Multiple quick transactions
    'unusual_time': 0.15,     # Late night
    'cash_advance': 0.05      # Cash advance (high risk)
}


def generate_legitimate_transaction(user_id: int, timestamp: datetime) -> dict:
    """Generate a legitimate transaction."""
    # Amount: log-normal distribution centered around $50
    amount = np.random.lognormal(mean=3.9, sigma=1.0)
    amount = round(min(max(amount, 1), 5000), 2)
    
    # Merchant category
    category_weights = [0.25, 0.15, 0.15, 0.15, 0.08, 0.05, 0.07, 0.05, 0.03, 0.02]
    merchant = np.random.choice(MERCHANT_CATEGORIES, p=category_weights)
    
    # Time features
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    # Distance from home (most transactions are close)
    distance_home = np.random.exponential(scale=5)  # Most within 5 miles
    distance_home = round(min(distance_home, 100), 1)
    
    # Distance from last transaction
    distance_last = np.random.exponential(scale=3)
    distance_last = round(min(distance_last, 50), 1)
    
    # Time since last transaction (minutes)
    time_since_last = np.random.exponential(scale=180)  # ~3 hours average
    time_since_last = round(min(time_since_last, 1440), 0)  # Max 24 hours
    
    # Velocity features
    velocity_1h = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    velocity_24h = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
    
    return {
        'user_id': user_id,
        'timestamp': timestamp,
        'amount': amount,
        'merchant_category': merchant,
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'distance_from_home': distance_home,
        'distance_from_last': distance_last,
        'time_since_last': time_since_last,
        'is_weekend': int(day_of_week >= 5),
        'is_night': int(hour < 6 or hour >= 22),
        'velocity_1h': velocity_1h,
        'velocity_24h': velocity_24h,
        'is_fraud': 0
    }


def generate_fraudulent_transaction(user_id: int, timestamp: datetime) -> dict:
    """Generate a fraudulent transaction with fraud patterns."""
    # Choose fraud pattern
    pattern = np.random.choice(
        list(FRAUD_PATTERNS.keys()),
        p=list(FRAUD_PATTERNS.values())
    )
    
    txn = generate_legitimate_transaction(user_id, timestamp)
    
    # Apply fraud patterns
    if pattern == 'high_amount':
        txn['amount'] = round(np.random.uniform(500, 3000), 2)
        
    elif pattern == 'unusual_location':
        txn['distance_from_home'] = round(np.random.uniform(50, 500), 1)
        txn['distance_from_last'] = round(np.random.uniform(30, 200), 1)
        
    elif pattern == 'rapid_succession':
        txn['velocity_1h'] = np.random.randint(3, 8)
        txn['velocity_24h'] = np.random.randint(8, 20)
        txn['time_since_last'] = np.random.randint(1, 10)
        
    elif pattern == 'unusual_time':
        txn['hour_of_day'] = np.random.choice([0, 1, 2, 3, 4, 5, 23])
        txn['is_night'] = 1
        
    elif pattern == 'cash_advance':
        txn['merchant_category'] = 'cash_advance'
        txn['amount'] = round(np.random.uniform(200, 1000), 2)
    
    # Add some noise to make fraud less obvious
    if np.random.random() < 0.3:
        txn['distance_from_home'] *= np.random.uniform(1.5, 3)
    if np.random.random() < 0.3:
        txn['amount'] *= np.random.uniform(1.2, 2)
    
    txn['is_fraud'] = 1
    
    return txn


def generate_transactions(n_transactions: int = N_TRANSACTIONS) -> pd.DataFrame:
    """Generate full transaction dataset."""
    print(f"🔄 Generating {n_transactions:,} transactions...")
    
    n_fraud = int(n_transactions * FRAUD_RATE)
    n_legit = n_transactions - n_fraud
    
    print(f"   Legitimate: {n_legit:,} ({n_legit/n_transactions*100:.2f}%)")
    print(f"   Fraudulent: {n_fraud:,} ({n_fraud/n_transactions*100:.2f}%)")
    
    transactions = []
    start_date = datetime(2023, 1, 1)
    n_users = 5000
    
    # Generate legitimate transactions
    for i in range(n_legit):
        user_id = np.random.randint(1, n_users + 1)
        days_offset = np.random.uniform(0, 365)
        hours_offset = np.random.uniform(0, 24)
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset)
        
        txn = generate_legitimate_transaction(user_id, timestamp)
        txn['transaction_id'] = f'TXN{i+1:08d}'
        transactions.append(txn)
    
    # Generate fraudulent transactions
    for i in range(n_fraud):
        user_id = np.random.randint(1, n_users + 1)
        days_offset = np.random.uniform(0, 365)
        hours_offset = np.random.uniform(0, 24)
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset)
        
        txn = generate_fraudulent_transaction(user_id, timestamp)
        txn['transaction_id'] = f'TXN{n_legit+i+1:08d}'
        transactions.append(txn)
    
    df = pd.DataFrame(transactions)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reorder columns
    cols = ['transaction_id', 'user_id', 'timestamp', 'amount', 'merchant_category',
            'hour_of_day', 'day_of_week', 'distance_from_home', 'distance_from_last',
            'time_since_last', 'is_weekend', 'is_night', 'velocity_1h', 'velocity_24h', 'is_fraud']
    df = df[cols]
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Fraud rate: {df['is_fraud'].mean()*100:.3f}%")
    print(f"   Avg transaction: ${df['amount'].mean():.2f}")
    print(f"   Avg fraud amount: ${df[df['is_fraud']==1]['amount'].mean():.2f}")
    
    return df


def main():
    """Generate and save data."""
    df = generate_transactions()
    
    # Save files
    project_root = Path(__file__).parent.parent
    
    # Save to raw
    raw_dir = project_root / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / 'transactions.csv', index=False)
    
    # Save to sample
    sample_dir = project_root / 'data' / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(sample_dir / 'transactions.csv', index=False)
    
    print(f"\n💾 Saved data to:")
    print(f"   {raw_dir / 'transactions.csv'}")
    print(f"   {sample_dir / 'transactions.csv'}")
    
    print("\n✅ Data generation complete!")
    
    return df


if __name__ == '__main__':
    main()
