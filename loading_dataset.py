import pandas as pd
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np

# 1. Setup Connection
# Make sure these credentials match your docker-compose.yml!
engine = create_engine('postgresql://byu_student:your_secure_password@localhost:5432/olist_sales')

# 2. Path to your data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, 'init-db', 'data')

# 3. Loop through every file in that folderclear
for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        # Create a clean table name (e.g., 'olist_orders_dataset' becomes 'orders')
        table_name = filename.replace('olist_', '').replace('_dataset', '').replace('.csv', '')
        
        # Read the CSV
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        
        # Push to Postgres
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
print("Successfully uploaded all datasets!")

# Now calculate RFM features
rfm_sql = """
-- Calculate Recency (days since last purchase), Frequency (orders), Monetary (spend)
WITH customer_orders AS (
    SELECT
        o.customer_id,
        DATE(MAX(o.order_purchase_timestamp)) AS last_purchase_date,
        COUNT(DISTINCT o.order_id) AS order_count,
        SUM(oi.price + oi.freight_value) AS total_spend
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY o.customer_id
)
SELECT
    customer_id,
    last_purchase_date,
    CAST(EXTRACT(EPOCH FROM (NOW() - last_purchase_date)) / 86400 AS INTEGER) AS recency_days,
    order_count AS frequency,
    total_spend AS monetary
FROM customer_orders;
"""

try:
    rfm_df = pd.read_sql(rfm_sql, con=engine)
    rfm_df.to_sql('rfm_features', con=engine, if_exists='replace', index=False)
    print("RFM features calculated and saved!")
    print("Column names:", rfm_df.columns.tolist())

    rfm_df['R_Segment'] = pd.cut(rfm_df['recency_days'], bins=4, labels=['Best', 'Good', 'Average', 'Lost'])

    # Segment Frequency
    rfm_df['F_Segment'] = pd.cut(rfm_df['frequency'], bins=3, labels=['Low', 'Medium', 'High'])

    # Segment Monetary
    rfm_df['M_Segment'] = pd.cut(rfm_df['monetary'], bins=3, labels=['Low', 'Medium', 'High'])

    # Create binary target: 1 if customer is a repeat customer, 0 otherwise
    median_frequency = rfm_df['frequency'].median()
    median_monetary = rfm_df['monetary'].median()
    rfm_df['will_purchase'] = ((rfm_df['frequency'] >= median_frequency) & (rfm_df['monetary'] >= median_monetary)).astype(int)

    # ==================== STEP 2.2: MODELING WITH SCIKIT-LEARN ====================
    print("\n" + "="*60)
    print("STEP 2.2: MODELING WITH SCIKIT-LEARN")
    print("="*60 + "\n")
    
    # Prepare Data: Select features (X) and target (Y)
    # Features: RFM scores + segmented groups
    feature_columns = ['R_Segment', 'F_Segment', 'M_Segment']
    X = rfm_df[feature_columns].copy()
    X = pd.get_dummies(X)  # Convert categories to numeric
    Y = rfm_df['will_purchase'].copy()
    
    # Remove any rows with NaN values
    valid_indices = X.notna().all(axis=1) & Y.notna()
    X = X[valid_indices]
    Y = Y[valid_indices]
    
    print(f"Total customers: {len(X)}")
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{Y.value_counts()}\n")
    
    # Train-Test Split: 80% training, 20% testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}\n")
    
    # Train Model: Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, Y_train)
    
    # Make predictions
    Y_pred_rf = rf_model.predict(X_test)
    
    # Evaluate Random Forest
    print("\n" + "-"*60)
    print("RANDOM FOREST RESULTS")
    print("-"*60)
    rf_accuracy = accuracy_score(Y_test, Y_pred_rf)
    rf_f1 = f1_score(Y_test, Y_pred_rf)
    rf_cm = confusion_matrix(Y_test, Y_pred_rf)
    
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"F1-Score: {rf_f1:.4f}")
    print(f"\nConfusion Matrix:\n{rf_cm}")
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred_rf, target_names=['No Purchase', 'Will Purchase']))
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance)
    
    # Train Model: Logistic Regression (for comparison)
    print("\n" + "-"*60)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, Y_train)
    
    # Make predictions
    Y_pred_lr = lr_model.predict(X_test)
    
    # Evaluate Logistic Regression
    print("\n" + "-"*60)
    print("LOGISTIC REGRESSION RESULTS")
    print("-"*60)
    lr_accuracy = accuracy_score(Y_test, Y_pred_lr)
    lr_f1 = f1_score(Y_test, Y_pred_lr)
    lr_cm = confusion_matrix(Y_test, Y_pred_lr)
    
    print(f"Accuracy: {lr_accuracy:.4f}")
    print(f"F1-Score: {lr_f1:.4f}")
    print(f"\nConfusion Matrix:\n{lr_cm}")
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred_lr, target_names=['No Purchase', 'Will Purchase']))
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"Random Forest - Accuracy: {rf_accuracy:.4f}, F1-Score: {rf_f1:.4f}")
    print(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, F1-Score: {lr_f1:.4f}")
    
    # Save models and predictions
    rfm_df['RF_Prediction'] = np.nan
    rfm_df.loc[valid_indices, 'RF_Prediction'] = rf_model.predict(X)
    
    rfm_df['LR_Prediction'] = np.nan
    rfm_df.loc[valid_indices, 'LR_Prediction'] = lr_model.predict(X)
    
    rfm_df.to_sql('rfm_predictions', con=engine, if_exists='replace', index=False)
    print("\nPredictions saved to 'rfm_predictions' table!")

except Exception as e:
    print("Failed to compute RFM or model:", e)
    import traceback
    traceback.print_exc()


