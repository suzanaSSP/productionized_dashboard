import pandas as pd
import os
from sqlalchemy import create_engine

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
        
        print(f"Uploading {filename} to table '{table_name}'...")
        
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
    print(rfm_df.head())
except Exception as e:
    print("Failed to compute RFM:", e)

# Segment Recency into 4 groups
rfm_df['R_Segment'] = pd.cut(rfm_df['recency_days'], bins=4, labels=['Best', 'Good', 'Average', 'Lost'])

# Segment Frequency
rfm_df['F_Segment'] = pd.cut(rfm_df['frequency'], bins=3, labels=['Low', 'Medium', 'High'])

# Segment Monetary
rfm_df['M_Segment'] = pd.cut(rfm_df['monetary'], bins=3, labels=['Low', 'Medium', 'High'])

# Create binary target: 1 if customer purchased in last 30 days, 0 otherwise
threshold = rfm_df['recency_days'].quantile(0.25)
rfm_df['will_purchase'] = (rfm_df['recency_days'] <= threshold).astype(int)

print(f"Most recent purchase: {rfm_df['last_purchase_date'].max()}")

print(rfm_df[['customer_id', 'recency_days', 'will_purchase']].head(10))
print(rfm_df['will_purchase'].value_counts())
# Shows how many customers are 1 vs 0