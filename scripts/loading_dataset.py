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
        
        # Read the CSV
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        
        # Push to Postgres
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
print("Successfully uploaded all datasets!")




