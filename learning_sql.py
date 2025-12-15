import pandas as pd
import sqlite3

sales_df = pd.read_csv('Retail_Transactions_Dataset.csv')

# Connect to the database (creates the file if it doesn't exist)
conn = sqlite3.connect('sales.db')
cursor = conn.cursor()

customers_df = sales_df[['Customer_Name','Customer_Category', 'City']].copy()
customers_df = customers_df.drop_duplicates(subset=['Customer_Name']) 
customers_df[Customer_ID] = customer_df.index + 1
customers_df = customers_df[['Customer_ID', 'Customer_Name', 'Customer_Category', 'City']]
sales_df = pd.merge(
    sales_df, 
    customers_df, 
    on=['Customer_Name', 'Customer_Category', 'City'], 
    how='left'
)

products_df = sales_df[['Product', 'Store_Type', 'Promotion']]
products_df[Product_ID] = products_df.index + 1
products_df = products_df[['Product_ID', 'Product', 'Store_Type', 'Promotion']]

sales_df = pd.merge(
    sales_df, 
    products_df, 
    on=['Product', 'Store_Type', 'Promotion'], 
    how='left'
)

transactions_df = sales_df[['Transaction_ID', 'Date', 'Total_Items', 'Total_Cost', 'Payment_Method', 'Discount_Applied']].copy()
transactions_df = transactions_df.drop_duplicates(subset=['Transaction_ID'])

transactions_df.to_sql('Transactions', conn, if_exists='replace', index=False)
customers_df.to_sql('Customers', conn, if_exists='replace', index=False)
products_df.to_sql('Products', conn, if_exists='replace', index=False)

