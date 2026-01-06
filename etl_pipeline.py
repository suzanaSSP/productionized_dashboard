import pandas as pd
from datetime import datetime
import logging
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self):
        # Connect to PostgreSQL instead of SQLite
        self.engine = create_engine('postgresql://byu_student:your_secure_password@localhost:5432/olist_sales')
    
    def load_csv(self, filepath):
        """EXTRACT: Load raw CSV data"""
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath)
    
    def calculate_rfm(self):
        """TRANSFORM: Calculate Recency, Frequency, Monetary using SQL"""
        logger.info("Calculating RFM scores...")
        query = """
        -- Calculate Recency (days since last purchase), Frequency (orders), Monetary (spend)
        WITH customer_orders AS (
            SELECT
                o.customer_id,
                DATE(MAX(o.order_purchase_timestamp::timestamp)) AS last_purchase_date,
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
        
        rfm_df = pd.read_sql_query(query, self.engine)
        return rfm_df
    
    def segment_customers(self, rfm_df):
        """TRANSFORM: Segment customers into 4 tiers using quantiles"""
        logger.info("Segmenting customers...")
        
        rfm_df['frequency'] = rfm_df['frequency'].fillna(0)
        rfm_df['monetary'] = rfm_df['monetary'].fillna(0)

        rfm_df['recency_score'] = pd.qcut(rfm_df['recency_days'], 4, labels=[4,3,2,1], duplicates='drop')
        rfm_df['frequency_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 4, labels=[1,2,3,4], duplicates='drop')
        rfm_df['monetary_score'] = pd.qcut(rfm_df['monetary'], 4, labels=[1,2,3,4], duplicates='drop')
        
        rfm_df['overall_rfm_score'] = (
            rfm_df['recency_score'].astype(int) + 
            rfm_df['frequency_score'].astype(int) + 
            rfm_df['monetary_score'].astype(int)
        )
        
        # Define segments based on RFM score
        def assign_segment(score):
            if score >= 10:
                return 'Champions'
            elif score >= 8:
                return 'Loyal'
            elif score >= 5:
                return 'At-Risk'
            else:
                return 'Lost'
        
        rfm_df['segment'] = rfm_df['overall_rfm_score'].apply(assign_segment)
        return rfm_df
    
    def load_to_database(self, rfm_df):
        """LOAD: Load RFM scores into database"""
        logger.info("Loading RFM scores to database...")
        rfm_df['calculated_date'] = datetime.now().date()
        rfm_df.to_sql('rfm_scores', self.engine, if_exists='replace', index=False)
      
    
    def run_pipeline(self):
        """Execute full ETL pipeline"""
        logger.info("Starting ETL pipeline...")
        rfm_df = self.calculate_rfm()
        rfm_df = self.segment_customers(rfm_df)
        self.load_to_database(rfm_df)
        logger.info("✅ ETL Pipeline completed successfully")
        
        # Print summary
        print(f"\nSegmentation Summary:")
        print(rfm_df['segment'].value_counts())
        self.graph()

    def graph(self):
        """Optional: Graphing function to visualize RFM segments"""
        
        rfm_df = pd.read_sql_table('rfm_scores', self.engine)
        
        plt.figure(figsize=(10,6))
        plt.bar(rfm_df['segment'].value_counts().index, rfm_df['segment'].value_counts().values)
        plt.title('Customer Segments Distribution')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.show()

# Usage
if __name__ == '__main__':
    pipeline = ETLPipeline()
    pipeline.run_pipeline()