Requirements Document for SQL-Powered Predictive Sales Dashboard
1. Project Overview
A web-based sales analytics dashboard that segments customers using RFM analysis and predicts purchase likelihood, enabling the sales team to prioritize high-value customers.

2. Functional Requirements
2.1 Backend/Data Processing
    FR1: Extract and load sales transaction data into PostGres database with normalized schema (Customers, Transactions, Products tables)
    FR2: Calculate RFM scores (Recency, Frequency, Monetary) using SQL queries
    FR3: Segment customers into 4+ tiers (e.g., "Champions," "Loyal," "At-Risk," "Lost") based on RFM quantiles
    FR4: Train and expose a classification model that predicts next-month purchase probability
    FR5: Provide REST API endpoints to retrieve customer segments, RFM scores, and predictions
2.2 Frontend/Dashboard
    FR6: Display customer segmentation distribution (pie/bar chart)
    FR7: Show RFM metrics with interactive filters (by segment, date range, product category)
    FR8: Display model performance metrics (Accuracy, F1-Score, Confusion Matrix)
    FR9: Show individual customer details with RFM scores and predicted purchase likelihood

3. User Stories
AS A: Sales Manager
I WANT TO: See which customers are most likely to purchase next month
SO THAT: I can prioritize outreach and allocate resources efficiently

Acceptance Criteria:
- Customer list sorted by purchase probability (high to low)
- Filter by segment and date range
- Export customer list with contact info and scores

AS A: Data Analyst
I WANT TO: Validate that the RFM segmentation is working correctly
SO THAT: I can trust the model's predictions

Acceptance Criteria:
- View RFM score distributions in histograms
- See confusion matrix and F1-score
- Drill down into individual customer RFM calculations

4. Technical Stack
Layer:	Technology
Database:	PostgreSQL (Development and Production) 
Backend API:	Python (Flask/FastAPI) 
Data Processing:	Pandas, NumPy, Scikit-learn
Frontend:	React + Recharts (or similar)
Dashboard:	Streamlit (for MVP) or custom React

6. Data Model
customers (customer_id, customer_unique_id, customer_zip_code, customer_city, customer_state)
order_items (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
products (product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm)
rfm_scores (customer_id, last_purchase_date, recency_days, frequency, monetary, recency_score, frequency_score, monetary_score, overall_rfm_score, segment, calculated_date)

8. Constraints & Assumptions
Using publicly available dataset (Kaggle Online Retail)
No real-time data ingestion (batch updates sufficient)

10. Implementation Phases
Phase 1 (Local Development):
- Build dashboard with SQLite
- Develop RFM segmentation and ML model
- Test locally with sample data

