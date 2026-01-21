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
customers (customer_id, name, email, signup_date, country)
transactions (transaction_id, customer_id, product_id, invoice_date, quantity, amount)
products (product_id, product_name, category, price)
rfm_scores (customer_id, recency, frequency, monetary, segment, prediction)

7. Success Metrics
✅ Dashboard loads in <3 seconds
✅ Model achieves F1-Score ≥ 0.70
✅ Segmentation has clear business interpretation
✅ Sales team can act on insights within 5 minutes of viewing
✅ AWS deployment completes with zero downtime
✅ Monthly AWS costs stay within $50 budget (Free Tier optimized)

8. Constraints & Assumptions
Using publicly available dataset (Kaggle Online Retail)
SQLite for local development, AWS RDS PostgreSQL for production
Initial MVP targets internal use only
No real-time data ingestion (batch updates sufficient)
AWS Free Tier eligible services prioritized for cost optimization
Data stored in AWS S3 with appropriate retention policies

9. AWS Cloud Architecture
9.1 Compute: AWS EC2 (t3.micro) or Lambda for serverless API
9.2 Database: AWS RDS PostgreSQL (db.t3.micro) with automated backups
9.3 Storage: AWS S3 for data exports, model artifacts, and static assets
9.4 Monitoring: AWS CloudWatch for logs, metrics, and alarms
9.5 Security: VPC configuration, Security Groups, IAM roles with least privilege
9.6 Cost Management: AWS Cost Explorer and budgets to monitor spending

10. Implementation Phases
Phase 1 (Local Development):
- Build dashboard with SQLite
- Develop RFM segmentation and ML model
- Test locally with sample data

Phase 2 (AWS Deployment):
- Set up AWS account and configure IAM
- Create RDS PostgreSQL instance and migrate data
- Deploy API to EC2/Lambda
- Configure S3 buckets for storage
- Set up CloudWatch monitoring
- Implement CI/CD pipeline
- Conduct load testing and security audit
