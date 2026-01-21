# Implementation Plan
## SQL-Powered Predictive Sales Dashboard

---

## Sprint 1: Database Setup (Week 1)
**Goal:** Create database schema and load data

### Tasks:
- [✅ ] **Task 1.1:** Download Kaggle Online Retail dataset
- [✅ ] **Task 1.2:** Connect sales_data.db to docker and Postgressql
- [✅ ] **Task 1.3:** Write SQL schema for 5 tables (customers, transactions, products, rfm_scores, predictions)
- [ ✅] **Task 1.4:** Load CSV data into database using Pandas `.to_sql()`
- [✅ ] **Task 1.5:** Create indexes on `customer_id`, `invoice_date`, `segment`
- [ ✅] **Task 1.6:** Test queries: `SELECT * FROM customers LIMIT 10`

**Deliverables:**
- docker connection
- `sales_data.db` populated with data
- `load_data.py` script

---

## Sprint 2: ETL Pipeline & RFM Calculation (Week 2)
**Goal:** Calculate RFM scores and segment customers

### Tasks:
- [ ✅] **Task 2.1:** Write SQL query to calculate Recency, Frequency, Monetary
- [✅ ] **Task 2.2:** Create Python class `ETLPipeline` in `etl_pipeline.py`
- [ ✅] **Task 2.3:** Implement `calculate_rfm()` method
- [✅ ] **Task 2.4:** Implement `segment_customers()` using quantiles
- [ ✅] **Task 2.5:** Load RFM scores into `rfm_scores` table
- [ ✅] **Task 2.6:** Verify segmentation: Count customers in each segment

**Deliverables:**
- `etl_pipeline.py` script
- `rfm_scores` table populated
- Validation: 4+ segments with reasonable distribution

---

## Sprint 3: Machine Learning Model (Week 3)
**Goal:** Train and evaluate purchase prediction model

### Tasks:
- [✅ ] **Task 3.1:** Create `ml_model.py` with `PurchasePredictor` class
- [✅ ] **Task 3.2:** Prepare training data (features: RFM scores, target: next-month purchase)
- [✅ ] **Task 3.3:** Train Random Forest Classifier using Scikit-learn
- [✅ ] **Task 3.4:** Calculate metrics: Accuracy, F1-Score, Confusion Matrix
- [✅ ] **Task 3.5:** Save predictions to `predictions` table
- [ ✅] **Task 3.6:** Save trained model as `model.pkl`

**Deliverables:**
- `ml_model.py` script
- `model.pkl` file
- `predictions` table populated
- Model achieves F1-Score ≥ 0.70

---

## Sprint 4: Backend API (Week 4)
**Goal:** Build Flask REST API

### Tasks:
- [✅] **Task 4.1:** Create `backend/app.py` with Flask setup
- [✅ ] **Task 4.2:** Implement `GET /api/customers/segments`
- [✅] **Task 4.3:** Implement `GET /api/customers/rfm` with filters
- [✅] **Task 4.4:** Implement `GET /api/customers/{id}/details`
- [✅ ] **Task 4.5:** Implement `GET /api/model/metrics`
- [✅ ] **Task 4.6:** Implement `POST /api/export/customers` (CSV download)
- [✅ ] **Task 4.7:** Test all endpoints using Postman or curl

**Deliverables:**
- `backend/app.py` running on `localhost:5000`
- 5 working API endpoints
- API response time < 500ms

---

## Sprint 5: Frontend Dashboard (Week 5-6)
**Goal:** Build React dashboard

### Tasks:
- [✅] **Task 5.1:** Create React app: `npx create-react-app frontend`
- [ ] **Task 5.2:** Create `services/api.js` with fetch functions
- [ ] **Task 5.3:** Build `Dashboard.jsx` component
- [ ] **Task 5.4:** Build `SegmentationChart.jsx` (pie/bar chart using Recharts)
- [ ] **Task 5.5:** Build `RFMMetrics.jsx` (table with filters)
- [ ] **Task 5.6:** Build `ModelMetrics.jsx` (display accuracy, F1, confusion matrix)
- [ ] **Task 5.7:** Build `CustomerDetails.jsx` (individual customer view)
- [ ] **Task 5.8:** Build `ExportButton.jsx` (CSV download)
- [ ] **Task 5.9:** Style dashboard with CSS
- [ ] **Task 5.10:** Test: Dashboard loads in < 3 seconds

**Deliverables:**
- React app running on `localhost:3000`
- 5+ interactive components
- Meets all FR6-FR10 from Requirements Doc

---

## Sprint 6: Testing & Documentation (Week 7)
**Goal:** Finalize project for portfolio

### Tasks:
- [ ] **Task 6.1:** Write unit tests for API endpoints
- [ ] **Task 6.2:** Write README.md with setup instructions
- [ ] **Task 6.3:** Create demo video showing dashboard features
- [ ] **Task 6.4:** Take screenshots for portfolio
- [ ] **Task 6.5:** Document lessons learned
- [ ] **Task 6.6:** Push code to GitHub

**Deliverables:**
- Complete GitHub repository
- README with screenshots
- 2-3 minute demo video

---

## Sprint 7: AWS Cloud Deployment (Week 8-9)
**Goal:** Deploy application to AWS production environment

### Tasks:
- [ ] **Task 7.1:** Set up AWS account and configure billing alerts
- [ ] **Task 7.2:** Create IAM users and roles with least privilege access
- [ ] **Task 7.3:** Set up AWS RDS PostgreSQL instance (db.t3.micro)
- [ ] **Task 7.4:** Migrate SQLite data to RDS using `pg_dump` and data migration scripts
- [ ] **Task 7.5:** Update backend connection strings to use RDS endpoint
- [ ] **Task 7.6:** Create S3 bucket for static assets and CSV exports
- [ ] **Task 7.7:** Configure S3 bucket policies and lifecycle rules
- [ ] **Task 7.8:** Deploy backend API to AWS EC2 (t3.micro) or Lambda
- [ ] **Task 7.9:** Configure Security Groups (allow HTTP/HTTPS, restrict SSH)
- [ ] **Task 7.10:** Set up CloudWatch alarms for CPU, memory, and API errors
- [ ] **Task 7.11:** Configure CloudWatch Logs for application logging
- [ ] **Task 7.12:** Create GitHub Actions CI/CD pipeline for automated deployment
- [ ] **Task 7.13:** Perform load testing using Apache JMeter or Locust
- [ ] **Task 7.14:** Conduct security audit (IAM policies, encryption at rest/transit)
- [ ] **Task 7.15:** Document AWS architecture and deployment process

**Deliverables:**
- Backend API running on AWS EC2/Lambda with public endpoint
- RDS PostgreSQL database with migrated data
- S3 bucket configured for exports
- CloudWatch monitoring and alarms active
- CI/CD pipeline functional
- Deployment documentation in README
- Monthly AWS cost < $50 (Free Tier optimized)

---

## Timeline Overview

```
Week 1: Database    ████████░░░░░░░░░░░░░░░░░░░░░░
Week 2: ETL         ░░░░░░░░████████░░░░░░░░░░░░░░
Week 3: ML          ░░░░░░░░░░░░░░░░████░░░░░░░░░░
Week 4: API         ░░░░░░░░░░░░░░░░░░░░██░░░░░░░░
Week 5-6: UI        ░░░░░░░░░░░░░░░░░░░░░░████████
Week 7: Docs        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██
Week 8-9: AWS       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████
```

---

## Success Criteria (From Requirements Doc)

- ✅ Dashboard loads in < 3 seconds
- ✅ API responds in < 500ms
- ✅ Model F1-Score ≥ 0.70
- ✅ 5 REST endpoints functional
- ✅ Interactive filters working
- ✅ CSV export working
- ✅ GitHub repo with README
- ✅ AWS deployment with zero downtime
- ✅ CloudWatch monitoring active
- ✅ Monthly AWS costs < $50

---

## Risk Management

| Risk | Mitigation |
|---|---|
| Dataset quality issues | Clean data in Sprint 1, validate early |
| Model underfitting | Try different algorithms (Logistic Regression, XGBoost) |
| React not connecting to API | Test API with Postman first, then integrate |
| Performance issues | Use indexes, pagination, caching |
| AWS cost overruns | Set up billing alarms, use Free Tier eligible services |
| RDS migration failures | Test migration on smaller dataset first, keep SQLite backup |
| Security vulnerabilities | Regular security audits, principle of least privilege for IAM |

---

## Next Steps

1. Create a GitHub repository
2. Start with Sprint 1, Task 1.1
3. Check off tasks as you complete them
4. Update this plan if priorities change
5. After Sprint 6, begin AWS deployment planning