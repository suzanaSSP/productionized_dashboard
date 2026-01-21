# Technical Specification Document
## SQL-Powered Predictive Sales Dashboard

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│         Dashboard UI + Customer Segmentation View        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Backend API (Flask/FastAPI)                 │
│    /api/segments, /api/rfm, /api/predictions            │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    ┌────────┐  ┌─────────┐  ┌──────────┐
    │Postgres│  │ Pandas  │  │ Scikit   │
    │   DB   │  │ Process │  │  Model   │
    └────────┘  └─────────┘  └──────────┘
```

---

## 2. Database Design

### 2.1 PostgreSQL

```sql
-- Create the Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_unique_id VARCHAR(50),
    customer_zip_code_prefix INT,
    customer_city TEXT,
    customer_state TEXT
);

-- Create the Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    order_status TEXT,
    order_purchase_timestamp TIMESTAMP,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP
);

-- Create the Order Items table (the bridge between orders and products)
CREATE TABLE IF NOT EXISTS order_items (
    order_id VARCHAR(50),
    order_item_id INT,
    product_id VARCHAR(50),
    seller_id VARCHAR(50),
    shipping_limit_date TIMESTAMP,
    price DECIMAL(10, 2),
    freight_value DECIMAL(10, 2)
);

-- RFM Scores Table (pre-calculated for performance)
CREATE TABLE rfm_scores (
    customer_id TEXT PRIMARY KEY,
    last_purchased_date DATE,
    recency_days INTEGER,
    frequency INTEGER,
    monetary REAL,
    recency_score INTEGER,
    frequency_score INTEGER,
    monetary_score INTEGER,
    overall_rfm_score INTEGER,
    segment TEXT,
    calculated_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Model Predictions Table
CREATE TABLE predictions (
    customer_id TEXT PRIMARY KEY,
    purchase_probability REAL,
    predicted_label INTEGER,
    confidence REAL,
    prediction_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create indexes for performance
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(invoice_date);
CREATE INDEX idx_rfm_segment ON rfm_scores(segment);
```

---

## 3. Backend API Specification

### 3.1 REST Endpoints

#### **GET /api/customers/segments**
Retrieve customer segmentation summary

**Response:**
```json
{
  "segments": {
    "Champions": 150,
    "Loyal": 320,
    "At-Risk": 180,
    "Lost": 250
  },
  "total_customers": 900,
  "last_updated": "2025-12-31"
}
```

---

#### **GET /api/customers/rfm**
Get RFM scores with optional filters

**Query Parameters:**
- `segment` (optional): Filter by segment (Champions, Loyal, At-Risk, Lost)
- `limit` (optional): Number of records (default: 100)
- `sort_by` (optional): Sort by recency, frequency, or monetary

**Response:**
```json
{
  "data": [
    {
      "customer_id": "C001",
      "recency": 5,
      "frequency": 25,
      "monetary": 3500.50,
      "segment": "Champions",
      "rfm_score": 555
    }
  ],
  "total": 900,
  "page": 1
}
```

---

#### **GET /api/customers/{customer_id}/details**
Get individual customer profile with predictions

**Path Parameters:**
- `customer_id` (required): Customer ID (e.g., C001)

**Response:**
```json
{
  "customer_id": "C001",
  "city": "campinas",
  "rfm": {
    "recency": 5,
    "frequency": 25,
    "monetary": 3500.50,
    "segment": "Champions"
  },
  "prediction": {
    "purchase_probability": 0.92,
    "predicted_next_month": true,
    "confidence": 0.88
  },
  "transaction_history": [
    {
      "date": "2025-12-20",
      "amount": 250.00,
      "products": ["Product A", "Product B"]
    }
  ]
}
```

---

#### **GET /api/model/metrics**
Retrieve model performance metrics

**Response:**
```json
{
  "accuracy": 0.78,
  "f1_score": 0.75,
  "precision": 0.82,
  "recall": 0.70,
  "confusion_matrix": {
    "true_positive": 450,
    "true_negative": 280,
    "false_positive": 95,
    "false_negative": 75
  },
  "model_trained_date": "2025-12-31",
  "training_samples": 900
}
```
---

### 3.2 API Implementation (Flask Example)

---

## 4. Data Processing Pipeline

### 4.1 ETL Flow

---

## 5. Machine Learning Model

### 5.1 Model Architecture

---

## 6. Frontend (React) Specification

### 6.1 Component Structure

```
src/
├── components/
│   ├── Dashboard.jsx         (Main layout)
│   ├── SegmentationChart.jsx (Pie/Bar chart)
│   ├── RFMMetrics.jsx        (Table with filters)
│   ├── CustomerDetails.jsx   (Individual customer view)
│   ├── ModelMetrics.jsx      (Performance dashboard)
│   └── ExportButton.jsx      (CSV export)
├── services/
│   └── api.js                (API calls to backend)
├── App.jsx
└── index.js
```
---

## 7. Deployment Architecture

### 7.1 Development Environment

- **Database:** PostgreSQL (local file: `data/sales_data.db`)
- **Backend API:** Flask on `localhost:5000`
- **Frontend:** React dev server on `localhost:3000`
- **Tools:** VS Code, Git

### 7.2 Production Environment (Future)

- **Database:** PostgreSQL on cloud
- **Frontend:** React build deployed to CDN
- **CI/CD:** GitHub Actions for automated testing and deployment

---


## 8. Error Handling

| Error Type | HTTP Code | Response Format |
|---|---|---|
| Invalid customer_id | 404 | `{"error": "Customer not found"}` |
| Database connection failure | 500 | `{"error": "Internal server error"}` |
| Missing required parameters | 400 | `{"error": "segment parameter required"}` |
| Model not trained | 503 | `{"error": "Model unavailable"}` |

---

## 9. Testing Strategy (Future)

### Unit Tests
- API endpoint response validation
- ETL pipeline RFM calculation accuracy
- ML model prediction logic

### Integration Tests
- Frontend → Backend API communication
- Database query performance
- CSV export functionality
- End-to-end dashboard workflow

### Performance Tests
- API response time < 500ms
- Dashboard load time < 3 seconds
- Handle 10,000+ customer records without degradation
---

## 10. Dependencies

### Backend (requirements.txt)
```
Flask==2.3.0
Flask-CORS==4.0.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.0
sqlite3
python-dotenv==1.0.0
```

### Frontend (package.json)
```
react==18.2.0
recharts==2.5.0
axios==1.3.0
```

---

**Document Version:** 2.0  
**Last Updated:** January 20, 2026  
**Status:** Ready for Implementation
