-- Create the Orders Table
CREATE TABLE IF NOT EXISTS olist_orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT,
    order_status TEXT,
    order_purchase_timestamp TIMESTAMP
);

-- Import from CSV
COPY olist_orders 
FROM '/docker-entrypoint-initdb.d/data/olist_orders_dataset.csv' 
DELIMITER ',' 
CSV HEADER;

-- Create the Items Table
CREATE TABLE IF NOT EXISTS olist_order_items (
    order_id TEXT,
    order_item_id INT,
    product_id TEXT,
    price DECIMAL,
    freight_value DECIMAL,
    FOREIGN KEY (order_id) REFERENCES olist_orders(order_id)
);

-- Import order items
COPY olist_order_items 
FROM '/docker-entrypoint-initdb.d/data/olist_order_items_dataset.csv' 
DELIMITER ',' 
CSV HEADER;

-- Create Customers Table (optional but helpful for RFM segmentation)
CREATE TABLE IF NOT EXISTS olist_customers (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT,
    customer_zip_code_prefix TEXT,
    customer_city TEXT,
    customer_state TEXT
);

-- Import customers
COPY olist_customers 
FROM '/docker-entrypoint-initdb.d/data/olist_customers_dataset.csv' 
DELIMITER ',' 
CSV HEADER;