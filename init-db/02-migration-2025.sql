-- 1. Create a temporary variable to find the 'Time Gap'
-- It calculates the difference between Today and the newest 2018 order
DO $$
DECLARE
    time_gap INTERVAL;
BEGIN
    SELECT (CURRENT_DATE - MAX(order_purchase_timestamp)) INTO time_gap 
    FROM olist_orders;

    -- 2. Shift every order timestamp forward by that gap
    -- This preserves the exact spacing between orders but brings them to 2025
    UPDATE olist_orders
    SET order_purchase_timestamp = order_purchase_timestamp + time_gap;

    -- 3. If you have other tables (like payments or reviews), shift them too
    UPDATE olist_order_payments SET payment_installments_date = payment_installments_date + time_gap;
    UPDATE olist_order_reviews SET review_creation_date = review_creation_date + time_gap;
END $$;