from flask import Blueprint, jsonify, request
from services.database import get_db_connection
from datetime import datetime

customer_bp = Blueprint('customers', __name__, url_prefix='/api/customers')

@customer_bp.route('/segments', methods=['GET'])
def get_segments():
    """Get customer segmentation summary"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT segment, COUNT(*) as count 
        FROM rfm_scores 
        GROUP BY segment
        """
        cursor.execute(query)
        segments = {row[0]: row[1] for row in cursor.fetchall()}
        
        query_total = "SELECT COUNT(*) FROM customers"
        cursor.execute(query_total)
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'segments': segments,
            'total_customers': total,
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@customer_bp.route('/rfm', methods=['GET'])
def get_rfm_scores():
    """Get RFM scores with optional filters"""
    try:
        segment = request.args.get('segment')
        limit = request.args.get('limit', default=100, type=int)
        sort_by = request.args.get('sort_by', default='recency_days')

        # Whitelist sortable columns to avoid SQL injection
        allowed_sort = {'recency_days', 'frequency', 'monetary', 'overall_rfm_score'}
        if sort_by not in allowed_sort:
            sort_by = 'recency_days'
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        base_query = """
        SELECT c.customer_id, r.recency_days, r.frequency, 
               r.monetary, r.segment, r.overall_rfm_score
        FROM rfm_scores r
        JOIN customers c ON r.customer_id = c.customer_id
        """
        
        params = []
        if segment:
            base_query += " WHERE r.segment = %s"
            params.append(segment)
        
        # Psycopg2 uses %s for parameters
        base_query += f" ORDER BY r.{sort_by} DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        data = [
            {
                'customer_id': row[0],
                'recency': row[1],
                'frequency': row[2],
                'monetary': row[3],
                'segment': row[4],
                'rfm_score': row[5]
            }
            for row in rows
        ]
        
        conn.close()
        
        return jsonify({'data': data, 'total': len(data)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@customer_bp.route('/<customer_id>/details', methods=['GET'])
def get_customer_details(customer_id):
    """Get individual customer details with RFM and predictions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get customer info
        query = "SELECT customer_id, customer_city, customer_state FROM customers WHERE customer_id = %s"
        cursor.execute(query, (customer_id,))
        customer = cursor.fetchone()
        
        if not customer:
            return jsonify({'error': 'Customer not found'}), 404
        
        # Get RFM scores
        query = """
        SELECT recency_days, frequency, monetary, segment, overall_rfm_score
        FROM rfm_scores WHERE customer_id = %s
        """
        cursor.execute(query, (customer_id,))
        rfm = cursor.fetchone()
        
        # Get predictions
        query = """
        select rp."LR_Prediction", rs.customer_id  
        from rfm_predictions rp 
        join rfm_scores rs on rs.frequency = rp.frequency  
        where rs.customer_id = %s   
        """
        cursor.execute(query, (customer_id,))
        prediction = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            'customer_id': customer[0],
            'customer_city': customer[1],
            'customer_state': customer[2],
            'rfm': {
                'recency': rfm[0] if rfm else None,
                'frequency': rfm[1] if rfm else None,
                'monetary': rfm[2] if rfm else None,
                'segment': rfm[3] if rfm else None,
                'overall_score': rfm[4] if rfm else None
            },
            'prediction': {
                'Algorithm Prediction': prediction[0] if prediction else None,

            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500