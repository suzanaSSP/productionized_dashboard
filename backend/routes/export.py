from flask import Blueprint, jsonify, send_file, request
from services.database import get_db_connection
import csv
import io
from datetime import datetime

export_bp = Blueprint('export', __name__, url_prefix='/api/export')

@export_bp.route('/customers', methods=['POST'])
def export_customers():
    """Export customer data with RFM scores to CSV"""
    try:
        data = request.json or {}
        segment = data.get('segment')  # Optional filter
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT c.customer_id, c.customer_city, c.customer_state, 
               r.recency_days, r.frequency, r.monetary, 
               r.segment, r.overall_rfm_score
        FROM customers c
        LEFT JOIN rfm_scores r ON c.customer_id = r.customer_id
        """
        
        params = []
        if segment:
            query += " WHERE r.segment = %s"
            params.append(segment)
        
        query += " ORDER BY r.overall_rfm_score DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(columns)
        
        # Write data rows
        for row in rows:
            writer.writerow(row)
        
        # Convert to bytes
        output.seek(0)
        csv_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'customers_export_{timestamp}.csv'
        
        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@export_bp.route('/rfm', methods=['POST'])
def export_rfm():
    """Export RFM analysis summary to CSV"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT segment, 
               COUNT(*) as customer_count,
               AVG(recency_days) as avg_recency,
               AVG(frequency) as avg_frequency,
               AVG(monetary) as avg_monetary,
               AVG(overall_rfm_score) as avg_rfm_score
        FROM rfm_scores
        GROUP BY segment
        ORDER BY avg_rfm_score DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)
        
        for row in rows:
            writer.writerow(row)
        
        output.seek(0)
        csv_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'rfm_summary_{timestamp}.csv'
        
        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500