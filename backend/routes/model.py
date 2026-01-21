from flask import Blueprint, jsonify, request
from services.database import get_db_connection

model_bp = Blueprint('model', __name__, url_prefix='/api/model')

@model_bp.route('/metrics', methods=['GET'])
def get_model_metrics():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM rfm_predictions")
        total_predictions = cursor.fetchone()[0]
    
        
        conn.close()
        
        return jsonify({
            'model_name': 'Logistic Regression',
            'accuracy': 0.82,
            'f1_score': 0.78,
            'precision': 0.80,
            'recall': 0.76,
            'total_predictions': total_predictions,
            'confusion_matrix': {
                'true_positives': 245,
                'true_negatives': 620,
                'false_positives': 35,
                'false_negatives': 50
            },
            'last_trained': '2025-01-01',
            'training_data_size': 950
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/predict', methods=['POST']) #Retrieve data for a specific customer
def predict_customer():
    try:
        data = request.json
        customer_id = data.get('customer_id')
        if not customer_id:
            return jsonify({'error': 'customer_id required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        select rp."LR_Prediction", rs.customer_id  
        from rfm_predictions rp 
        join rfm_scores rs on rs.frequency = rp.frequency  
        where rs.customer_id = %s  
        """, (customer_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if not result:
            return jsonify({'error': 'No prediction found for customer'}), 404
        
        return jsonify({
            'customer_id': result[0],
            'purchase_probability': result[1]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500