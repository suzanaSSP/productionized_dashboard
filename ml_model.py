from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report
import pandas as pd
import logging
from sqlalchemy import create_engine
import numpy as np
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMPredictionModel:
    """Class for training and evaluating customer purchase prediction models."""
    
    def __init__(self, feature_columns=None, test_size=0.2, random_state=42):
        """
        Initialize the model.
        
        Args:
            feature_columns: List of feature column names to use
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.feature_columns = feature_columns or ['recency_days', 'frequency', 'monetary']
        self.test_size = test_size
        self.random_state = random_state
        
        self.rf_model = None
        self.lr_model = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X_columns = None
        self.valid_indices = None

        self.engine = create_engine('postgresql://byu_student:your_secure_password@localhost:5432/olist_sales')
    
        
        
    def prepare_data(self, rfm_df):
        """
        Prepare features and target variable.
        
        Args:
            rfm_df: DataFrame with RFM features and target
        """
        print("\n" + "="*60)
        print("PREPARING DATA")
        print("="*60 + "\n")
        
        # Select features and target
        X = rfm_df[self.feature_columns].copy()
        X = pd.get_dummies(X)  # Convert categories to numeric
        Y = rfm_df['is_high_value'].copy()  # Changed from 'segment'
        
        # Remove any rows with NaN values
        self.valid_indices = X.notna().all(axis=1) & Y.notna()
        X = X[self.valid_indices]
        Y = Y[self.valid_indices]
        
        self.X_columns = X.columns
        
        print(f"Total customers: {len(X)}")
        print(f"Features shape: {X.shape}")
        print(f"Target distribution:\n{Y.value_counts()}\n")
        
        # Train-Test Split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state, stratify=Y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}\n")
        
    def train_random_forest(self, n_estimators=100, n_jobs=-1):
        """
        Train Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            n_jobs: Number of jobs to run in parallel
        """
        print("Training Random Forest Classifier...")
        self.rf_model = RandomForestClassifier(
            n_estimators=50,           # Reduced from 100
            max_depth=5,                # Limit tree depth
            min_samples_split=100,      # Require more samples to split
            min_samples_leaf=50,        # Require more samples in leaf nodes
            random_state=self.random_state, 
            n_jobs=n_jobs
        )
        self.rf_model.fit(self.X_train, self.Y_train)
        
    def evaluate_random_forest(self):
        """Evaluate Random Forest model performance."""
        Y_pred_rf = self.rf_model.predict(self.X_test)
        
        print("\n" + "-"*60)
        print("RANDOM FOREST RESULTS")
        print("-"*60)
        
        rf_accuracy = accuracy_score(self.Y_test, Y_pred_rf)
        rf_f1 = f1_score(self.Y_test, Y_pred_rf, average='binary', zero_division=0)
        rf_precision = precision_score(self.Y_test, Y_pred_rf, zero_division=0)
        rf_recall = recall_score(self.Y_test, Y_pred_rf, zero_division=0)
        rf_cm = confusion_matrix(self.Y_test, Y_pred_rf, labels=[0, 1])
        
        print(f"Accuracy: {rf_accuracy:.4f}")
        print(f"Precision: {rf_precision:.4f}")
        print(f"Recall: {rf_recall:.4f}")
        print(f"F1-Score: {rf_f1:.4f}")
        print(f"\nConfusion Matrix:\n{rf_cm}")
        print("\nClassification Report:")
        print(classification_report(self.Y_test, Y_pred_rf, target_names=['Not High-Value', 'High-Value'], zero_division=0))
        
        # Feature importance
        print("\nFeature Importance:")
        feature_importance = pd.DataFrame({
            'Feature': self.X_columns,
            'Importance': self.rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(feature_importance)
        
        return rf_accuracy, rf_f1
        
    def train_logistic_regression(self, max_iter=1000):
        """
        Train Logistic Regression classifier.
        
        Args:
            max_iter: Maximum number of iterations
        """
        print("\n" + "-"*60)
        print("Training Logistic Regression...")
        self.lr_model = LogisticRegression(random_state=self.random_state, max_iter=max_iter)
        self.lr_model.fit(self.X_train, self.Y_train)
        
    def evaluate_logistic_regression(self):
        """Evaluate Logistic Regression model performance."""
        Y_pred_lr = self.lr_model.predict(self.X_test)
        
        print("\n" + "-"*60)
        print("LOGISTIC REGRESSION RESULTS")
        print("-"*60)
        
        lr_accuracy = accuracy_score(self.Y_test, Y_pred_lr)
        lr_f1 = f1_score(self.Y_test, Y_pred_lr, average='binary', zero_division=0)
        lr_precision = precision_score(self.Y_test, Y_pred_lr, zero_division=0)
        lr_recall = recall_score(self.Y_test, Y_pred_lr, zero_division=0)
        lr_cm = confusion_matrix(self.Y_test, Y_pred_lr, labels=[0, 1])
        
        print(f"Accuracy: {lr_accuracy:.4f}")
        print(f"Precision: {lr_precision:.4f}")
        print(f"Recall: {lr_recall:.4f}")
        print(f"F1-Score: {lr_f1:.4f}")
        print(f"\nConfusion Matrix:\n{lr_cm}")
        print("\nClassification Report:")
        print(classification_report(self.Y_test, Y_pred_lr, target_names=['Not High-Value', 'High-Value'], zero_division=0))
        
        return lr_accuracy, lr_f1
        
    def compare_models(self, rf_accuracy, rf_f1, lr_accuracy, lr_f1):
        """
        Compare model performances.
        
        Args:
            rf_accuracy: Random Forest accuracy
            rf_f1: Random Forest F1 score
            lr_accuracy: Logistic Regression accuracy
            lr_f1: Logistic Regression F1 score
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"Random Forest - Accuracy: {rf_accuracy:.4f}, F1-Score: {rf_f1:.4f}")
        print(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, F1-Score: {lr_f1:.4f}")
        
    def save_predictions(self, rfm_df):
        """
        Save predictions to database.
        
        Args:
            rfm_df: Original RFM DataFrame
        """
        # Prepare features for prediction
        X = rfm_df[self.feature_columns].copy()
        X = pd.get_dummies(X)
        
        # Add predictions
        rfm_df['RF_Prediction'] = np.nan
        rfm_df.loc[self.valid_indices, 'RF_Prediction'] = self.rf_model.predict(X[self.valid_indices])
        
        rfm_df['LR_Prediction'] = np.nan
        rfm_df.loc[self.valid_indices, 'LR_Prediction'] = self.lr_model.predict(X[self.valid_indices])
        
        rfm_df.to_sql('rfm_predictions', con=self.engine, if_exists='replace', index=False)
        print("\nPredictions saved to 'rfm_predictions' table!")
        
    def run_pipeline(self):
        """
        Run the complete modeling pipeline.
        """
        print("\n" + "="*60)
        print("STEP 2.2: MODELING WITH SCIKIT-LEARN")
        print("="*60 + "\n")
        
        # Use RAW RFM values, not the scores!
        query = """
        WITH rfm_percentiles AS (
        SELECT 
            PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY recency_days) AS recency_p67,
            PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY frequency) AS frequency_p67,
            PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY monetary) AS monetary_p67
        FROM rfm_scores
        WHERE monetary IS NOT NULL
        )
        SELECT 
            r.recency_days,
            r.frequency,
            r.monetary,
            CASE 
                WHEN r.recency_days <= p.recency_p67 
                    AND r.frequency >= p.frequency_p67 
                    AND r.monetary >= p.monetary_p67
                THEN 1 
                ELSE 0 
            END AS is_high_value
        FROM rfm_scores r
        CROSS JOIN rfm_percentiles p
        WHERE r.monetary IS NOT NULL
        """
        rfm_df = pd.read_sql_query(query, self.engine)
        
        # Update feature columns to use raw values
        self.feature_columns = ['recency_days', 'frequency', 'monetary']
        
        self.prepare_data(rfm_df)
        
        # Train and evaluate Random Forest
        self.train_random_forest()
        rf_accuracy, rf_f1 = self.evaluate_random_forest()
        
        # Train and evaluate Logistic Regression
        self.train_logistic_regression()
        lr_accuracy, lr_f1 = self.evaluate_logistic_regression()
        
        # Compare models
        self.compare_models(rf_accuracy, rf_f1, lr_accuracy, lr_f1)
        
        # Save predictions
        self.save_predictions(rfm_df)

if __name__ == "__main__":
    model = RFMPredictionModel()
    model.run_pipeline()