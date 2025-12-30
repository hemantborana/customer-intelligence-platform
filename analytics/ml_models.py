import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import silhouette_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self, db_path='database/customer_intelligence.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """prepare datasets for ml models"""
        print(" Preparing data for ML models...\n")
        
        # load customer and transaction data
        df_customers = pd.read_sql('SELECT * FROM customers', self.conn)
        df_transactions = pd.read_sql('SELECT * FROM transactions', self.conn)
        
        # create customer features
        customer_features = df_transactions.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'std', 'count'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'total_spent', 'avg_amount', 
                                     'std_amount', 'transaction_count', 'first_purchase', 'last_purchase']
        
        # calculate recency
        customer_features['first_purchase'] = pd.to_datetime(customer_features['first_purchase'])
        customer_features['last_purchase'] = pd.to_datetime(customer_features['last_purchase'])
        reference_date = pd.Timestamp('2024-12-31')
        customer_features['recency_days'] = (reference_date - customer_features['last_purchase']).dt.days
        customer_features['tenure_days'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.days
        
        # merge with customer data
        self.df_ml = df_customers.merge(customer_features, on='customer_id', how='left')
        self.df_ml['total_spent'] = self.df_ml['total_spent'].fillna(0)
        self.df_ml['transaction_count'] = self.df_ml['transaction_count'].fillna(0)
        self.df_ml['recency_days'] = self.df_ml['recency_days'].fillna(999)
        self.df_ml['tenure_days'] = self.df_ml['tenure_days'].fillna(0)
        self.df_ml['std_amount'] = self.df_ml['std_amount'].fillna(0)
        
        # create churn label (inactive or no recent transactions)
        self.df_ml['churned'] = ((self.df_ml['is_active'] == 0) | 
                                 (self.df_ml['recency_days'] > 180)).astype(int)
        
        print(f" Prepared {len(self.df_ml)} customer records")
        print(f" Features created: {self.df_ml.shape[1]} columns\n")
        
    def regression_models(self):
        """build regression models for customer lifetime value"""
        print("=" * 60)
        print("1. REGRESSION MODELS - Customer Lifetime Value Prediction")
        print("=" * 60)
        
        # prepare features for regression
        feature_cols = ['age', 'transaction_count', 'avg_amount', 'recency_days', 
                       'tenure_days', 'segment_id']
        X = self.df_ml[feature_cols].fillna(0)
        y = self.df_ml['total_spent']
        
        # remove customers with zero spending
        mask = y > 0
        X = X[mask]
        y = y[mask]
        
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        regression_results = {}
        
        # linear regression
        print("\n Linear Regression:")
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mse_lr)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
        
        print(f"  MSE: ${mse_lr:.2f}")
        print(f"  RMSE: ${rmse_lr:.2f}")
        print(f"  MAE: ${mae_lr:.2f}")
        print(f"  R²: {r2_lr:.4f}")
        print(f"  MAPE: {mape_lr:.2f}%")
        
        regression_results['linear'] = {
            'mse': float(mse_lr), 'rmse': float(rmse_lr), 'mae': float(mae_lr),
            'r2': float(r2_lr), 'mape': float(mape_lr)
        }
        
        # ridge regression with regularization
        print("\n Ridge Regression (L2 Regularization):")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_test_scaled)
        
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        rmse_ridge = np.sqrt(mse_ridge)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        
        print(f"  RMSE: ${rmse_ridge:.2f}")
        print(f"  R²: {r2_ridge:.4f}")
        
        regression_results['ridge'] = {
            'rmse': float(rmse_ridge), 'r2': float(r2_ridge)
        }
        
        # lasso regression
        print("\n Lasso Regression (L1 Regularization):")
        lasso = Lasso(alpha=1.0)
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        
        rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
        r2_lasso = r2_score(y_test, y_pred_lasso)
        
        print(f"  RMSE: ${rmse_lasso:.2f}")
        print(f"  R²: {r2_lasso:.4f}")
        
        regression_results['lasso'] = {
            'rmse': float(rmse_lasso), 'r2': float(r2_lasso)
        }
        
        # xgboost for comparison (bonus points!)
        print("\n XGBoost Regression:")
        xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        xgb_reg.fit(X_train, y_train)
        y_pred_xgb = xgb_reg.predict(X_test)
        
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        r2_xgb = r2_score(y_test, y_pred_xgb)
        
        print(f"  RMSE: ${rmse_xgb:.2f}")
        print(f"  R²: {r2_xgb:.4f}")
        
        regression_results['xgboost'] = {
            'rmse': float(rmse_xgb), 'r2': float(r2_xgb)
        }
        
        self.models['regression'] = {
            'linear': lr, 'ridge': ridge, 'lasso': lasso, 'xgboost': xgb_reg,
            'scaler': scaler, 'features': feature_cols
        }
        self.results['regression'] = regression_results
        
        print("\n Best Model: XGBoost" if r2_xgb > max(r2_lr, r2_ridge, r2_lasso) else "\n Best Model: Linear Regression")
    
    def classification_models(self):
        """build classification models for churn prediction"""
        print("\n" + "=" * 60)
        print("2. CLASSIFICATION MODELS - Churn Prediction")
        print("=" * 60)
        
        # prepare features
        feature_cols = ['age', 'transaction_count', 'avg_amount', 'recency_days', 
                       'tenure_days', 'segment_id']
        X = self.df_ml[feature_cols].fillna(0)
        y = self.df_ml['churned']
        
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, stratify=y)
        
        # scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        classification_results = {}
        
        # logistic regression
        print("\n Logistic Regression:")
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_train_scaled, y_train)
        y_pred_log = log_reg.predict(X_test_scaled)
        
        acc_log = accuracy_score(y_test, y_pred_log)
        prec_log = precision_score(y_test, y_pred_log, zero_division=0)
        rec_log = recall_score(y_test, y_pred_log, zero_division=0)
        f1_log = f1_score(y_test, y_pred_log, zero_division=0)
        
        print(f"  Accuracy: {acc_log:.4f}")
        print(f"  Precision: {prec_log:.4f}")
        print(f"  Recall: {rec_log:.4f}")
        print(f"  F1-Score: {f1_log:.4f}")
        
        classification_results['logistic'] = {
            'accuracy': float(acc_log), 'precision': float(prec_log),
            'recall': float(rec_log), 'f1': float(f1_log)
        }
        
        # decision tree
        print("\n Decision Tree Classifier:")
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        
        acc_dt = accuracy_score(y_test, y_pred_dt)
        f1_dt = f1_score(y_test, y_pred_dt, zero_division=0)
        
        print(f"  Accuracy: {acc_dt:.4f}")
        print(f"  F1-Score: {f1_dt:.4f}")
        
        classification_results['decision_tree'] = {
            'accuracy': float(acc_dt), 'f1': float(f1_dt)
        }
        
        # svm with hyperparameter tuning
        print("\n SVM Classifier (with Grid Search):")
        svm = SVC(kernel='rbf', random_state=42)
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1')
        grid_search.fit(X_train_scaled[:1000], y_train[:1000])  # subset for speed
        
        print(f"  Best Params: {grid_search.best_params_}")
        print(f"  Best CV F1: {grid_search.best_score_:.4f}")
        
        classification_results['svm'] = {
            'best_params': grid_search.best_params_,
            'best_cv_f1': float(grid_search.best_score_)
        }
        
        # xgboost classifier (bonus!)
        print("\n XGBoost Classifier:")
        xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
        xgb_clf.fit(X_train, y_train)
        y_pred_xgb = xgb_clf.predict(X_test)
        
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
        
        print(f"  Accuracy: {acc_xgb:.4f}")
        print(f"  F1-Score: {f1_xgb:.4f}")
        
        classification_results['xgboost'] = {
            'accuracy': float(acc_xgb), 'f1': float(f1_xgb)
        }
        
        self.models['classification'] = {
            'logistic': log_reg, 'decision_tree': dt, 'xgboost': xgb_clf,
            'scaler': scaler, 'features': feature_cols
        }
        self.results['classification'] = classification_results
        
        print("\n Best Classifier: XGBoost" if f1_xgb > max(f1_log, f1_dt) else "\n Best Classifier: Logistic Regression")
    
    def clustering_models(self):
        """build clustering models for customer segmentation"""
        print("\n" + "=" * 60)
        print("3. CLUSTERING MODELS - Customer Segmentation")
        print("=" * 60)
        
        # prepare features
        feature_cols = ['age', 'total_spent', 'transaction_count', 'recency_days']
        X = self.df_ml[feature_cols].fillna(0)
        
        # scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # determine optimal k using elbow method
        print("\n Finding optimal number of clusters...")
        inertias = []
        silhouette_scores = []
        k_range = range(2, 8)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # use k with best silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"  Optimal K: {optimal_k} (Silhouette Score: {max(silhouette_scores):.4f})")
        
        # final kmeans model
        print(f"\n K-Means Clustering (K={optimal_k}):")
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.df_ml['cluster'] = kmeans_final.fit_predict(X_scaled)
        
        # analyze clusters
        cluster_summary = self.df_ml.groupby('cluster')[feature_cols].mean().round(2)
        cluster_sizes = self.df_ml['cluster'].value_counts().sort_index()
        
        print("\nCluster Summary:")
        for cluster_id in range(optimal_k):
            print(f"\n  Cluster {cluster_id} ({cluster_sizes[cluster_id]} customers):")
            print(f"    - Avg Age: {cluster_summary.loc[cluster_id, 'age']:.0f}")
            print(f"    - Avg Spent: ${cluster_summary.loc[cluster_id, 'total_spent']:.2f}")
            print(f"    - Avg Transactions: {cluster_summary.loc[cluster_id, 'transaction_count']:.0f}")
            print(f"    - Avg Recency: {cluster_summary.loc[cluster_id, 'recency_days']:.0f} days")
        
        clustering_results = {
            'optimal_k': int(optimal_k),
            'silhouette_score': float(max(silhouette_scores)),
            'cluster_sizes': cluster_sizes.to_dict(),
            'cluster_summary': cluster_summary.to_dict()
        }
        
        self.models['clustering'] = {
            'kmeans': kmeans_final, 'scaler': scaler, 
            'features': feature_cols, 'optimal_k': optimal_k
        }
        self.results['clustering'] = clustering_results
    
    def model_validation(self):
        """comprehensive model validation"""
        print("\n" + "=" * 60)
        print("4. MODEL VALIDATION & COMPARISON")
        print("=" * 60)
        
        # cross validation for regression
        print("\n Regression Model Cross-Validation (5-Fold):")
        feature_cols = ['age', 'transaction_count', 'avg_amount', 'recency_days', 
                       'tenure_days', 'segment_id']
        X = self.df_ml[feature_cols].fillna(0)
        y = self.df_ml['total_spent']
        mask = y > 0
        X = X[mask]
        y = y[mask]
        
        lr = self.models['regression']['linear']
        cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
        print(f"  Linear Regression R² CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # classification validation
        print("\n Classification Model Cross-Validation (5-Fold):")
        y_class = self.df_ml['churned']
        log_reg = self.models['classification']['logistic']
        
        X_scaled = self.models['classification']['scaler'].transform(X)
        cv_scores_class = cross_val_score(log_reg, X_scaled, y_class[mask], cv=5, scoring='f1')
        print(f"  Logistic Regression F1 CV: {cv_scores_class.mean():.4f} (+/- {cv_scores_class.std():.4f})")
        
        print("\n Model Validation Completed!")
        
        self.results['validation'] = {
            'regression_cv_r2': float(cv_scores.mean()),
            'classification_cv_f1': float(cv_scores_class.mean())
        }
    
    def save_models(self):
        """save trained models"""
        import pickle
        
        with open('analytics/ml_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        
        print("\n Models saved to: analytics/ml_models.pkl")
        
        # save results
        import json
        with open('analytics/ml_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(" Results saved to: analytics/ml_results.json")
    
    def run_ml_pipeline(self):
        """execute complete ml pipeline"""
        print("\n" + "*" * 30)
        print("MACHINE LEARNING PIPELINE")
        print("*" * 30 + "\n")
        
        self.prepare_data()
        self.regression_models()
        self.classification_models()
        self.clustering_models()
        self.model_validation()
        self.save_models()
        
        print("\n" + "=" * 60)
        print(" ML PIPELINE COMPLETED!")
        print("=" * 60)
        
        self.conn.close()

if __name__ == "__main__":
    ml = MLPipeline()
    ml.run_ml_pipeline()