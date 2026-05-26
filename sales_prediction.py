import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
from functools import lru_cache
import time
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for models and scaler
primary_model = None  # XGBoost (fast, primary)
secondary_model = None  # Gradient Boosting (fallback, ensemble)
random_forest_model = None  # Random Forest
lightgbm_model = None  # LightGBM
linear_regression_model = None  # Linear Regression
scaler = None
feature_names = None
prediction_cache = {}  # Simple cache for frequent predictions
cache_expiry = 300  # Cache validity in seconds

# Performance metrics
inference_times = []
model_inference_times = {
    'xgboost': [],
    'gradient_boosting': [],
    'random_forest': [],
    'lightgbm': [],
    'linear_regression': []
}
MAX_CACHE_SIZE = 1000

# Model accuracy (R² scores) computed at training time
model_r2_scores = {
    'xgboost': 0,
    'gradient_boosting': 0,
    'random_forest': 0,
    'lightgbm': 0,
    'linear_regression': 0
}

def generate_enhanced_sample_data():
    """Generate enhanced sample sales data with more realistic patterns"""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Base features
    data = {
        'advertising_spend': np.random.uniform(1000, 50000, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day_of_week': np.random.randint(1, 8, n_samples),
        'inventory': np.random.uniform(10, 1000, n_samples),
        'price': np.random.uniform(20, 200, n_samples),
        'discount_percent': np.random.uniform(0, 50, n_samples),
        'website_traffic': np.random.uniform(100, 10000, n_samples),
        'social_media_engagement': np.random.uniform(0, 10000, n_samples),
        'customer_reviews_count': np.random.uniform(0, 500, n_samples),
        'average_rating': np.random.uniform(2, 5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create engineered features for better prediction
    df['ad_efficiency'] = df['advertising_spend'] / (df['website_traffic'] + 1)
    df['price_discount_ratio'] = df['price'] / (df['discount_percent'] + 1)
    df['engagement_traffic_ratio'] = df['social_media_engagement'] / (df['website_traffic'] + 1)
    df['inventory_turnover'] = df['inventory'] / (df['advertising_spend'] / 10000)
    df['rating_count_product'] = df['average_rating'] * (df['customer_reviews_count'] + 1)
    df['seasonal_factor'] = np.sin(2 * np.pi * df['month'] / 12)  # Seasonal pattern
    df['weekly_factor'] = np.cos(2 * np.pi * df['day_of_week'] / 7)  # Weekly pattern
    
    # Target variable with stronger relationships and non-linear patterns
    df['sales'] = (
        0.6 * df['advertising_spend'] / 1000 +
        80 * (df['month'] / 12) +
        50 * (df['day_of_week'] / 8) +
        2.5 * df['inventory'] +
        -1.2 * df['price'] +
        2.0 * df['discount_percent'] +
        0.4 * df['website_traffic'] +
        0.08 * df['social_media_engagement'] +
        12 * df['customer_reviews_count'] / 100 +
        600 * df['average_rating'] +
        100 * df['seasonal_factor'] +
        80 * df['weekly_factor'] +
        200 * np.log1p(df['rating_count_product'] / 100)
    ) + np.random.normal(0, 80, n_samples)
    
    df['sales'] = df['sales'].clip(lower=0)
    
    return df

def engineer_features(X):
    """Create engineered features from raw features"""
    X_engineered = X.copy()
    
    X_engineered['ad_efficiency'] = X_engineered['advertising_spend'] / (X_engineered['website_traffic'] + 1)
    X_engineered['price_discount_ratio'] = X_engineered['price'] / (X_engineered['discount_percent'] + 1)
    X_engineered['engagement_traffic_ratio'] = X_engineered['social_media_engagement'] / (X_engineered['website_traffic'] + 1)
    X_engineered['inventory_turnover'] = X_engineered['inventory'] / (X_engineered['advertising_spend'] / 10000)
    X_engineered['rating_count_product'] = X_engineered['average_rating'] * (X_engineered['customer_reviews_count'] + 1)
    X_engineered['seasonal_factor'] = np.sin(2 * np.pi * X_engineered['month'] / 12)
    X_engineered['weekly_factor'] = np.cos(2 * np.pi * X_engineered['day_of_week'] / 7)
    
    return X_engineered

def train_optimized_models():
    """Train optimized models: XGBoost, Gradient Boosting, Random Forest, LightGBM, and Linear Regression"""
    global primary_model, secondary_model, random_forest_model, lightgbm_model, linear_regression_model, scaler, feature_names, model_r2_scores
    
    print("Generating enhanced training data...")
    df = generate_enhanced_sample_data()
    
    # Separate features and target
    X = df.drop('sales', axis=1)
    y = df['sales']
    
    # Engineer features
    print("Engineering features...")
    X = engineer_features(X)
    feature_names = X.columns.tolist()
    
    print(f"Features: {feature_names}")
    print(f"Data shape: {X.shape}")
    print(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ==================== XGBOOST MODEL (Primary - Fast Inference) ====================
    print("\n" + "="*60)
    print("Training XGBoost model (Primary)...")
    print("="*60)
    
    primary_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',  # Faster tree building
        eval_metric='rmse'
    )
    
    primary_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # ==================== GRADIENT BOOSTING MODEL (Secondary - Ensemble) ====================
    print("\nTraining Gradient Boosting model (Secondary)...")
    
    secondary_model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        validation_fraction=0.2,
        n_iter_no_change=20
    )
    
    secondary_model.fit(X_train_scaled, y_train)
    
    # ==================== RANDOM FOREST MODEL ====================
    print("\nTraining Random Forest model...")
    
    random_forest_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )
    
    random_forest_model.fit(X_train_scaled, y_train)
    
    # ==================== LIGHTGBM MODEL ====================
    print("\nTraining LightGBM model...")
    
    lightgbm_model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lightgbm_model.fit(X_train_scaled, y_train)
    
    # ==================== LINEAR REGRESSION MODEL ====================
    print("\nTraining Linear Regression model...")
    
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train_scaled, y_train)
    
    # ==================== EVALUATION ====================
    print("\n" + "="*60)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Get predictions from all models
    y_pred_xgb_test = primary_model.predict(X_test_scaled)
    y_pred_gb_test = secondary_model.predict(X_test_scaled)
    y_pred_rf_test = random_forest_model.predict(X_test_scaled)
    y_pred_lgb_test = lightgbm_model.predict(X_test_scaled)
    y_pred_lr_test = linear_regression_model.predict(X_test_scaled)
    
    # Ensemble predictions (weighted average)
    y_pred_ensemble_test = (
        y_pred_xgb_test * 0.25 + 
        y_pred_gb_test * 0.20 + 
        y_pred_rf_test * 0.20 + 
        y_pred_lgb_test * 0.25 + 
        y_pred_lr_test * 0.10
    )
    
    # XGBoost metrics
    xgb_test_r2 = r2_score(y_test, y_pred_xgb_test)
    xgb_test_mae = mean_absolute_error(y_test, y_pred_xgb_test)
    xgb_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_test))
    xgb_test_mape = mean_absolute_percentage_error(y_test, y_pred_xgb_test)
    
    # Gradient Boosting metrics
    gb_test_r2 = r2_score(y_test, y_pred_gb_test)
    gb_test_mae = mean_absolute_error(y_test, y_pred_gb_test)
    gb_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb_test))
    gb_test_mape = mean_absolute_percentage_error(y_test, y_pred_gb_test)
    
    # Random Forest metrics
    rf_test_r2 = r2_score(y_test, y_pred_rf_test)
    rf_test_mae = mean_absolute_error(y_test, y_pred_rf_test)
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
    rf_test_mape = mean_absolute_percentage_error(y_test, y_pred_rf_test)
    
    # LightGBM metrics
    lgb_test_r2 = r2_score(y_test, y_pred_lgb_test)
    lgb_test_mae = mean_absolute_error(y_test, y_pred_lgb_test)
    lgb_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb_test))
    lgb_test_mape = mean_absolute_percentage_error(y_test, y_pred_lgb_test)
    
    # Linear Regression metrics
    lr_test_r2 = r2_score(y_test, y_pred_lr_test)
    lr_test_mae = mean_absolute_error(y_test, y_pred_lr_test)
    lr_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr_test))
    lr_test_mape = mean_absolute_percentage_error(y_test, y_pred_lr_test)
    
    # Ensemble metrics
    ensemble_test_r2 = r2_score(y_test, y_pred_ensemble_test)
    ensemble_test_mae = mean_absolute_error(y_test, y_pred_ensemble_test)
    ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble_test))
    ensemble_test_mape = mean_absolute_percentage_error(y_test, y_pred_ensemble_test)
    
    # Print results
    print("\n📊 XGBoost Results:")
    print(f"  R² Score: {xgb_test_r2:.4f}")
    print(f"  MAE: ${xgb_test_mae:.2f}")
    print(f"  RMSE: ${xgb_test_rmse:.2f}")
    print(f"  MAPE: {xgb_test_mape:.4f}")
    
    print("\n📊 Gradient Boosting Results:")
    print(f"  R² Score: {gb_test_r2:.4f}")
    print(f"  MAE: ${gb_test_mae:.2f}")
    print(f"  RMSE: ${gb_test_rmse:.2f}")
    print(f"  MAPE: {gb_test_mape:.4f}")
    
    print("\n📊 Random Forest Results:")
    print(f"  R² Score: {rf_test_r2:.4f}")
    print(f"  MAE: ${rf_test_mae:.2f}")
    print(f"  RMSE: ${rf_test_rmse:.2f}")
    print(f"  MAPE: {rf_test_mape:.4f}")
    
    print("\n📊 LightGBM Results:")
    print(f"  R² Score: {lgb_test_r2:.4f}")
    print(f"  MAE: ${lgb_test_mae:.2f}")
    print(f"  RMSE: ${lgb_test_rmse:.2f}")
    print(f"  MAPE: {lgb_test_mape:.4f}")
    
    print("\n📊 Linear Regression Results:")
    print(f"  R² Score: {lr_test_r2:.4f}")
    print(f"  MAE: ${lr_test_mae:.2f}")
    print(f"  RMSE: ${lr_test_rmse:.2f}")
    print(f"  MAPE: {lr_test_mape:.4f}")
    
    print("\n📊 Ensemble Results (All Models Combined):")
    print(f"  R² Score: {ensemble_test_r2:.4f}")
    print(f"  MAE: ${ensemble_test_mae:.2f}")
    print(f"  RMSE: ${ensemble_test_rmse:.2f}")
    print(f"  MAPE: {ensemble_test_mape:.4f}")
    
    # Return stats for display
    stats = {
        'xgb_r2': xgb_test_r2,
        'gb_r2': gb_test_r2,
        'rf_r2': rf_test_r2,
        'lgb_r2': lgb_test_r2,
        'lr_r2': lr_test_r2,
        'ensemble_r2': ensemble_test_r2,
        'ensemble_accuracy': ensemble_test_r2 * 100,
        'xgb_inference_time_ms': 0.5  # Placeholder
    }

    # Store R² scores globally for use in /predict and /model-comparison
    model_r2_scores['xgboost']           = round(xgb_test_r2, 4)
    model_r2_scores['gradient_boosting'] = round(gb_test_r2, 4)
    model_r2_scores['random_forest']     = round(rf_test_r2, 4)
    model_r2_scores['lightgbm']          = round(lgb_test_r2, 4)
    model_r2_scores['linear_regression'] = round(lr_test_r2, 4)

    return stats

def train_model():
    """Wrapper function for training"""
    return train_optimized_models()

def cached_predict(data):
    """Check if prediction exists in cache"""
    key = json.dumps(data, sort_keys=True)
    if key in prediction_cache:
        return prediction_cache[key]['value'], True
    return None, False

def update_cache(features, result):
    """Update prediction cache"""
    if len(prediction_cache) >= MAX_CACHE_SIZE:
        prediction_cache.pop(next(iter(prediction_cache)))
    
    key = json.dumps(features, sort_keys=True)
    prediction_cache[key] = {
        'value': result,
        'timestamp': time.time()
    }

# ==================== FLASK ENDPOINTS ====================

@app.route('/predict', methods=['POST'])
def predict():
    """Real-time sales prediction endpoint with caching using all models"""
    global primary_model, secondary_model, random_forest_model, lightgbm_model, linear_regression_model, scaler
    
    if primary_model is None or secondary_model is None or scaler is None:
        return jsonify({'error': 'Models not trained yet'}), 400
    
    start_time = time.time()
    
    try:
        data = request.json
        
        # Check cache first
        cached_result, is_cached = cached_predict(data)
        if cached_result:
            return jsonify({
                'predicted_sales': cached_result,
                'status': 'success',
                'inference_time_ms': 0.001,
                'from_cache': True
            })
        
        # Extract raw features in correct order
        raw_features = {
            'advertising_spend': float(data['advertising_spend']),
            'month': int(data['month']),
            'day_of_week': int(data['day_of_week']),
            'inventory': float(data['inventory']),
            'price': float(data['price']),
            'discount_percent': float(data['discount_percent']),
            'website_traffic': float(data['website_traffic']),
            'social_media_engagement': float(data['social_media_engagement']),
            'customer_reviews_count': float(data['customer_reviews_count']),
            'average_rating': float(data['average_rating'])
        }
        
        # Create dataframe for feature engineering
        df_input = pd.DataFrame([raw_features])
        df_engineered = engineer_features(df_input)
        
        # Reshape for prediction
        X = np.array(df_engineered).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        xgb_start = time.time()
        xgb_pred = float(primary_model.predict(X_scaled)[0])
        model_inference_times['xgboost'].append((time.time() - xgb_start) * 1000)
        
        gb_start = time.time()
        gb_pred = float(secondary_model.predict(X_scaled)[0])
        model_inference_times['gradient_boosting'].append((time.time() - gb_start) * 1000)
        
        rf_start = time.time()
        rf_pred = float(random_forest_model.predict(X_scaled)[0])
        model_inference_times['random_forest'].append((time.time() - rf_start) * 1000)
        
        lgb_start = time.time()
        lgb_pred = float(lightgbm_model.predict(X_scaled)[0])
        model_inference_times['lightgbm'].append((time.time() - lgb_start) * 1000)
        
        lr_start = time.time()
        lr_pred = float(linear_regression_model.predict(X_scaled)[0])
        model_inference_times['linear_regression'].append((time.time() - lr_start) * 1000)
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (xgb_pred * 0.25 + gb_pred * 0.20 + rf_pred * 0.20 + lgb_pred * 0.25 + lr_pred * 0.10)
        
        # Ensure non-negative prediction
        ensemble_pred = max(0, ensemble_pred)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
        
        result_value = round(ensemble_pred, 2)
        
        # Update cache
        update_cache(raw_features, result_value)

        # Per-model inference times for this request (ms)
        per_model_times = {
            'xgboost':           round(model_inference_times['xgboost'][-1], 4)           if model_inference_times['xgboost'] else 0,
            'gradient_boosting': round(model_inference_times['gradient_boosting'][-1], 4) if model_inference_times['gradient_boosting'] else 0,
            'random_forest':     round(model_inference_times['random_forest'][-1], 4)     if model_inference_times['random_forest'] else 0,
            'lightgbm':          round(model_inference_times['lightgbm'][-1], 4)          if model_inference_times['lightgbm'] else 0,
            'linear_regression': round(model_inference_times['linear_regression'][-1], 4) if model_inference_times['linear_regression'] else 0,
        }

        # Best model = highest R² on test set
        best_model = max(model_r2_scores, key=model_r2_scores.get)

        # Fastest model = lowest inference time this request
        fastest_model = min(per_model_times, key=per_model_times.get)

        return jsonify({
            'predicted_sales': result_value,
            'xgb_prediction': round(xgb_pred, 2),
            'gb_prediction': round(gb_pred, 2),
            'rf_prediction': round(rf_pred, 2),
            'lgb_prediction': round(lgb_pred, 2),
            'lr_prediction': round(lr_pred, 2),
            'ensemble_prediction': result_value,
            'status': 'success',
            'inference_time_ms': round(inference_time, 4),
            'per_model_inference_times_ms': per_model_times,
            'model_r2_scores': model_r2_scores,
            'best_model': best_model,
            'fastest_model': fastest_model,
            'from_cache': False
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple records (time-efficient)"""
    global primary_model, secondary_model, random_forest_model, lightgbm_model, linear_regression_model, scaler
    
    if primary_model is None or secondary_model is None or scaler is None:
        return jsonify({'error': 'Models not trained yet'}), 400
    
    start_time = time.time()
    
    try:
        data_list = request.json.get('predictions', [])
        
        if not data_list or len(data_list) == 0:
            return jsonify({'error': 'No predictions provided'}), 400
        
        predictions = []
        
        for data in data_list:
            raw_features = {
                'advertising_spend': float(data['advertising_spend']),
                'month': int(data['month']),
                'day_of_week': int(data['day_of_week']),
                'inventory': float(data['inventory']),
                'price': float(data['price']),
                'discount_percent': float(data['discount_percent']),
                'website_traffic': float(data['website_traffic']),
                'social_media_engagement': float(data['social_media_engagement']),
                'customer_reviews_count': float(data['customer_reviews_count']),
                'average_rating': float(data['average_rating'])
            }
            
            df_input = pd.DataFrame([raw_features])
            df_engineered = engineer_features(df_input)
            X = np.array(df_engineered).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            xgb_pred = float(primary_model.predict(X_scaled)[0])
            gb_pred = float(secondary_model.predict(X_scaled)[0])
            rf_pred = float(random_forest_model.predict(X_scaled)[0])
            lgb_pred = float(lightgbm_model.predict(X_scaled)[0])
            lr_pred = float(linear_regression_model.predict(X_scaled)[0])
            
            ensemble_pred = max(0, (xgb_pred * 0.25 + gb_pred * 0.20 + rf_pred * 0.20 + lgb_pred * 0.25 + lr_pred * 0.10))
            
            predictions.append({
                'predicted_sales': round(ensemble_pred, 2),
                'input': raw_features
            })
        
        batch_inference_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'predictions': predictions,
            'total_predictions': len(predictions),
            'batch_inference_time_ms': round(batch_inference_time, 2),
            'average_time_per_prediction_ms': round(batch_inference_time / len(predictions), 4),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    if primary_model is None:
        return jsonify({'error': 'Models not trained'}), 400
    
    feature_importance_xgb = pd.DataFrame({
        'feature': feature_names,
        'importance': primary_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_rf = pd.DataFrame({
        'feature': feature_names,
        'importance': random_forest_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_lgb = pd.DataFrame({
        'feature': feature_names,
        'importance': lightgbm_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    return jsonify({
        'model_type': 'Multi-Algorithm Ensemble (XGBoost + GB + RF + LightGBM + Linear Regression)',
        'features': feature_importance_xgb.to_dict('records'),
        'feature_importance_comparison': {
            'xgboost': feature_importance_xgb.head(5).to_dict('records'),
            'random_forest': feature_importance_rf.head(5).to_dict('records'),
            'lightgbm': feature_importance_lgb.head(5).to_dict('records')
        },
        'xgb_n_estimators': primary_model.n_estimators,
        'gb_n_estimators': secondary_model.n_estimators,
        'rf_n_estimators': random_forest_model.n_estimators,
        'lgb_n_estimators': lightgbm_model.n_estimators,
        'ensemble_strategy': 'Weighted Average (XGB 25%, GB 20%, RF 20%, LGB 25%, LR 10%)',
        'caching_enabled': True,
        'cache_size': len(prediction_cache),
        'average_inference_time_ms': round(avg_inference_time, 4),
        'real_time_capable': True
    })

@app.route('/model-comparison', methods=['GET'])
def model_comparison():
    """Compare performance across all models"""
    avg_times = {
        'xgboost': np.mean(model_inference_times['xgboost']) if model_inference_times['xgboost'] else 0,
        'gradient_boosting': np.mean(model_inference_times['gradient_boosting']) if model_inference_times['gradient_boosting'] else 0,
        'random_forest': np.mean(model_inference_times['random_forest']) if model_inference_times['random_forest'] else 0,
        'lightgbm': np.mean(model_inference_times['lightgbm']) if model_inference_times['lightgbm'] else 0,
        'linear_regression': np.mean(model_inference_times['linear_regression']) if model_inference_times['linear_regression'] else 0
    }
    
    return jsonify({
        'model_inference_times_ms': {
            'xgboost': round(avg_times['xgboost'], 4),
            'gradient_boosting': round(avg_times['gradient_boosting'], 4),
            'random_forest': round(avg_times['random_forest'], 4),
            'lightgbm': round(avg_times['lightgbm'], 4),
            'linear_regression': round(avg_times['linear_regression'], 4)
        },
        'model_r2_scores': model_r2_scores,
        'best_model': max(model_r2_scores, key=model_r2_scores.get) if any(model_r2_scores.values()) else 'N/A',
        'fastest_model': min(avg_times, key=avg_times.get),
        'slowest_model': max(avg_times, key=avg_times.get),
        'average_inference_time_ms': round(np.mean(list(avg_times.values())), 4)
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'API is running',
        'models_trained': (primary_model is not None and secondary_model is not None and 
                          random_forest_model is not None and lightgbm_model is not None and
                          linear_regression_model is not None),
        'scaler_ready': scaler is not None
    }), 200

@app.route('/performance-stats', methods=['GET'])
def performance_stats():
    """Get model performance statistics"""
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    return jsonify({
        'total_predictions': len(inference_times),
        'average_inference_time_ms': round(avg_inference_time, 4),
        'min_inference_time_ms': round(min(inference_times), 4) if inference_times else 0,
        'max_inference_time_ms': round(max(inference_times), 4) if inference_times else 0,
        'predictions_per_second': round(1000 / avg_inference_time, 2) if avg_inference_time > 0 else 0,
        'cache_hit_rate': f"{(len(prediction_cache) / max(len(inference_times), 1) * 100):.2f}%"
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SALES PREDICTION API - MULTI-ALGORITHM VERSION")
    print("="*60)
    print("Training optimized ML models...")
    print("\nModels included:")
    print("  ✓ XGBoost")
    print("  ✓ Gradient Boosting")
    print("  ✓ Random Forest")
    print("  ✓ LightGBM")
    print("  ✓ Linear Regression")
    print("="*60)
    stats = train_model()
    print("\n" + "="*60)
    print("Starting Flask server...")
    print("="*60)
    print("API available at http://localhost:5000")
    print("\nEndpoints:")
    print("  POST /predict - Single real-time prediction with caching")
    print("  POST /batch-predict - Multiple predictions (time-efficient)")
    print("  GET /model-info - Model information and feature importance")
    print("  GET /model-comparison - Model performance comparison")
    print("  GET /performance-stats - Real-time performance metrics")
    print("  GET /health - Health check")
    print("\nEnsemble Performance:")
    print(f"  - R² Score: {stats['ensemble_r2']:.4f}")
    print(f"  - Accuracy: {stats['ensemble_accuracy']:.2f}%")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
