# Sales Analytics Dashboard

A modern, AI-powered Sales Analytics Dashboard built with HTML, CSS, JavaScript, Python, Flask, and Machine Learning.
This project provides real-time business insights, interactive visualizations, forecasting, and intelligent sales prediction using multiple ML algorithms.

---

## Features

### Interactive Analytics Dashboard

* Real-time KPI tracking
* Revenue analysis
* Order monitoring
* Profit margin insights
* Average order value calculations

### Advanced Data Visualization

* Dynamic charts using Chart.js
* Revenue trends
* Product performance analysis
* Regional sales distribution
* Customer segmentation
* Forecast visualization

### AI-Powered Sales Prediction

Integrated machine learning models:

* XGBoost
* Gradient Boosting
* Random Forest
* LightGBM
* Linear Regression

Features include:

* Ensemble prediction system
* Feature engineering
* Model comparison
* Prediction confidence metrics
* Real-time inference

### Smart Forecasting

* Sales forecasting for future months
* Trend analysis
* Seasonal pattern recognition
* Confidence-based forecasting

### Modern UI/UX

* Dark / Light / Midnight themes
* Responsive design
* Animated dashboard components
* Professional enterprise layout
* Customizable interface

### Backend API

* Flask REST API
* Real-time prediction endpoints
* Caching system for faster inference
* Optimized ML model serving

---

# Tech Stack

## Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

## Backend

* Python
* Flask
* Flask-CORS

## Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* NumPy
* Pandas

---

# Project Structure

```bash
├── sales_analytics_dashboard.html
├── sales_prediction.py
├── README.md
```

---

# Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/your-username/sales-analytics-dashboard.git
cd sales-analytics-dashboard
```

---

## 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn flask flask-cors xgboost lightgbm
```

---

## 3. Run Backend Server

```bash
python sales_prediction.py
```

The Flask API will start on:

```bash
http://127.0.0.1:5000
```

---

## 4. Open Frontend

Open:

```bash
sales_analytics_dashboard.html
```

in your browser.

---

# Machine Learning Workflow

## Data Processing

* Synthetic dataset generation
* Feature engineering
* Scaling using StandardScaler

## Engineered Features

* Advertising efficiency
* Seasonal factors
* Weekly trends
* Inventory turnover
* Engagement ratios

## Ensemble Prediction

The final prediction combines outputs from:

* XGBoost
* Gradient Boosting
* Random Forest
* LightGBM
* Linear Regression

Weighted ensemble improves:

* Accuracy
* Stability
* Generalization

---

# API Endpoint

## Predict Sales

### POST `/predict`

### Example Request

```json
{
  "advertising_spend": 25000,
  "month": 12,
  "day_of_week": 3,
  "inventory": 500,
  "price": 75,
  "discount_percent": 15,
  "website_traffic": 5000,
  "social_media_engagement": 3000,
  "customer_reviews_count": 120,
  "average_rating": 4.5
}
```

### Example Response

```json
{
  "predicted_sales": 8421.52,
  "status": "success"
}
```

---

# Dashboard Modules

* Dashboard Overview
* Revenue Analytics
* Product Analytics
* Customer Insights
* Regional Analysis
* Forecasting
* AI Predictor
* Data Records
* Reporting System

---

# Key Highlights

* Real-time analytics
* AI-powered forecasting
* Multiple ML algorithms
* Interactive visualizations
* Enterprise UI design
* Responsive dashboard
* REST API integration
* Theme customization

---

# Future Improvements

* Database integration (MySQL/PostgreSQL)
* Authentication system
* Live streaming analytics
* Export reports as PDF/Excel
* Cloud deployment
* User role management
* Real-time data ingestion

---

# Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# License

This project is licensed under the MIT License.

---

# Author

Developed by **ISHA RAWAT**

---

# Support

If you like this project, consider starring the repository and sharing it with others.
# sales-analytics-using-5-algo
