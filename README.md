# Walmart Sales Forecasting and Retail Analytics Platform

## Project Overview

This project is an end-to-end retail analytics and sales forecasting platform developed using Python, Machine Learning, XGBoost, and Streamlit. The system is designed to analyze large-scale Walmart retail sales data, generate forecasting insights, and provide an interactive business intelligence dashboard for operational and strategic decision-making.

The project simulates a real-world enterprise analytics workflow commonly used in retail, supply chain, and business intelligence environments.

The solution includes:

* Data Cleaning and Preprocessing Pipeline
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning Forecasting
* XGBoost Predictive Modeling
* Business KPI Analytics
* Interactive Streamlit Dashboard
* Future Sales Forecasting

---

# Business Problem

Retail companies experience significant fluctuations in weekly sales due to:

* Seasonal demand
* Holiday events
* Promotional markdowns
* Economic conditions
* Store characteristics
* Customer purchasing behavior

Without accurate forecasting systems, companies may experience:

* Overstocking
* Inventory shortages
* Revenue loss
* Poor supply chain planning
* Inefficient resource allocation

This project aims to build a forecasting system capable of predicting weekly retail sales using historical sales data and operational business features.

---

# Project Objectives

The main objectives of this project are:

1. Build a professional retail forecasting pipeline
2. Analyze sales behavior and business performance
3. Generate accurate weekly sales predictions
4. Create an interactive analytics dashboard
5. Simulate enterprise-level data science workflow
6. Provide business-ready forecasting insights

---

# Dataset Information

Dataset Source:

* Walmart Retail Sales Forecasting Dataset

Main datasets used:

| Dataset      | Description                                  |
| ------------ | -------------------------------------------- |
| train.csv    | Historical weekly sales data                 |
| features.csv | Economic indicators and markdown information |
| stores.csv   | Store metadata including type and size       |

Dataset scale:

* 421,570 sales records
* 45 stores
* 81 departments
* 143 weekly periods

---

# Technology Stack

## Programming and Analytics

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

## Data Visualization

* Matplotlib
* Seaborn

## Dashboard and Application

* Streamlit

## Development Environment

* macOS Terminal
* Nano Editor
* Homebrew

## Version Control

* Git
* GitHub

---

# Project Workflow

## 1. Data Cleaning and Preprocessing

The first stage focused on building a reliable data processing pipeline:

* Data loading and validation
* Data type conversion
* Dataset merging
* Missing value handling
* Duplicate removal
* Categorical feature encoding
* Dataset integrity validation

Output:

* walmart_clean.csv

---

## 2. Exploratory Data Analysis (EDA)

Comprehensive business and statistical analysis was performed to understand sales behavior and operational patterns.

EDA included:

* Sales distribution analysis
* Monthly sales trend analysis
* Holiday impact analysis
* Store performance analysis
* Department performance analysis
* Correlation analysis
* Seasonality analysis
* Outlier detection

Generated visual analytics:

* correlation_heatmap.png
* monthly_sales_trend.png
* holiday_sales_comparison.png
* seasonality_analysis.png
* top_departments.png
* store_type_performance.png

---

## 3. Feature Engineering

Advanced time-series and forecasting features were developed to improve predictive performance.

Engineered features:

| Feature Category  | Features                           |
| ----------------- | ---------------------------------- |
| Time Features     | Year, Month, Week, Quarter         |
| Calendar Features | DayOfWeek, IsWeekend               |
| Lag Features      | Lag_1_Week_Sales, Lag_4_Week_Sales |
| Rolling Features  | Rolling_Mean_4, Rolling_Mean_12    |

Output:

* walmart_featured.csv

---

## 4. Machine Learning Forecasting

Two forecasting models were developed and evaluated.

### Random Forest Regressor

Performance:

| Metric   | Score    |
| -------- | -------- |
| MAE      | 1,294.39 |
| RMSE     | 4,439.88 |
| R² Score | 0.9456   |

---

### Professional XGBoost Forecasting

Performance:

| Metric   | Score    |
| -------- | -------- |
| MAE      | 1,312.98 |
| RMSE     | 2,772.87 |
| R² Score | 0.9840   |

The XGBoost model delivered significantly stronger predictive accuracy and became the final forecasting model used in the dashboard application.

Generated outputs:

* pro_xgboost_results.csv
* future_sales_forecast.csv
* pro_xgboost_forecast.png
* pro_xgboost_feature_importance.png

---

# Dashboard Application

A professional Streamlit dashboard was developed to visualize business KPIs, forecasting outputs, and sales analytics.

Dashboard capabilities:

* Business KPI overview
* Sales trend visualization
* Forecast comparison analysis
* Store-level analytics
* Interactive filtering
* Retail performance monitoring
* Forecast visualization

Main application file:

* app.py

---

# Business Insights

Several business insights were identified during analysis:

## Holiday Impact

Holiday periods generate higher average sales compared to regular weeks.

## Seasonal Behavior

Sales increase significantly during Q4, especially in November and December.

## Store Performance

Store Type A consistently produces the highest sales performance.

## Forecasting Drivers

The most important forecasting variables were:

* Previous week sales
* Rolling sales averages
* Seasonal calendar effects
* Holiday indicators

---

# Project Structure

```bash
walmart_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── screenshots/
│
├── app.py
├── walmart_cleaning.py
├── walmart_eda.py
├── walmart_feature_engineering.py
├── walmart_forecasting.py
├── walmart_xgboost_pro.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dashboard Deployment

The application can be deployed using:

* Streamlit Community Cloud
* Render
* Railway
* Docker
* AWS EC2

---

# Installation Guide

## Clone Repository

```bash
git clone https://github.com/your-username/walmart-sales-forecasting-dashboard.git
```

## Open Project Directory

```bash
cd walmart-sales-forecasting-dashboard
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# Future Improvements

Potential enterprise-level enhancements:

* Real-time forecasting
* API integration
* Power BI integration
* Cloud deployment
* User authentication
* Automated retraining pipeline
* Deep learning forecasting models
* Retail inventory optimization
* Demand planning system

---

# Author

Nabilla Salsabilla

Business Analytics | Data Analytics | Machine Learning | Forecasting | Supply Chain Analytics

---

# Project Status

Status: Production-Ready Portfolio Project

This project was developed as a professional data analytics and forecasting portfolio designed to demonstrate practical enterprise-level skills in:

* Retail analytics
* Business intelligence
* Machine learning forecasting
* Dashboard development
* Data engineering workflow
* Predictive analytics

