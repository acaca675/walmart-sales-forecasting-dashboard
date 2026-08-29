# Walmart Sales Forecasting & Retail Intelligence

**End-to-End Data Analytics, Forecasting & Business Intelligence Project**

**Author:** Nabilla Salwa Salsabilla
**LinkedIn:** https://www.linkedin.com/in/nabillasalsa/
**Repository:** https://github.com/acaca675/walmart-sales-forecasting-dashboard

[![Python](https://img.shields.io/badge/Python-3.x-3776AB)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-1F6FEB)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458)](https://pandas.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Model_Evaluation-F7931E)](https://scikit-learn.org/)

---

## Results at a Glance

| | |
|---|---:|
| Test-period WAPE | 8.07% |
| Test-period R² | 0.9853 |
| MAE improvement over naive baseline | 17.17% |
| RMSE improvement over naive baseline | 22.25% |
| Strongest predictor | Rolling 4-week sales average |
| Model bias | Underprediction (median error 31.43) |
| Forecast horizon | 12 weeks, 690.7M total projected sales |

<details>
<summary><strong>Table of Contents</strong></summary>

**Part I — Business Case**
[Executive Summary](#executive-summary) · [Business Context](#1-business-context) · [Business Problem](#2-business-problem) · [Business Questions](#3-business-questions)

**Part II — Data & Methodology**
[Dataset](#4-dataset) · [Analytical Approach](#5-analytical-approach) · [Train / Test Strategy](#6-train--test-strategy) · [Forecasting Methodology & Model Evaluation](#7-forecasting-methodology--model-evaluation)

**Part III — Results & Insights**
[What Drives the Predictions](#8-what-drives-the-predictions) · [Forecast Error Analysis](#9-forecast-error-analysis) · [Future Forecast & Assumptions](#10-future-forecast--assumptions) · [Business Implications](#11-business-implications)

**Part IV — Using This Project**
[Interactive Dashboard](#12-interactive-dashboard) · [Project Architecture](#13-project-architecture) · [Reproducibility](#14-reproducibility) · [Limitations](#15-limitations) · [Future Improvements](#16-future-improvements) · [Key Takeaways](#key-takeaways)

</details>

---

## Executive Summary

This project analyzes Walmart's historical weekly sales data and builds a forecasting pipeline to support near-term sales planning. It combines exploratory analysis, time-series feature engineering, a naive baseline, an XGBoost regression model, forecast error diagnostics, and a 12-week forward forecast — presented through an interactive Streamlit dashboard.

The goal is not a high R² in isolation. It is to show how raw transactional data becomes a decision-support tool: a model whose output can be translated into inventory, staffing, and planning actions.

```text
Raw Data → Validation → Feature Engineering → Exploratory Analysis →
Baseline Forecast → XGBoost Model → Evaluation → Error Analysis →
12-Week Forecast → Business Insight → Dashboard
```

---

# Part I — Business Case

## 1. Business Context

Retail sales fluctuate across stores, departments, seasons, holidays, and macroeconomic conditions. For a network the size of Walmart's, that variability directly affects inventory replenishment, workforce scheduling, promotional planning, and supply chain decisions.

Historical data explains what already happened. Forecasting answers a different question: **what level of sales should the business expect over the coming weeks** — and this project addresses that from both an analytics and a decision-support angle.

---

## 2. Business Problem

The project centers on three connected questions:

- **Performance** — which stores, departments, and periods drive Walmart's sales?
- **Drivers** — which historical and contextual variables carry the strongest predictive signal?
- **Forecasting** — does machine learning meaningfully outperform a naive historical baseline?

In short: descriptive analytics, predictive analytics, and business interpretation, treated as one workflow rather than three separate exercises.

---

## 3. Business Questions

**Sales performance**
1. How does weekly sales performance trend over time?
2. Which stores and departments generate the most revenue?
3. How does performance vary across months, quarters, and years?

**Seasonality & holidays**
4. How do holiday periods affect weekly sales?
5. Which periods need closer forecasting attention?

**Forecasting**
6. Can historical patterns predict future weekly sales?
7. Does XGBoost outperform a naive baseline?
8. Which features contribute most to the model's predictions?

**Reliability & decision-making**
9. Where does the model make its largest errors, and does it over- or under-predict?
10. What assumptions does forecasting beyond the historical window require?
11. What should management actually do with the forecast?

---

# Part II — Data & Methodology

## 4. Dataset

| Metric | Value |
|---|---:|
| Total rows | 421,570 |
| Stores | 45 |
| Departments | 81 |
| Historical period | 2010–2012 |
| Observation level | Store × Department × Week |
| Target variable | Weekly Sales |

Contextual variables include store, department, weekly sales, holiday indicator, temperature, fuel price, CPI, unemployment, store size, store type, and date-derived fields.

---

## 5. Analytical Approach

Before any modeling, the dataset was validated for date consistency, store and department coverage, missing values, duplicates, and chronological ordering — confirming the full 421,570-row, 45-store, 81-department structure described above.

**Feature engineering** produced 21 modeling features across four groups:

| Group | Features |
|---|---|
| Store & product structure | Store, Dept, Size, Type_Code |
| Calendar | Year, Month, Week, Quarter, DayOfWeek, IsWeekend |
| External context | Temperature, Fuel_Price, CPI, Unemployment, IsHoliday |
| Historical sales | Lag_1_Week_Sales, Lag_4_Week_Sales, Lag_12_Week_Sales, Rolling_Mean_4, Rolling_Mean_12, Rolling_Std_4 |

The lag and rolling-average features matter most here: weekly retail sales are strongly autocorrelated, and recent history turns out to carry more signal than most static variables (see Section 7).

---

## 6. Train / Test Strategy

The split is chronological, not random — a random split would let future information leak into training, which defeats the purpose of a forecasting evaluation.

| Split | Period | Rows |
|---|---|---:|
| Training | 2010-04-30 → 2012-04-20 | 303,345 |
| Testing | 2012-04-27 → 2012-10-26 | 79,610 |

This mirrors how a forecasting model would actually operate: trained on the past, evaluated on a genuinely unseen future window.

---

## 7. Forecasting Methodology & Model Evaluation

A naive baseline was included deliberately — a model should demonstrate real improvement over a simple historical guess, not be judged in isolation.

| Metric | Naive Baseline | XGBoost | Improvement |
|---|---:|---:|---:|
| MAE | 1,546.01 | 1,280.50 | 17.17% |
| RMSE | 3,431.09 | 2,667.55 | 22.25% |
| R² | 0.9757 | 0.9853 | — |
| WAPE | 9.75% | 8.07% | — |

XGBoost was selected for its ability to capture non-linear relationships, feature interactions, seasonality, and store/department-level differences. The largest gain is in RMSE, meaning the model is specifically better at controlling large individual errors, not just average error.

---

# Part III — Results & Insights

## 8. What Drives the Predictions

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Rolling_Mean_4 | 0.4257 |
| 2 | Lag_1_Week_Sales | 0.3913 |
| 3 | Lag_4_Week_Sales | 0.0473 |
| 4 | Rolling_Mean_12 | 0.0440 |
| 5 | IsHoliday | 0.0163 |
| 6 | Quarter | 0.0150 |
| 7 | Month | 0.0122 |
| 8 | Week | 0.0103 |
| 9 | Dept | 0.0068 |
| 10 | Rolling_Std_4 | 0.0067 |

The four-week rolling mean and the one-week lag alone account for over 80% of total feature importance. In this setup, **recent sales momentum outpredicts nearly every static store or economic variable** — a finding that should shape how planning teams weight recent trends versus long-run averages.

---

## 9. Forecast Error Analysis

| Metric | Value |
|---|---:|
| Mean error | 132.53 |
| Median error | 31.43 |

The model leans toward **underprediction**. This is not a cosmetic detail: systematic underprediction can translate into understocking, understaffing during demand spikes, and overly conservative revenue planning. It should be monitored as its own metric, not absorbed into a single accuracy score.

---

## 10. Future Forecast & Assumptions

| Metric | Value |
|---|---:|
| Forecast horizon | 12 weeks (2012-11-02 → 2013-01-18) |
| Total forecast | 690,732,346.36 |
| Average weekly forecast | 57,561,028.86 |
| Peak week | 2012-12-21 (83,575,734.32) |
| Lowest week | 2013-01-18 (47,394,976.19) |

The forecast shows a clear build-up through the December holiday period, followed by a drop into January.

Forecasting beyond the historical window requires stated assumptions: future calendar dates are known and holiday status can be derived from them, store and department structures stay consistent, historical feature relationships hold, and economic variables (CPI, unemployment, fuel price) don't undergo structural shocks. The output should be read as **a model-based scenario under these assumptions, not a guaranteed outcome.**

---

## 11. Business Implications

| Finding | Implication | Recommended Action |
|---|---|---|
| Rolling mean (4-week) and 1-week lag dominate predictive power | Recent sales momentum matters more than annual or long-term averages | Track short-term trends continuously rather than relying on yearly baselines |
| Holiday-related features and December forecast spike are material | Seasonal demand is real and model-detectable | Prepare inventory, staffing, and capacity ahead of the December period specifically |
| Model shows a measurable underprediction bias | Blind trust in point forecasts risks understocking | Apply a safety buffer or scenario range for high-demand periods rather than using the raw forecast number |
| Aggregate accuracy (WAPE 8.07%) masks store/department variation | A strong network-level score doesn't guarantee even performance everywhere | Pair the forecast with store- and department-level error monitoring to flag underperforming segments |

**Decision loop:** forecast → identify high-demand periods → review store/department risk → adjust inventory and staffing → monitor actuals against forecast → retrain as new data arrives.

---

# Part IV — Using This Project

## 12. Interactive Dashboard

The results are surfaced through a Streamlit dashboard built as a business-intelligence interface, not a static model report.

| Page | Purpose |
|---|---|
| Executive Overview | Network-wide KPIs and revenue trend |
| Sales Analytics | Store, department, seasonality, and holiday breakdown |
| Forecasting | Model diagnostics, actual vs. predicted, feature importance, 12-week forecast |
| Store Performance | Store-level benchmarking and trend |
| Dataset Explorer | Interactive view of the processed dataset and data quality |
| About Platform | Methodology, architecture, and how to interpret the forecast |

**Preview**

  ![Executive Overview](data/Walmart dasboard.png)
  
---

## 13. Project Architecture

```text
walmart-sales-forecasting-dashboard/
├── app.py
├── walmart_xgboost_pro.py
├── requirements.txt
├── README.md
└── data/
    ├── raw/
    └── processed/
        ├── walmart_featured.csv
        ├── model_evaluation.csv
        ├── test_predictions.csv
        ├── future_sales_forecast.csv
        ├── future_total_sales_forecast.csv
        ├── store_forecast_error.csv
        ├── department_forecast_error.csv
        ├── xgboost_feature_importance.csv
        ├── xgboost_actual_vs_predicted.png
        ├── xgboost_feature_importance.png
        └── future_sales_forecast.png
```

Trained model artifacts are excluded from version control where appropriate and can be regenerated by running the forecasting pipeline.

---

## 14. Reproducibility

```bash
git clone https://github.com/acaca675/walmart-sales-forecasting-dashboard.git
cd walmart-sales-forecasting-dashboard

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python3 walmart_xgboost_pro.py   # regenerate the forecasting pipeline outputs
streamlit run app.py             # launch the dashboard at http://localhost:8501
```

---

## 15. Limitations

- **Historical coverage** spans a limited window (2010–2012), so longer-term structural shifts may not be represented.
- **Future external variables** — CPI, unemployment, fuel price — aren't known with certainty; the forecast inherits that uncertainty.
- **Forecast horizon** is fixed at 12 weeks; accuracy is expected to degrade at longer horizons.
- **Interpretability** — XGBoost trades some transparency for predictive power. Feature importance reflects predictive contribution, not causal effect.
- **Holiday effects** observed historically may not repeat identically in future periods.

---

## 16. Future Improvements

- Rolling-origin cross-validation; comparison against LightGBM and classical time-series models
- Prediction intervals and multi-horizon evaluation
- Store–department interaction features, exponentially weighted moving averages, promotional/event variables
- Automated drift detection, scheduled retraining, store-level diagnostics
- Scenario analysis, confidence ranges, and an inventory-recommendation layer on top of the dashboard

---

## Key Takeaways

- XGBoost improves on the naive baseline by 17.17% (MAE) and 22.25% (RMSE), reaching a WAPE of 8.07% on a chronological test split.
- Recent sales history — not store size or economic indicators — is the dominant predictive signal.
- The model underpredicts on average, which has direct operational implications and should be monitored explicitly.
- Aggregate performance should always be paired with store- and department-level error checks before being used operationally.
- Every forward-looking number in this project is a scenario under stated assumptions, not a commitment.

---

## Author

**Nabilla Salwa Salsabilla**
Data Analytics · Business Intelligence · Forecasting

[LinkedIn](https://www.linkedin.com/in/nabillasalsa/) · [Repository](https://github.com/acaca675/walmart-sales-forecasting-dashboard)
