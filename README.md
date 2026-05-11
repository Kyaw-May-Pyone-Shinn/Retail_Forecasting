# Retail Decision Intelligence System

A machine learning-powered retail analytics application that combines sales forecasting, store performance monitoring, promotion analysis, and interactive business intelligence dashboards. The project uses historical retail sales data to train forecasting models and presents actionable insights through a Streamlit web application.

## Project Overview

Retail businesses need accurate sales forecasts and store-level performance insights to support decisions around promotions, staffing, inventory planning, and commercial strategy. This project builds an end-to-end retail decision intelligence system that:

- Cleans and prepares historical retail sales data
- Engineers time-based and lag-based forecasting features
- Trains baseline and advanced machine learning regression models
- Uses XGBoost for store-level sales forecasting
- Provides an interactive Streamlit dashboard for business users
- Highlights store risks, promotion effectiveness, and future sales scenarios

The system is designed as a practical business analytics portfolio project, with emphasis on both predictive modelling and decision-oriented dashboarding.

## Key Features

### 1. Executive Overview Dashboard

The executive dashboard provides a high-level summary of business performance, including:

- Total sales
- Customer count
- Average daily sales
- Sales per customer
- Recent sales growth
- 30-day sales forecast
- Promotion uplift
- Count of high-risk stores
- Actual sales versus forecast trend
- Average sales by weekday
- Monthly sales trend
- Top and bottom performing stores

### 2. Store Intelligence

The store intelligence page allows users to analyse individual store performance. It includes:

- Store-level sales KPIs
- 30-day forecast for the selected store
- Promotion uplift analysis
- Sales per customer
- Performance score
- Risk score
- Priority classification
- Sales trend with rolling averages
- Promo versus non-promo sales distribution
- Recommended business action

### 3. Forecast and Scenario Planning

The forecasting module generates future sales projections using the trained XGBoost model. It includes:

- Adjustable forecast horizon from 7 to 60 days
- Baseline forecast
- Promotion scenario
- Best-case scenario
- Worst-case scenario
- Forecast demand profile by date

The scenario views are planning assumptions based on uplift/downside adjustments, while the baseline forecast is generated from the trained machine learning model.

### 4. Business Chatbot

The dashboard includes a simple rule-based business chatbot that answers common management questions such as:

- Performance summary
- KPI overview
- Best performing stores
- Worst performing stores
- Highest sales stores
- Lowest sales stores
- Forecast summary
- Promotion analysis

This feature helps non-technical users interact with the dashboard using business-oriented questions.

## Machine Learning Approach

The modelling workflow is implemented in the Jupyter notebook `Retail_Forecasting.ipynb`.

### Data Preparation

The workflow includes:

- Loading retail sales and store metadata
- Merging sales and store-level information
- Removing closed-store records
- Handling date conversion
- Creating calendar-based features
- Creating lag features
- Creating rolling mean features
- Preparing the final forecasting dataset

### Feature Engineering

The model uses features such as:

- Store ID
- Day of week
- Customers
- Promo indicator
- School holiday indicator
- Store type
- Assortment type
- Competition distance
- Promo2 information
- Year, month, day, and week of year
- Lag sales features
- Rolling average sales features

### Models Used

Two regression models were compared:

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest Regressor | 495.62 | 706.05 |
| XGBoost Regressor | 363.13 | 523.42 |

XGBoost achieved lower error and was selected as the final forecasting model.

### Final Model

The final model is an `XGBRegressor` with the following configuration:

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
```

The trained model is saved as:

```text
model/xgb_sales_forecast_model.pkl
```

## Dashboard Pages

The Streamlit application contains four main pages:

1. **Executive Overview**  
   High-level business performance, sales trends, forecasts, and risk signals.

2. **Store Intelligence**  
   Detailed diagnostics for a selected store, including forecast, promotion response, and risk scoring.

3. **Forecast & Scenarios**  
   Forecast horizon control and business planning scenarios.

4. **Business Chatbot**  
   Rule-based assistant for answering KPI and store-performance questions.

## Project Structure

```text
Retail_Forecasting_ML/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── store.csv
│   │
│   └── processed/
│       └── forecast_dataset.csv
│
├── model/
│   └── xgb_sales_forecast_model.pkl
│
├── notebooks/
│   └── Retail_Forecasting.ipynb
│
├── dashboard.py
├── requirements.txt
└── README.md
```

> Note: Adjust the folder structure if your repository uses different directory names.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- Jupyter Notebook

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
pandas
numpy
scikit-learn
xgboost
streamlit
plotly
joblib
matplotlib
seaborn
jupyter
```

## How to Run the Project

### 1. Train the Model

Open and run the notebook:

```text
notebooks/Retail_Forecasting.ipynb
```

The notebook will:

- Load the raw datasets
- Clean and merge data
- Engineer forecasting features
- Train Random Forest and XGBoost models
- Evaluate model performance
- Save the final XGBoost model
- Export the processed forecasting dataset

### 2. Configure File Paths

Before running the Streamlit dashboard, update the paths in `dashboard.py`:

```python
DATA_PATH = "data/processed/forecast_dataset.csv"
MODEL_PATH = "model/xgb_sales_forecast_model.pkl"
```

Avoid using absolute local paths such as:

```python
C:/Users/your-name/Downloads/...
```

Relative paths make the project easier to run on other machines and on GitHub.

### 3. Run the Streamlit Dashboard

```bash
streamlit run dashboard.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Dataset

The processed dataset used by the dashboard is `forecast_dataset.csv`. It contains store-level daily sales records with engineered forecasting features, including lag variables and rolling sales averages.

Main columns include:

- Store
- Date
- Sales
- Customers
- Promo
- SchoolHoliday
- StoreType
- Assortment
- CompetitionDistance
- Promo2
- Year, month, day, week of year
- Lag features
- Rolling mean features

## Business Value

This project supports retail decision-making by helping users:

- Forecast future sales at store level
- Identify high-risk stores requiring management attention
- Compare top and bottom performing stores
- Evaluate promotion effectiveness
- Understand weekly and monthly sales patterns
- Test simple commercial planning scenarios
- Convert model outputs into business actions

## Limitations

- The scenario analysis uses fixed percentage adjustments rather than separately trained scenario models.
- The chatbot is rule-based and does not use a large language model or external API.
- Forecast quality depends on the availability and reliability of historical sales, customer, promotion, and store-level features.
- The current dashboard uses a locally saved model file and dataset, so paths must be configured correctly before deployment.
- The model does not currently include external factors such as weather, local events, competitor activity, or economic indicators.

## Future Improvements

Potential extensions include:

- Add automated model retraining
- Deploy the dashboard to Streamlit Community Cloud or another cloud platform
- Add authentication for business users
- Add database integration instead of local CSV files
- Include external features such as weather, holidays, and events
- Add model explainability using SHAP
- Replace the rule-based chatbot with an LLM-powered analytics assistant
- Add inventory and staffing recommendation modules


## Author

**Kate**  
MSc Data Analytics  
Business Intelligence and Business Analytics Portfolio Project

## License

This project is intended for academic and portfolio purposes. Add a license file if you plan to make the repository open source.
