# Seoul Bike Demand Forecasting | Regression, Holt-Winters & SARIMAX

### Time Series Forecasting, Statistical Modeling, and Weather-Based Demand Prediction with Python

**Project Type:** Applied Data Science and Time Series Forecasting  
**Field:** Data Science | Machine Learning | Statistical Forecasting | Urban Mobility  
**Programming Language:** Python  
**Environment:** Google Colab | Jupyter Notebook  
**Author:** Data Engineer Luis Barahona
## Holt-Winters Forecasting — Visual Reference

![Holt-Winters Method — Minitab](https://support.minitab.com/es-mx/minitab/media/generated-content/images/winters_method_utility_usage.PNG)

*Figure: Holt-Winters seasonal forecasting example. Source: Minitab.*

**Reference:** [Minitab — Interpreting Key Results for Winters' Method](https://support.minitab.com/es-mx/minitab/help-and-how-to/statistical-modeling/time-series/how-to/winters-method/interpret-the-results/key-results/)

**Image credit:** © Minitab, LLC. All rights reserved.
---

## 1. Project Overview

This project explores statistical and machine learning techniques to forecast daily bicycle rental demand in Seoul, South Korea.

Using the Seoul Bike Sharing Demand dataset, the notebook investigates relationships between bicycle rentals, historical demand, time, and meteorological conditions.

The study compares five forecasting configurations based on three main approaches:

1. Linear Regression
2. Holt-Winters Exponential Smoothing
3. SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Variables)

The objective is to evaluate how different forecasting methods capture demand patterns and how weather-related variables may contribute to prediction accuracy.

---

## 2. Project Objectives

- Import and prepare historical bicycle rental data.
- Explore hourly and daily rental demand.
- Identify missing values and duplicate records.
- Aggregate hourly observations into daily demand.
- Analyze meteorological variables related to bicycle rentals.
- Implement simple and multiple linear regression.
- Apply moving averages and exponential smoothing.
- Compare Holt-Winters configurations.
- Implement SARIMAX with exogenous variables.
- Evaluate forecasting performance using statistical metrics.
- Visualize observed demand and predicted values.

---

## 3. Dataset Description

**Dataset:** Seoul Bike Sharing Demand

**Required file:** `SeoulBikeData.csv`

**Location:** Seoul, South Korea

The dataset contains historical bicycle rental observations and meteorological information.

### Main Variables

| Variable | Description |
|:---|:---|
| Date | Observation date |
| Rented Bike Count | Number of rented bicycles |
| Hour | Hour of observation |
| Temperature | Air temperature in °C |
| Humidity | Relative humidity (%) |
| Wind speed | Wind speed in m/s |
| Visibility | Visibility measurements |
| Dew point temperature | Dew point in °C |
| Solar Radiation | Solar radiation |
| Rainfall | Rainfall in millimeters |
| Snowfall | Snowfall in centimeters |
| Seasons | Seasonal classification |
| Holiday | Holiday indicator |
| Functioning Day | Service availability indicator |

### Target Variable

The original dataset contains hourly observations.

The notebook transforms these observations into daily totals:

```python
daily_demand = df.groupby('Date')['Rented Bike Count'].sum()
```

The resulting target variable represents the total number of bicycle rentals recorded per day.

---

## 4. Data Preparation

### 4.1 Data Loading

The dataset is imported using Pandas.

The notebook uses ISO-8859-1 encoding to read the CSV file.

### 4.2 Data Cleaning

The preprocessing workflow includes:

- Missing-value inspection.
- Data type verification.
- Date conversion.
- Duplicate detection.
- Duplicate removal.
- Categorical variable formatting.

### 4.3 Feature Engineering

Additional features are created, including:

- Day of the week.
- Daily bicycle rental totals.
- Daily averages of meteorological variables.
- Lagged temperature observations.
- Lagged humidity observations.

These transformations prepare the dataset for statistical forecasting.

---

## 5. Training and Testing Strategy

The notebook selects the final 50 days of the daily demand series.

The observations are divided chronologically:

| Dataset | Observations |
|:---|---:|
| Training | 40 days |
| Testing | 10 days |
| Total | 50 days |

This chronological separation helps preserve the temporal ordering of observations.

However, the evaluation covers a relatively short period, limiting the ability to establish long-term forecasting performance.

---

## 6. Forecasting Models

### Model 1 — Simple Linear Regression

Simple Linear Regression uses the observation's temporal position as a predictor of daily rental demand.

The model is expressed as:

```text
Predicted Demand = Intercept + Slope × Time
```

The purpose is to establish a simple forecasting baseline based on the overall time trend.

### Model 2 — Multiple Linear Regression

Multiple Linear Regression incorporates meteorological variables to estimate bicycle rental demand.

The input variables include:

- Temperature
- Humidity
- Wind speed
- Visibility
- Dew point temperature
- Solar radiation
- Rainfall
- Snowfall

Daily weather averages are used to construct the model's feature matrix.

This experiment investigates how external environmental conditions relate to bicycle rental activity.

### Model 3 — Additive Holt-Winters

The Holt-Winters method extends exponential smoothing by incorporating trend and seasonality.

The notebook implements additive trend and additive seasonality with a weekly seasonal period:

```python
ExponentialSmoothing(
    train['Total Daily Demand'],
    trend='add',
    seasonal='add',
    seasonal_periods=7
)
```

The model produces forecasts for the 10-day testing period.

### Model 4 — Holt-Winters with Damped Trend

A second Holt-Winters configuration includes an additive damped trend.

The damped trend reduces the impact of extrapolating an existing trend too aggressively into the future.

The configuration uses:

```python
ExponentialSmoothing(
    train['Total Daily Demand'],
    trend='add',
    damped_trend=True,
    seasonal='add',
    seasonal_periods=7
)
```

Its performance is compared with the original additive Holt-Winters model.

### Model 5 — SARIMAX

SARIMAX extends ARIMA by incorporating seasonal components and exogenous variables.

The notebook uses daily meteorological features and lagged observations to construct the explanatory variable matrix.

The SARIMAX model includes:

- Nonseasonal autoregressive components.
- Differencing.
- Moving-average components.
- Seasonal terms.
- Exogenous weather variables.

A parameter search evaluates combinations of:

```text
p, d, q, P, D, Q ∈ {0, 1}
```

The seasonal period is configured as seven days.

The Akaike Information Criterion (AIC) is used to select a candidate model.

### Rolling One-Step Forecasting

The SARIMAX experiment generates sequential one-step forecasts.

After each prediction, the actual observed demand is added to the historical data before estimating the next day.

Therefore, the procedure represents a rolling one-step-ahead evaluation rather than a single fixed-origin 10-day forecast.

---

## 7. Experimental Results

The following results were extracted from the notebook's saved execution outputs.

### Forecasting Performance

| Model | Test MAE | Test RMSE | Test R² |
|:---|---:|---:|---:|
| Simple Linear Regression | 4,341.33 | 4,626.45 | -1.06 |
| Multiple Linear Regression | 3,586.54 | 4,079.83 | -0.60 |
| Additive Holt-Winters | 3,173.59 | **3,635.30** | — |
| Holt-Winters with Damped Trend | **3,134.90** | 3,702.87 | — |
| SARIMAX | 5,493.83 | Not separately reported | — |

### SARIMAX Additional Metric

The notebook reports:

**MAPE: approximately 37%**

The saved output reports this value as a decimal fraction of approximately 0.37.

### Interpretation

The results show that:

- Multiple Linear Regression reduced both MAE and RMSE compared with Simple Linear Regression.
- Additive Holt-Winters achieved the lowest recorded RMSE.
- The damped Holt-Winters configuration achieved the lowest recorded MAE.
- SARIMAX reported a higher MAE than the other evaluated models.
- Both linear regression models produced negative testing R² values, indicating performance below a constant-mean predictor under this metric.

The results do not demonstrate that SARIMAX was the most accurate model.

Additionally, the SARIMAX rolling evaluation uses updated actual observations, making its forecasting protocol different from the fixed-origin approaches.

A strictly controlled comparison requires consistent forecast origins, information availability, and evaluation procedures across all models.

---

## 8. Evaluation Metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between observed and predicted bicycle rental counts.

Lower values indicate smaller average prediction errors.

### Root Mean Squared Error (RMSE)

RMSE places greater emphasis on larger forecasting errors.

It helps identify models that produce significant deviations from observed demand.

### Coefficient of Determination (R²)

R² evaluates model performance relative to a constant-mean prediction benchmark.

Negative R² values indicate that predictions perform worse than that benchmark.

### Mean Absolute Percentage Error (MAPE)

MAPE measures relative forecasting errors as a percentage.

The metric requires caution when actual demand values are zero or close to zero.

---

## 9. Data Visualization

The notebook includes:

- Historical daily rental demand plots.
- Training and testing comparisons.
- Simple Linear Regression forecasts.
- Multiple Linear Regression forecasts.
- Seven-day moving averages.
- Simple Exponential Smoothing forecasts.
- Holt-Winters forecasting graphs.
- SARIMAX rolling predictions.
- Model comparison charts.
- MAE and RMSE bar charts.

These visualizations support the interpretation of demand patterns and differences between model predictions.

---

## 10. Technologies Used

| Technology | Purpose |
|:---|:---|
| Python | Programming language |
| Pandas | Data processing and time series preparation |
| NumPy | Numerical calculations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Regression and evaluation metrics |
| Statsmodels | Exponential smoothing and SARIMAX |
| Google Colab | Cloud notebook environment |
| Jupyter Notebook | Interactive analysis |

---

## 11. How to Run

### Google Colab

[Open the Seoul Bike Demand Forecasting Notebook](https://colab.research.google.com/drive/1w7RbacGG-U8mA-N6tRHtcPXBAO5C9MiE)

### Execution Instructions

1. Open the notebook in Google Colab.
2. Upload the `SeoulBikeData.csv` dataset.
3. Execute the import and preprocessing cells.
4. Generate the daily demand series.
5. Run the regression experiments.
6. Execute the Holt-Winters models.
7. Run the SARIMAX parameter search.
8. Review the evaluation metrics and visualizations.

### Local Python Environment

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels notebook
```

The notebook contains Google Colab-specific functionality and may require modifications for local execution.

---

## 12. Important Limitations

This notebook is an academic forecasting experiment.

Important methodological considerations include:

**Short training period**

The experiment uses only 40 training observations and 10 testing observations.

**Holt-Winters convergence**

The saved outputs contain optimization convergence warnings for the Holt-Winters models.

Their fitted configurations should be reviewed before interpreting the results as definitive.

**SARIMAX convergence**

Some SARIMAX configurations also produce convergence warnings.

A low AIC value does not automatically guarantee reliable forecasting performance.

**Exogenous variables**

The notebook uses weather observations from the testing period.

For operational forecasting, these variables would need to be available in advance through weather forecasts or another valid information source.

**Evaluation consistency**

SARIMAX uses sequential updating with actual testing observations, while the other models generate predictions using different procedures.

This limits direct comparability.

**Variable management**

Some variables are reused and overwritten across model sections.

Clearer variable names would improve reproducibility.

---

## 13. Future Improvements

Potential improvements include:

- Expanding the historical training window.
- Implementing rolling-origin cross-validation.
- Using identical forecasting protocols for every model.
- Incorporating weather forecasts rather than realized future observations.
- Comparing different seasonal periods.
- Investigating model convergence warnings.
- Adding confidence and prediction intervals.
- Including seasonal naïve forecasting baselines.
- Evaluating additional regression and machine learning models.
- Developing an interactive demand forecasting dashboard.

---

## 14. Practical Applications

The techniques explored in this notebook can support research into:

- Bicycle-sharing demand forecasting.
- Urban mobility analytics.
- Transportation planning.
- Fleet availability analysis.
- Weather-sensitive demand modeling.
- Operational resource allocation.
- Smart city data analytics.

These methods may help build future forecasting tools when supported by reliable datasets and appropriate validation.

---

## 15. Conclusion

This project demonstrates how statistical and machine learning techniques can be applied to bicycle-sharing demand forecasting.

By comparing regression models, Holt-Winters exponential smoothing, and SARIMAX, the notebook illustrates the impact of temporal patterns and meteorological variables on prediction methods.

In the recorded experiments, additive Holt-Winters achieved the lowest RMSE, while its damped-trend variant achieved the lowest MAE.

Although SARIMAX offers a flexible framework for incorporating seasonal behavior and external variables, the saved evaluation does not demonstrate superior predictive performance.

The project provides a practical foundation for further research in time series forecasting, mobility analytics, and statistical model evaluation.

---

## 16. Project Resources

### Google Colab Notebook

[Open the Original Notebook](https://colab.research.google.com/drive/1w7RbacGG-U8mA-N6tRHtcPXBAO5C9MiE)

### Minitab — Winters Method Interpretation

[Interpreting Key Results for Winters' Method](https://support.minitab.com/es-mx/minitab/help-and-how-to/statistical-modeling/time-series/how-to/winters-method/interpret-the-results/key-results/)

This reference explains the interpretation of fitted values, forecasting accuracy measures, and the importance of examining seasonal patterns when evaluating Winters models.

### Statsmodels

[Statsmodels — Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)

---

## Author

**Data Engineer Luis Barahona**

Mexican Data Engineer and 2026 graduate of Universidad del Caribe, passionate about Data Engineering, Data Science, Machine Learning, Artificial Intelligence, and continuous professional development.

---

## Contact

| Platform | Contact |
|:---|:---|
| Email | [190300395@ucaribe.edu.mx](mailto:190300395@ucaribe.edu.mx) |
| GitHub | [github.com/LuisBarahonaMX](https://github.com/LuisBarahonaMX) |
| LinkedIn | [linkedin.com/in/luis-barahona-mx](https://www.linkedin.com/in/luis-barahona-mx/) |
| Instagram | [@luisbarahonamx](https://www.instagram.com/luisbarahonamx/) |
| X | [@luisbarahonaMX](https://x.com/luisbarahonaMX) |
| Website | [SIKO AI Platform](https://sikoai.web.app/) |
