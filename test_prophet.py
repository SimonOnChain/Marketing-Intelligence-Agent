"""Test Prophet forecasting."""
import pandas as pd
import numpy as np
from prophet import Prophet

# Test data
data = pd.DataFrame({
    'ds': pd.date_range('2018-01-01', periods=20, freq='W'),
    'y': [200000 + i*5000 + np.random.normal(0, 20000) for i in range(20)]
})

print("Input data:")
print(data.tail())

# Log transform
data_log = data.copy()
data_log['y'] = np.log1p(data_log['y'])

model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model.fit(data_log)

future = model.make_future_dataframe(periods=4, freq='W')
forecast = model.predict(future)

# Show last 5 rows with inverse transform
print("\nForecast (log-transformed back):")
for _, row in forecast.tail(6).iterrows():
    val = np.expm1(row['yhat'])
    print(f"{row['ds'].strftime('%Y-%m-%d')}: {val:,.0f}")
