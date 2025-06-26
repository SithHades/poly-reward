import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class ARIMAForecaster:
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, series: pd.Series):
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()

    def predict(self, steps=1):
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet.")
        return self.fitted_model.forecast(steps=steps)

    def update(self, new_series: pd.Series):
        # Refit on new data (for simplicity)
        self.fit(new_series)

# TODO: Add LSTM/Prophet support for more advanced forecasting 