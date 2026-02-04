import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def _create_dataset(self, data: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    def train_and_predict(self, df: pd.DataFrame, forecast_days: int) -> np.ndarray:
        data = df[['close']].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = self._create_dataset(scaled_data, self.window_size)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.window_size, 1)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        last_window = scaled_data[-self.window_size:]
        predictions = []
        current_batch = last_window.reshape((1, self.window_size, 1))
        
        for _ in range(forecast_days):
            pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

class XGBoostModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

    def train_and_predict(self, df: pd.DataFrame, forecast_days: int) -> np.ndarray:
        df = df.copy()
        for i in range(1, 6):
            df[f'lag_{i}'] = df['close'].shift(i)
        
        df.dropna(inplace=True)
        X = df[[f'lag_{i}' for i in range(1, 6)]].values
        y = df['close'].values
        
        self.model.fit(X, y)
        
        current_features = list(df['close'].values[-5:])[::-1]
        predictions = []
        
        for _ in range(forecast_days):
            pred = self.model.predict(np.array([current_features]).reshape(1, -1))[0]
            predictions.append(pred)
            current_features = [pred] + current_features[:-1]
            
        return np.array(predictions)

class MonteCarloSimulator:
    @staticmethod
    def simulate(df: pd.DataFrame, forecast_days: int, num_simulations: int = 1000) -> np.ndarray:
        returns = df['close'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        last_price = df['close'].iloc[-1]
        simulated_paths = np.zeros((num_simulations, forecast_days))
        
        for i in range(num_simulations):
            prices = [last_price]
            for _ in range(forecast_days):
                prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
            simulated_paths[i, :] = prices[1:]
            
        return np.mean(simulated_paths, axis=0)

class PredictionEnsemble:
    def __init__(self):
        self.lstm = LSTMModel()
        self.xgb = XGBoostModel()
        self.mc = MonteCarloSimulator()

    def get_ensemble_forecast(self, df: pd.DataFrame, forecast_days: int):
        lstm_pred = self.lstm.train_and_predict(df, forecast_days)
        xgb_pred = self.xgb.train_and_predict(df, forecast_days)
        mc_pred = self.mc.simulate(df, forecast_days)
        
        ensemble_pred = (lstm_pred * 0.4) + (xgb_pred * 0.4) + (mc_pred * 0.2)
        std_dev = df['close'].pct_change().std() * df['close'].iloc[-1]
        
        return {
            "forecast": ensemble_pred.tolist(),
            "probs": {
                "30%_range": [ensemble_pred[-1] - 0.5 * std_dev, ensemble_pred[-1] + 0.5 * std_dev],
                "50%_range": [ensemble_pred[-1] - 1.0 * std_dev, ensemble_pred[-1] + 1.0 * std_dev],
                "20%_range": [ensemble_pred[-1] - 1.5 * std_dev, ensemble_pred[-1] + 1.5 * std_dev]
            }
        }
