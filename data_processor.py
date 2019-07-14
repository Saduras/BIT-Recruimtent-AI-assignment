import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, window_size, forecast_size=1, shift=1,split_ratio=0.8):
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.shift = shift
        self.split_ratio = split_ratio
        self.scaler = MinMaxScaler(feature_range=(-1,1))

    def preprocess(self, df):
        x = np.array(df['value']).reshape(-1,1)

        # translate to differences?
        diffs = []
        for i in range(1, len(x)):
            diffs.append(x[i] - x[i-1])

        # normalize values
        scaled = self.scaler.fit_transform(diffs)
        series = pd.DataFrame(scaled)

        # create windows
        series_s = series.copy()
        for i in range(self.window_size + self.forecast_size - 1):
            series = pd.concat([series, series_s.shift(-(i+self.shift))], axis=1)
        series.dropna(axis=0, inplace=True)

        # split into train & test set
        nsplit = round(self.split_ratio * series.shape[0])

        train = series.iloc[:nsplit, :]
        test = series.iloc[nsplit:, :]

        train_X = train.iloc[:,:-self.forecast_size].values
        train_y = train.iloc[:,-self.forecast_size:].values
        test_X = test.iloc[:,:-self.forecast_size].values
        test_y = test.iloc[:,-self.forecast_size:].values

        train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
        test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)

        return train_X, train_y, test_X, test_y, diffs

    def postprocess(self, y):
        return self.scaler.inverse_transform(y)