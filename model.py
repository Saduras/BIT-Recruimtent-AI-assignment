import time

from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import load_model, save_model

class LSTMModel():
    def __init__(self, window_size, forecast_size):
        self.model = Sequential()
        self.model.add(LSTM(input_shape=(window_size,1), 
                            output_dim=window_size, 
                            return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(forecast_size))
        self.model.add(Activation("linear"))
        
        self.model.compile(loss='mse', optimizer='adam')

    def fit(self, train_X, train_y, epochs=10):
        start = time.time()
        self.model.fit(train_X, train_y, 
                        batch_size=512, 
                        epochs=epochs, 
                        validation_split=0.1, 
                        shuffle=False)
        print(f'Training Time : {time.time() - start:0.2f}s')

    def predict(self, input):
        return self.model.predict(input)

    def load(self, path):
        self.model = load_model(path)
        print(f'loaded model from {path}')

    def save(self, path):
        save_model(self.model, path)
        print(f'saved model at {path}')