import time

from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

class LSTM_model:
    def __init__(self, window_size):
        self.model = Sequential()
        self.model.add(LSTM(input_shape=(window_size,1), output_dim=window_size, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation("linear"))
        
        self.model.compile(loss='mse', optimizer='adam')

    def fit(self, train_X, train_y):
        start = time.time()
        self.model.fit(train_X,train_y, batch_size=512, nb_epoch=3, validation_split=0.1)
        print(f'Training Time : {time.time() - start:0.2f}s')

    def predict(self, input):
        return self.model.predict(input)