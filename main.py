import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.metrics import mean_squared_error

from model import LSTMModel
from data_processor import DataProcessor

def plot_data(x, y, label=''):
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(dates.MonthLocator(interval=3))
    plt.plot(x,y,label=label)
    plt.gcf().autofmt_xdate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', type=str, default='./data/timeseries_1h.csv',
                        help='Path to csv dataset to use')
    parser.add_argument('-w','--window-size', type=int, default=50,
                        help='Length of input sequence to predict next datapoint')                        
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    # Preprocess input and reshapes to 
    # (num_samples, window_size, 1)
    processor = DataProcessor(args.window_size)
    train_X, train_y, test_X, test_y = processor.preprocess(df)

    lstm = LSTMModel(args.window_size)
    print(lstm.model.summary())

    lstm.fit(train_X, train_y)

    preds = lstm.predict(test_X)

    preds = processor.postprocess(preds)
    actuals = processor.postprocess(test_y)

    mean_squared_error(preds, actuals)

    plt.plot(actuals, label='truth')
    plt.plot(preds, label='prediction')
    plt.legend()

    figpath='./plot.png'
    plt.savefig(figpath)

if __name__ == "__main__":
    main()