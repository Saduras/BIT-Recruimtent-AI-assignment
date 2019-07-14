import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.metrics import mean_squared_error

from model import LSTMModel
from data_processor import DataProcessor

def plot_time_data(x, y, label=''):
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(dates.MonthLocator(interval=3))
    plt.plot(x,y,label=label)
    plt.gcf().autofmt_xdate()

def plot_test_datapoint(test_x, test_y, pred, forcast):
    plt.plot(test_x, label='input')
    off = len(test_x)
    xs = list(range(off, off + forcast))
    plt.plot(xs, pred, label='prediction')
    plt.plot(xs, test_y, label='label')

    plt.legend()
    plt.savefig('./plot_single.png')

def plot_moving_window(timestamps, dataset, preds_moving):
    timestamps = dates.datestr2num(timestamps)
    fig2 = plt.figure()
    plot_time_data(timestamps[:len(dataset)], dataset, label='full dataset')

    x = float(timestamps[len(dataset)])
    step_size = float(timestamps[-1] - timestamps[-2])
    end = float(x + len(preds_moving) * step_size)
    xs = []
    while x < end:
        xs.append(x)
        x += step_size

    plot_time_data(xs, preds_moving, label='prediction')
    plt.legend()
    plt.savefig('./plot_moving.png')

def moving_test_window_preds(model, start_X, n_future_preds, step):
    preds_moving = []
    moving_test_window = start_X.reshape(1, start_X.shape[0], start_X.shape[1])

    for i in range(0,n_future_preds, step):
        preds = model.predict(moving_test_window)

        preds_moving.extend(preds[0])
        preds = preds.reshape(1,-1,1)

        moving_test_window = np.concatenate((moving_test_window[:,preds.shape[1]:,:], \
                                            preds), axis=1)
    return preds_moving

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', type=str, default='./data/timeseries_1h.csv',
                        help='Path to csv dataset to use')
    parser.add_argument('-m','--model-path', type=str, default='./model.h5',
                        help='Path for saving/loading trained model')
    parser.add_argument('-w','--window-size', type=int, default=50,
                        help='Length of input sequence to predict next datapoint')
    parser.add_argument('-f','--forcast', type=int, default=5,
                        help='Length of predicted sequence')
    parser.add_argument('--shift', type=int, default=1,
                        help='By how many steps a training/test sequence is shifted to its previous')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--eval-only', action='store_true',
                        help='If set model will be loaded from path instead of trained')
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df.iloc[::24,:]

    # Preprocess input and reshapes to 
    # (num_samples, window_size, 1)
    processor = DataProcessor(window_size=args.window_size, 
                            forcast_size=args.forcast,
                            shift=args.shift)
    train_X, train_y, test_X, test_y, raw_series = processor.preprocess(df)

    lstm = LSTMModel(args.window_size, args.forcast)
    print(lstm.model.summary())
    if not args.eval_only:
        lstm.fit(train_X, train_y, epochs=args.epochs)
        lstm.save(args.model_path)
    else:
        lstm.load(args.model_path)

    # evaluation and plots
    preds = lstm.predict(test_X[-1].reshape(1,-1, 1))
    preds = processor.postprocess(preds)
    plot_test_datapoint(test_X[-1], test_y[-1], preds[0], args.forcast)

    preds_moving = moving_test_window_preds(lstm, test_X[0,:], 
                                            n_future_preds=500,
                                            step=args.forcast)
    preds_moving = np.array(preds_moving).reshape(-1,1)
    preds_moving = processor.postprocess(preds_moving)

    plot_moving_window(df['datetime'], raw_series, preds_moving)

if __name__ == "__main__":
    main()