
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates

def plot_time_data(x, y, label=''):
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(dates.MonthLocator(interval=3))
    plt.plot(x,y,label=label)
    plt.gcf().autofmt_xdate()

def plot_test_datapoint(test_x, test_y, pred, forecast):
    plt.plot(test_x, label='input')
    off = len(test_x)
    xs = list(range(off, off + forecast))
    plt.plot(xs, test_y, label='label')
    plt.plot(xs, pred, label='prediction')

    plt.legend()
    plt.savefig('./plot_single.png')

def plot_moving_window(timestamps, dataset, preds_moving):
    timestamps = dates.datestr2num(timestamps)
    fig2 = plt.figure()

    # revert diff calculation
    dataset = np.array(dataset).cumsum()
    preds_moving = dataset[-1] + preds_moving.cumsum()

    plot_time_data(timestamps[:len(dataset)], dataset, label='full dataset')

    # x-axis for future predicution
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