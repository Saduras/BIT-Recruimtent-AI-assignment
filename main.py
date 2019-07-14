import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates

def load_data(path):
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        # remove header line and convert to list
        data = list(reader)[1:]

    assert len(data) > 0, 'dataset is empty'
    assert len(data[0]) == 2, 'unexpected number of columns'
    print(f'{len(data)} datapoints loaded from {path}')
    return np.array(data)

def plot_data(x, y, figpath='./plot.png'):
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(dates.MonthLocator(interval=3))
    plt.plot(x,y)
    plt.gcf().autofmt_xdate()
    plt.savefig(figpath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', type=str, default='./data/timeseries_1h.csv',
                        help='Path to csv dataset to use')
    args = parser.parse_args()

    data = load_data(args.dataset)

    x = dates.datestr2num(data[:,0])
    y = data[:,1].astype(np.float)

    plot_data(x,y)

if __name__ == "__main__":
    main()