import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import pandas as pd

from fbprophet import Prophet

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

    df = pd.read_csv(args.dataset)
    df.columns = ['ds', 'y']

    print(df.head())

    model = Prophet(changepoint_prior_scale=0.01)
    model.fit(df)

    future = model.make_future_dataframe(periods=180 * 24, freq='H')
    forecast = model.predict(future)

    fig1 = model.plot(forecast)

    figpath='./plot.png'
    plt.savefig(figpath)

if __name__ == "__main__":
    main()