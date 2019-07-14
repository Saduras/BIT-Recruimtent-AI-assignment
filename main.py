import argparse
import csv

def load_data(path):
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        # remove header line and convert to list
        data = list(reader)[1:]

    assert len(data) > 0, 'dataset is empty'
    assert len(data[0]) == 2, 'unexpected number of columns'
    print(f'{len(data)} datapoints loaded from {path}')
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', type=str, default='./data/timeseries_1h.csv',
                        help='Path to csv dataset to use')
    args = parser.parse_args()

    data = load_data(args.dataset)

if __name__ == "__main__":
    main()