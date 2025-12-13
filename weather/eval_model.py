import torch
import torch.nn.functional as F
import argparse

def parse_args():
    """
    argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/weather_data.csv')
    parser.add_argument('--previous-steps', type=int, default=5)
    parser.add_argument('--model-path', type=str, default='/models/', help='path to trained model')
    parser.add_argument('--data-path', type=str, default='data/reanalysis-era5-land-timeseries-sfc-2m-temperaturemnrm5mgr.csv', help='path to training data')
    parser.add_argument('--data-split', type=float, default=0.8, help='train-test split ratio')
    return parser.parse_args()

def prep_data(args):
    """
    to prepare training data
    """
    data = pd.read_csv(args.data_path)
    
    test = data[int(len(data)*args.data_split):][['t2m']]
    test_data = torch.tensor(test, dtype=torch.float32)
    return test_data

def make_windows(series, k):
    X, y = [], []
    for i in range(len(series) - k):
        X.append(series[i:i+k])
        y.append(series[i+k])
    return torch.stack(X), torch.stack(y)

def main():
    args = parse_args()

    test_data = prep_data(args)

    X_test, y_test = make_windows(test_data, args.previous_steps)

    model = torch.load(args.model_path)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
    
    mse = F.mse_loss(y_pred, y_test)
    print(f"Test MSE: {mse.item()}")

if __name__ == "__main__":
    main()
