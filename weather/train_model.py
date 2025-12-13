import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import pandas as pd

class TDNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    """
    argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/weather_data.csv')
    parser.add_argument('--previous-steps', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--model', type=str, default='nn', help='model type')
    parser.add_argument('--save-path', type=str, default='/models/', help='path to save the trained model')
    parser.add_argument('--lam', type=float, default=0.8, help='lambda for TD-lambda')
    parser.add_argument('--data-path', type=str, default='data/reanalysis-era5-land-timeseries-sfc-2m-temperaturemnrm5mgr.csv', help='path to training data')
    parser.add_argument('--data-split', type=float, default=0.8, help='train-test split ratio')
    return parser.parse_args()

def prep_data(args):
    """
    to prepare training data
    """
    data = pd.read_csv(args.data_path)
    
    train = data[0:int(len(data)*args.data_split)][['t2m']]
    train_data = torch.tensor(train, dtype=torch.float32)

    return train_data

def train_nn(model, optimizer, train_data, args):
    """
    to train a basic neural network
    """
    for t in range(len(train_data) - args.previous_steps - 1):
        state = train_data[t:t+args.previous_steps]
        reward = train_data[t+args.previous_steps]

        pred = model(state)

        loss = (reward - pred).pow(2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def train_td0(model, optimizer, train_data, args):
    """
    to train a TD(0) model
    """
    for t in range(len(train_data) - args.previous_steps - 1):
        state = train_data[t:t+args.previous_steps]
        next_state = train_data[t+1:t+args.previous_steps+1]

        reward = train_data[t+args.previous_steps]

        pred = model(state)                 # V(S_t)
        with torch.no_grad():
            next_value = model(next_state)  # V(S_{t+1})

        td_target = reward + args.gamma * next_value
        td_error = td_target - pred

        loss = td_error.pow(2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def train_tdlambda(model, train_data, args): 
    """
    to train a td-lambda model
    """
    eligibility = [torch.zeros_like(p) for p in model.parameters()]
    
    for t in range(len(train_data) - args.previous_steps - 1):
        state = train_data[t:t+k]
        next_state = train_data[t+1:t+k+1]
        reward = train_data[t+k]

        value = model(state)
        with torch.no_grad():
            next_value = model(next_state)

        td_error = reward + args.gamma * next_value - value
        td_error = torch.clamp(td_error, -1.0, 1.0)

        model.zero_grad()
        value.backward()

        with torch.no_grad():
            for p, e in zip(model.parameters(), eligibility):
                e.mul_(args.gamma * args.lam)
                e.copy_(p.grad)              
                e.clamp_(-5.0, 5.0)

                p.add_(args.alpha * td_error * e)
    
def main():
    args = parse_args()
    
    model = TDNet(args.previous_steps)
    optimizer = optim.Adam(model.parameters(), lr=args.alpha)

    train_data = prep_data(args)
    
    if args.model == 'nn':
        train_nn(model, optimizer, train_data, args)
        model.save(args.save_path, args.model, '_', args.previous_steps, '_', args.alpha, '_', args.gamma, '_model.pt')
    if args.model == 'td0':
        train_td0(model, optimizer, train_data, args)
        model.save(args.save_path, args.model, '_', args.previous_steps, '_', args.alpha, '_', args.gamma, '_model.pt')
    if args.model == 'tdlambda':
        train_tdlambda(model, optimizer, train_data, args)
        model.save(args.save_path, args.model, '_', args.previous_steps, '_', args.alpha, '_', args.gamma, '_', args.lam '_model.pt')
    
    
        
if __name__ == "__main__":
    main()