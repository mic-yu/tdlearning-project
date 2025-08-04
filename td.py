#from train import train
from utils import load_data_tensors
from torch.utils.data import TensorDataset, random_split
from GoalNet import GoalNet
import numpy as np
import pickle
import torch
import math

def get_episode_data(path, train_val_test_split):
    with open(path, 'rb') as f:
        obsList = pickle.load(f)

    #turn each episode into a single numpy array
    for i in range(len(obsList)):
        obsList[i] = torch.tensor(np.vstack(obsList[i]), dtype=torch.float32)


    #do train/val/test split by episodes
    num_eps = len(obsList)
    num_val = math.floor(train_val_test_split[1] * num_eps)
    num_test = math.ceil(train_val_test_split[2] * num_eps)
    num_train = num_eps - num_test - num_val

    train_list = obsList[:num_train]
    val_list = obsList[num_train: num_train + num_val]
    test_list = obsList[num_train + num_val:]

    return train_list, val_list, test_list


def get_nn_data(model, data_list, max_k):
    """
    Args:
        data (List of episodes)
    """
    new_ep_list = []
    for ep in data_list:
        tile_length = ep.size(dim=0) - 1
        for k in range(2, max_k+1):           
            h_feat = torch.tensor(k / max_k).tile((tile_length, 1)) #horizon feature
            h_feat_next = torch.tensor((k-1) / max_k).tile((tile_length, 1)) #next horizon feature
            print("h_feat.size: ", h_feat.size())
            print("ep.size: ", ep.size())
            boot_v = model(torch.cat((ep[1:, :-1], h_feat_next), dim=1))
            new_feature_data = torch.cat((ep[:-1, :-1], h_feat), dim=1)
            new_data = torch.cat((new_feature_data, boot_v), dim=1)
            
            new_terminal_sample = torch.cat((ep[-1, :-1], torch.tensor([k / max_k]), ep[-1, -1:]))
            print("new_sample size: ", new_terminal_sample.size())
            new_data = torch.cat((new_data, new_terminal_sample), dim=0)
            print("new data size: ", new_data.size())
            new_ep_list.append(new_data)

        #now do k=1
        last_h_feat = torch.tensor(1 / max_k).tile((tile_length, 1))
        last_k_sample = torch.cat(ep[:-1, :-1], last_h_feat, ep[1:, -1:])
        new_ep_list.append(last_k_sample)
        
        

    dataTensor = torch.cat(new_ep_list, dim=0)
    return dataTensor
    

def train_td(model, data, max_k):
    dataTensor = get_nn_data(model, data, max_k)
    nn_train_val_split = [0.8, 0.2]
    assert sum(nn_train_val_split) == 1
    nn_dataset = TensorDataset(dataTensor[:, :-1], dataTensor[:, -1:])
    generator = torch.Generator().manual_seed(37)
    trainSubset, valSubset = random_split(nn_dataset, nn_train_val_split, generator=generator)


    
if __name__ == '__main__':
    train_val_test_split = [0.6, 0.2, 0.2]
    path = "../roboClassifier/data/n_eps-500-env-base_agent_env_2025-06-05_20-56-48.pkl"
    max_k = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, valData, testData = load_data_tensors(path, train_val_test_split)
    trainList, _, _ = get_episode_data(path, train_val_test_split)

    input_size = valData.size(dim=1) #note data is features + labels, but we need an extra feature in input
    print("input sozie ; ", input_size)
    hidden_sizes = [10, 10]
    print(input_size)
    print(valData.dtype)
    print(trainList[0].dtype)
    model =  GoalNet(input_size, hidden_sizes).to(device=device)
    train_td(model, trainList, max_k)