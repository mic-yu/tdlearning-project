import sys
import math
import torch
import pickle
import numpy as np
import optuna
from torch.utils.data import TensorDataset

np.set_printoptions(threshold=sys.maxsize)



def load_data_tensors(path, train_val_test_split, horizon=None):
    """
    Labels data according to the horizon and returns the train, val and test datasets.

    Args:
        path (string): Path to pickled data formatted as a list of lists (list of episodes).
        train_val_test_split (list): Proportion of dataset to split into train/val/test datasets. E.g. [0.6, 0.2, 0.2]
        horizon (int): Used to label probability of scoring in the next horizon states from a given state.

    Returns: 
        list: List of 3 torch tensors corresponding to train, val, test datasets
    """
    print("Entered load data tensors")
    assert sum(train_val_test_split) == 1

    with open(path, 'rb') as f:
        obsList = pickle.load(f)


    #last_obs = []
    #turn each episode into a single numpy array
    for i in range(len(obsList)):
        obsList[i] = np.vstack(obsList[i]) 
        #last_obs.append(obsList[i][-1, :])
        #print(obsList[i])

        #print("obsList[i].shape: ", obsList[i].shape)
        # bad_num_original = [x for x in obsList[i][:, -1:] if x < -0.5 or x > 1.5]
        # if len(bad_num_original) > 0:
        #     print(f"Original {i} bad_num_original: ", bad_num_original)



    #for each episode label the data according to horizon
    if horizon is not None:
        for i in range(len(obsList)):
            if obsList[i][-1][-1] == 1:
                obsList[i][-(horizon + 1):, -1] = 1

    #do train/val/test split by episodes
    num_eps = len(obsList)
    num_val = math.floor(train_val_test_split[1] * num_eps)
    num_test = math.ceil(train_val_test_split[2] * num_eps)
    num_train = num_eps - num_test - num_val


    train_list = obsList[:num_train]
    val_list = obsList[num_train: num_train + num_val]
    test_list = obsList[num_train + num_val:]

    #print("Utils original numpy dtype: ", obsList[0].dtype)
    #concatenate each train, val and test list of np arrays into
    #each being a single tensor
    #convert train, val and test tensor into float32
    tensorList = []
    for npList in [train_list, val_list, test_list]:
        obsTensorList = [torch.tensor(x) for x in npList] #convert each numpy array into torch
        # for i in range(len(npList)):
        #     appendTensor = torch.tensor(npList[i]) #size[1] = 9
        #     obsTensorList.append(appendTensor)
        #     #obsTensor = torch.cat((obsTensor, appendTensor), 0)
        obsTensor = torch.cat(obsTensorList)
        #print("obsTensor.shape: ", obsTensor.size())
        # quit()
        # print("obsTensor dtype before conversion in utils: ", obsTensor.dtype)
        # bad_num = [x for x in obsTensor[:, -1:].flatten() if x < -1 or x > 2]
        # print("bad num in utils before conversion: ", bad_num)

        # print("Conversion in utils: ")
        obsTensor = obsTensor.to(torch.float32)
        # bad_num = [x for x in obsTensor[:, -1:].flatten() if x < -1 or x > 2]
        # print("bad numb after converstion in utils: ", bad_num)
        tensorList.append(obsTensor)

    return tensorList 
def load_datasets(path, train_val_test_split, horizon, device='cpu'):
    """
    Loads data from path, labels it according to horizon and stores it according to the 
    split, as the training, val and test datasets as torch datasets.

    Args:
        path (string): Path to pickled data formatted as a list of lists (list of episodes).
        train_val_test_split (list): Proportion of dataset to split into train/val/test datasets. E.g. [0.6, 0.2, 0.2]
        horizon (int): Used to label probability of scoring in the next horizon states from a given state.
        device (string): What device to store datasets on.

    Returns:
        tuple containing

        -torch dataset: train dataset
        -torch dataset: val dataset
        -torch dataset: test dataset

    """
    tensorList = load_data_tensors(path, train_val_test_split, horizon)
    tensorList = [t.to(device=device) for t in tensorList]

    trainDataset = TensorDataset(tensorList[0][:, :-1], tensorList[0][:, -1:])
    valDataset = TensorDataset(tensorList[1][:, :-1], tensorList[1][:, -1:])
    testDataset = TensorDataset(tensorList[2][:, :-1], tensorList[2][:, -1:])

    return trainDataset, valDataset, testDataset

def load_abs_datasets(path, train_val_test_split, horizon, device='cpu', preprocess = True):
    tensorList = load_data_tensors(path, train_val_test_split, horizon)
    tensorList = [t.to(device=device) for t in tensorList]
    if preprocess:
        tensorList = [preprocess_abs_dataset(t) for t in tensorList]

    trainDataset = TensorDataset(tensorList[0][:, :-1], tensorList[0][:, -1:])
    valDataset = TensorDataset(tensorList[1][:, :-1], tensorList[1][:, -1:])
    testDataset = TensorDataset(tensorList[2][:, :-1], tensorList[2][:, -1:])

    return trainDataset, valDataset, testDataset


def preprocess_abs_dataset(data):
    """
    Input tensor should have observations [agent.x, agent.y, ball.x, ball.y, agent.face_angle, label]
    """
    field_half_width = 4500
    data[:, :4] = data[:, :4] / field_half_width #normalize
    rad_angle = torch.deg2rad(data[:, 4])
    cos_face_angle = torch.unsqueeze(torch.cos(rad_angle), 1)
    sin_face_angle = torch.unsqueeze(torch.sin(rad_angle), 1)
    data = torch.cat((data[:, :4], cos_face_angle, sin_face_angle, data[:, 5:]), dim=1)
    return data

def get_pos_weight(dataset):
    """
    Returns the ratio of negative to positive samples in dataset.

    Args:
        dataset (torch dataset)
    
    Returns:
        float: the weight to give positive samples in order to match negative samples
    """
    num_positive = 0
    for i in range(len(dataset)):
        _, label = dataset[i]
        num_positive += label.item()


    num_negative = len(dataset) - num_positive
    print("train pos: ", num_positive)
    print("train neg: ", num_negative)
    pos_weight = num_negative / num_positive
    pos_weight = torch.tensor(pos_weight)
    return pos_weight


def get_accuracy(path):
    """
    Returns the proportion of total episodes in dataset in which agent scored.

    Args:
        path (string): Path to pickled data formatted as a list of lists (list of episodes).

    Returns:
        float: the "accuracy"
        
    """
    with open(path, 'rb') as f:
        obsList = pickle.load(f)

    #turn each episode into a single numpy array
    for i in range(len(obsList)):
        obsList[i] = np.vstack(obsList[i]) 

    #for each episode label the data according to horizon
    accuracy = 0
    for i in range(len(obsList)):
        if obsList[i][-1][-1] == 1:
            accuracy += 1
    accuracy = accuracy / len(obsList)
    return accuracy

def get_hidden_sizes_from_optuna(db_path, study_name, trial):
    """
    Returns the number of neurons in each hidden layer of a model from an optuna database

    Args:
        db_path (string): Path to optuna db
        study_name (string): Name of study in optuna db
        trial (Optuna trial): Optuna trial corresponding to model.
    
    Returns:
        List[int]: List of integers corrsponding to sizes of hidden layers. 
    """
    
    study = optuna.load_study(study_name=study_name, storage=db_path)
    params = trial.params
    hidden_sizes = []
    for i in range(params["n_layers"]):
        hidden_sizes.append(params[f"neurons_layer_{i}"])
    return hidden_sizes

def get_ba_from_conf(TN, FP, FN, TP):
    """
    Compute Balanced Accuracy (float) from TN (int), FP (int), FN (int), TP (int).
    """
    specificity = TN / (TN + FP)
    recall = TP / (TP + FN)
    BA = (recall + specificity) / 2 
    return BA

def get_relative_observation(agent_loc, object_loc):
    """
    Gets relative position of object to agent
    """
    
    x = object_loc[0] - agent_loc[0]
    y = object_loc[1] - agent_loc[1]
    agent_face_rad = np.radians(agent_loc[2])
    #print("agent_face_rad: ", agent_face_rad)
    
    # make sure angle is between -pi and pi which fits SimRobot notation
    if agent_face_rad > np.pi:
        agent_face_rad -= 2*np.pi
        
    angle = np.arctan2(y, x) - agent_face_rad
    #print("agent_face_rad: ", agent_face_rad)
    #print("angle: ", angle)
    

    # Rotate x, y by -agent angle
    xprime = x * np.cos(agent_face_rad) - y * np.sin(agent_face_rad)
    yprime = x * np.sin(agent_face_rad) + y * np.cos(agent_face_rad)
    # print(f"xprime: {xprime}, yprime: {yprime}, angle: {angle}")
    #denorm = max(FIELD_LENGTH, FIELD_WIDTH) + BORDER_STRIP_WIDTH
    denorm = 10000 # overwrite to agree with SimRobot
    return [xprime / denorm, yprime / denorm, np.sin(angle), np.cos(angle)]

if __name__ == "__main__":
    #print(help(load_data_tensors))
    print(help(load_datasets))

    # path = "./data/n_eps-500-env-base_agent_env_2025-06-15_14-57-21.pkl"
    # accuracy = get_accuracy(path)
    # print("accuracy: ", accuracy)

    #non-unique -angle rotation observation
    #goal_loc = [4500, 0]

    # agent_loc_1 = [3500, 100, 270]
    # ball_loc_1 = [3500,0]
    
    # agent_loc_2 = [4400, -1000, 0]
    # ball_loc_2 = [4500, -1000]

    # rel_obs_1 = get_relative_observation(agent_loc_1, ball_loc_1)
    # rel_obs_2 = get_relative_observation(agent_loc_2, ball_loc_2)
    # print("rel_obs_1: ", rel_obs_1)
    # print("rel_obs_2: ", rel_obs_2)
    # assert rel_obs_1 == rel_obs_2


    #plus angle rotation non-uniqueness
    # agent_loc_1 = [4500, 1000, 270]
    # ball_loc_1 = [4500, 500]

    # agent_loc_2 = [4500, -1000, 90]
    # ball_loc_2 = [4500, -500]

    # obs_1 = get_relative_observation(agent_loc_1, goal_loc)
    # obs_2 = get_relative_observation(agent_loc_2, goal_loc)
    # print("goal:")
    # print("obs_1: ", obs_1)
    # print("obs_2: ", obs_2)

    # print("ball:")
    # obs_1 = get_relative_observation(agent_loc_1, ball_loc_1)
    # obs_2 = get_relative_observation(agent_loc_2, ball_loc_2)
    # print("obs_1: ", obs_1)
    # print("obs_2: ", obs_2)
    






    

