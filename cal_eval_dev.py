import os
import pickle
import math
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import optuna
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import BinaryCalibrationError 
from sklearn.isotonic import IsotonicRegression

from GoalNet import GoalNet
from utils import load_data_tensors, load_datasets, get_hidden_sizes_from_optuna, get_ba_from_conf
from artifact_download import load_wandb_model
from td import get_episode_data


def train_iso_reg(model, X, Y):
    """
    Trains isotonic regression on dataste (X, Y) to calibrate model.

    Args:
        model (torch nn model): Model that is being calibrated.
        X (torch.Tensor): Calibration traning data inputs
        Y (torch.Tensor): Calibration training data labels

    Returns:
        iso_reg (sklearn iso reg model): trained isotonic regression model
    """
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds = 'clip')
    pred = model(X)
    iso_reg = iso_reg.fit(pred.squeeze().numpy(force=True), Y.squeeze().numpy(force=True))
    return iso_reg

def eval_uncal(model, X, Y):
    """
    Evaluates uncalibrated model on dataset (X, Y).

    Args:
        model (torch.nn model): Model to be evaluated.
        X (torch.Tensor): Features of evaluation dataset.
        Y (torch.Tensor): Labels of evaluation dataset.
    
    Returns:
        Tuple containing:
            -BA (float): Balanced Accuracy
            -bce (float): Binary Calibration Error (same as ECE: Expected Calibration Error)

    """
    bce_metric = BinaryCalibrationError(n_bins=10)
    bce_metric.reset()
    outputs = model(X)
    bce_metric.update(outputs, Y)
    bce = bce_metric.compute()
    print("BCE: ", bce)

    mx = confusion_matrix(Y.squeeze().clone(), outputs.round().to(dtype=torch.int64).clone(), labels=[0, 1])
    TN, FP, FN, TP = mx.ravel()
    specificity = TN / (TN + FP)
    recall = TP / (TP + FN)
    BA = (recall + specificity) / 2

    return BA, bce

def horizon_val_data(data_list, horizon, max_horizon):
    """
    Args:
        data (List of episodes)
    """

    new_ep_list = []
    for ep in data_list:
        tile_length = ep.size(dim=0) - 1


        #add k feature
        h_feat = torch.tensor(horizon / max_horizon).tile((tile_length, 1)) #horizon feature
        new_feature_data = torch.cat((ep[:-1, :-1], h_feat, ep[:-1, -1:]), dim=1)
        #label the data according to horizon k
        new_feature_data[-horizon:, -1] = ep[-1][-1]
        
        new_ep_list.append(new_feature_data)

        #now do k=1
        # last_h_feat = torch.tensor(1 / max_k).tile((tile_length, 1))
        # last_k_sample = torch.cat((ep[:-1, :-1], last_h_feat, ep[1:, -1:]), dim=1)
        # new_ep_list.append(last_k_sample)
        
    dataTensor = torch.cat(new_ep_list, dim=0)
    return dataTensor

def get_td_ba_bce(model, X, Y):
    """
    Evaluates model on dataset (X, Y). Model takes horizon/max_horizon as last input feature.

    Args:
        model (torch.nn model): Model to be evaluated.
        X (torch.Tensor): Features of evaluation dataset.
        Y (torch.Tensor): Labels of evaluation dataset.
        horizon (int): Horizon to do evaluation with
        max_horizon (int): Normalization constant
    
    Returns:
        Tuple containing:
            -BA (float): Balanced Accuracy
            -bce (float): Binary Calibration Error (same as ECE: Expected Calibration Error)

    """
    bce_metric = BinaryCalibrationError(n_bins=10)
    bce_metric.reset()
    outputs = model(X)
    bce_metric.update(outputs, Y)
    bce = bce_metric.compute()

    mx = confusion_matrix(Y.squeeze().clone(), outputs.round().to(dtype=torch.int64).clone(), labels=[0, 1])
    BA = get_ba_from_conf(*mx.ravel())
    return BA, bce

def get_td_ba_bce_calibrated(model, iso_reg, X, Y):
    """
    Evaluates model on dataset (X, Y). Model takes horizon/max_horizon as last input feature.

    Args:
        model (torch.nn model): Model to be evaluated.
        X (torch.Tensor): Features of evaluation dataset.
        Y (torch.Tensor): Labels of evaluation dataset.
        horizon (int): Horizon to do evaluation with
        max_horizon (int): Normalization constant
    
    Returns:
        Tuple containing:
            -BA (float): Balanced Accuracy
            -bce (float): Binary Calibration Error (same as ECE: Expected Calibration Error)

    """
    bce_metric = BinaryCalibrationError(n_bins=10)
    bce_metric.reset()
    uncal_pred = model(X)
    cal_pred = iso_reg.transform(uncal_pred.squeeze().numpy(force=True))
    bce_metric.update(torch.from_numpy(cal_pred), Y.squeeze())
    bce = bce_metric.compute()

    mx = confusion_matrix(Y.squeeze().numpy(), cal_pred.round().astype(np.int64), labels=[0, 1])
    BA = get_ba_from_conf(*mx.ravel())
    return BA, bce

def eval_cal(model, iso_reg, X, Y):
    """
    Evaluates calibrated model on datast (X, Y).

    Args:
        model (torch.nn model): Uncalibrated model.
        iso_reg (sklearn iso reg model): Trained iso reg model used to calibrate model outputs.
        X (torch.Tensor): Features of dataset.
        Y (torch.Tensor): Labels of dataset.

    Returns:
        Tuple containing:
            -BA (float): Balanced Accuracy of calibrated model.
            -bce (float): Expected Calibration Error of calibrated model.
    """
    uncal_pred = model(X)
    cal_pred = iso_reg.transform(uncal_pred.squeeze().numpy(force=True))
    
    bce_metric = BinaryCalibrationError(n_bins=10)
    bce_metric.reset()
    bce_metric.update(torch.from_numpy(cal_pred), Y_test.squeeze())
    bce = bce_metric.compute()

    target = Y.squeeze().numpy()
    value = cal_pred.round().astype(np.int64, casting='unsafe') 
    mx = confusion_matrix(target, value, labels=[0, 1])
    TN, FP, FN, TP = mx.ravel()
    specificity = TN / (TN + FP)
    recall = TP / (TP + FN)
    BA = (recall + specificity) / 2
    return BA, bce

def load_iso_reg(path):
    """
    Loads trained sklearn iso reg model from path.
    """
    with open(path, "rb") as f:
        iso_reg = pickle.load(f)
    return iso_reg

def save_iso_reg(iso_reg, path):
    """
    Saves trained sklearn iso reg model at path.
    """
    with open(path, "wb") as f:
        pickle.dump(iso_reg, f, protocol=5)

def cal_eval():
    """
    Old main function.
    """
    #path = "./data/n_eps-500-env-base_agent_env_2025-06-05_20-56-48.pkl"
    path = "./data/n_eps-2000-env-base_agent_env_2025-06-15_00-40-56.pkl"
    horizon = 50 #iteratons
    job = 2
    train_val_test_split = [0.6, 0.2, 0.2]
    assert sum(train_val_test_split) == 1

    #prepare the data
    tensorList = load_data_tensors(path, train_val_test_split, horizon)
    X_val = tensorList[1][:, :-1]
    Y_val = tensorList[1][:, -1:]
    X_test = tensorList[2][:, :-1]
    Y_test = tensorList[2][:, -1:]
    Y_val = Y_val.round().to(torch.int64)
    Y_test = Y_test.round().to(torch.int64)
    valDataset = TensorDataset(tensorList[1][:, :-1], torch.round(tensorList[1][:, -1:]).to(dtype=torch.int64))

    #storage_path = "./models/"
    #cluster = 4346754
    cluster = 4357100
    storage_path = f"./transfer/transfer_models_{cluster}_{job}_{horizon}/models/"
    db_dir = f"./transfer/transfer_databases_{cluster}_{job}_{horizon}/databases/"
    db_path = f"sqlite:///" + db_dir + "optuna_goalNet.db"
    #study_name = "test"
    study_name = "horizon_{}_goalNet_BABCE_BA".format(horizon)
    iso_reg_save_path = f"./iso_reg_{cluster}_{horizon}.pkl"

    study = optuna.load_study(study_name=study_name, storage=db_path)

    print("Best trial: ", study.best_trial.number)
    params = study.best_trial.params
    print("value: ", study.best_trial.values[0])

    input_size = valDataset[0][0].size(dim=0)
    print("input_size: ", input_size)
    hidden_sizes = get_hidden_sizes_from_optuna(db_path, study_name, study.best_trial)

    model = GoalNet(input_size, hidden_sizes)
    model_path = f"{storage_path}{study_name}/{study_name}_trial_{study.best_trial.number}.pth"
    print("model_path: ", model_path)
    assert os.path.exists(model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()

    with torch.no_grad():
        print("Uncalibrated model on validation set:")
        BA, bce = eval_uncal(model, X_val, Y_val)
        print("BA: ", BA)
        print("BCE: ", bce)
        print()

        print("Uncalibrated model on test set:")
        BA, bce = eval_uncal(model, X_test, Y_test)
        print("BA: ", BA)
        print("BCE: ", bce)
        print()
        
        #Calibrate on validation set
        print("calibrate on validation set")
        iso_reg = train_iso_reg(model, X_val, Y_val)
        
        #Evaluate on test set
        BA, bce = eval_cal(model, iso_reg, X_test, Y_test)
        print("Calibrated model on test set: ")
        print("BA: ", BA)
        print("BCE: ", bce)
        print()

def td_eval(horizon, device):
    path = "./data/n_eps-500-env-base_agent_env_2025-06-05_20-56-48.pkl"
    #path = "./data/n_eps-2000-env-base_agent_env_2025-06-15_00-40-56.pkl"
    train_val_test_split = [0.6, 0.2, 0.2]
    assert sum(train_val_test_split) == 1

    _, valList, testList = get_episode_data(path, train_val_test_split)
    valTensor = horizon_val_data(valList, horizon, 100)
    testTensor = horizon_val_data(testList, horizon, 100)
    X_val = valTensor[:, :-1]
    Y_val = valTensor[:, -1:]
    X_test = testTensor[:, :-1]
    Y_test = testTensor[:, -1:]
    Y_val = Y_val.round().to(torch.int64)
    Y_test = Y_test.round().to(torch.int64)


    #storage_path = "./models/"
    #cluster = 4346754
    
    #get model
    model_path = "./artifact_download/tdgoal_h100_ep500_epoch200_l5_trial_2025-08-27_16-10-28_pv5kiybp_best.pth"
    run_path = "dpinchuk-university-of-wisconsin-madison/tdgoal_h100_ep500_epoch200/pv5kiybp"
    assert os.path.exists(model_path)
    model = load_wandb_model(run_path, model_path, device)




    model.eval()

    with torch.no_grad():
        print("Uncalibrated model on validation set:")
        BA, bce = get_td_ba_bce(model, X_val, Y_val)
        print("BA: ", BA)
        print("BCE: ", bce)
        print()
     
        #Calibrate on validation set
        print("calibrate on validation set")
        iso_reg = train_iso_reg(model, X_val, Y_val)

        save_iso_reg(iso_reg, f"./best_td_iso_reg_h{horizon}.pth")

        print("Uncalibrated model on test set:")
        BA, bce = get_td_ba_bce(model, X_test, Y_test)
        print("BA: ", BA)
        print("BCE: ", bce)
        print()
        
        #Evaluate on test set
        BA, bce = get_td_ba_bce_calibrated(model, iso_reg, X_test, Y_test)
        print("Calibrated model on test set: ")
        print("BA: ", BA)
        print("BCE: ", bce)
        print()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    td_eval(10, device) 
