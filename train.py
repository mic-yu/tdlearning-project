import os
import pickle
import math
import torch
import numpy as np
import argparse
import wandb

from torch.utils.data import TensorDataset, random_split, DataLoader
from torch import nn
from tqdm import tqdm
from datetime import datetime
#from torcheval.metrics.classification import BinaryRecall, BinaryAccuracy, BinaryConfusionMatrix
from matplotlib import pyplot as plt
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.metrics import confusion_matrix

from GoalNet import GoalNet
from utils import load_datasets, get_pos_weight, get_accuracy, get_ba_from_conf

# parser = argparse.ArgumentParser()
# parser.add_argument("--horizon", required=False, default=10, type=int)
# args = parser.parse_args()
# horizon = args.horizon
# print("parsed horizon: ", horizon)

# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# run_name = current_time + "_" + f"h{horizon}"
# wandb_kwargs = {
#     "entity": "", #"dpinchuk-university-of-wisconsin-madison",
#     "project": "", #"optuna_test",
#     "name": run_name
# }
# wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=False)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)




def train(trial, model, trainDataLoader, cfg, params):
    """
    Args:
        optuna trial
        model,
        loss function
    Return: trained model or model with lowest loss before early stoppage or rasies pruning exception
    Saves model with lowest val loss every 5 evaluation epochs or at final epoch if no early stoppage. 
    """

    lr = params["lr"]
    loss_fn = params["loss_fn"]
    num_epochs = round(cfg.target_update_frequency / len(trainDataLoader))
    if num_epochs == 0:
        num_epochs = 1
    assert num_epochs > 0
    print("Innter loop epochs: ", num_epochs)


    #Data
    model_storage_path = params["model_storage_path"]
    study_name = cfg.optuna.study_name
    model_path = model_storage_path + "/{}_trial_{}.pth".format(study_name, trial.number) #for saving

    print("model_path train: ", model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    min_loss = float('inf')
    patience = 4 #times we can have an evaluation that is worse than minimum
                 # corresponds to patience*5 epochs
    patience_counter = 0
    #eps = 1e-5 #by how much loss should improve over patience period
    #train_loss_list = []
    #val_loss_list = []
    sig = nn.Sigmoid()

    #Training loop
    for epoch in range(num_epochs): #)position=0, leave=False):
        running_avg = 0.0
        running_count = 0
        model.train()
        step = 0
        for X, Y in trainDataLoader:
            optimizer.zero_grad()
            outputs = model(X)
            if cfg.train_forward_sig:
                outputs = sig(outputs)
            loss = loss_fn(outputs, Y)
            loss.backward()
            optimizer.step()
            
            running_avg = running_avg*running_count + loss.item()*X.size(dim=0)
            running_count = running_count + X.size(dim=0)
            running_avg = running_avg/running_count

            step = step + 1
            if (step + 1) % 500 == 0:
                print(f"Step: {step}. Loss: {running_avg}")
            if step > 10000:
                wandb.log({"train_loss": running_avg})
                break
            
        #print("Epoch {} complete with avg train loss {}".format(epoch, running_avg))
        #train_loss_list.append(running_avg)

        #Evaluation loop and early stopping
        '''
        if (epoch + 1) % 5 == 0:
            val_loss, _ = eval(model, valDataLoader, loss_fn)
            trial.report(-val_loss, epoch)
            print("Epoch {} complete with avg val loss {}".format(epoch, val_loss))
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(model.state_dict(), model_path)
                patience_counter = 0
            if trial.should_prune():
                #trial.set_user_attr("pruned", True)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                _, conf_mx = eval(model, valDataLoader, loss_fn)
                BA = get_ba_from_conf(*conf_mx.ravel())
                trial.report(BA, epoch + 1)
                trial.set_user_attr("epochs_trained", epoch + 1)
                raise optuna.TrialPruned()
            else:
                patience_counter += 1
                if patience_counter > patience:
                    trial.set_user_attr("epochs_trained", epoch + 1)
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    return model 
        '''
    '''
    trial.set_user_attr("epochs_trained", epoch+1)
    '''

    #model_path = model_storage_path + "/{}_trial_{}".format(study_name, trial.number)
    torch.save(model.state_dict(), model_path)
    return model

# input_size = trainDataset[0][0].size(dim=0)
# myModel = Model(input_size, hidden_sizes)
# train(myModel)

def eval(model, valDataLoader, loss_fn):
    """
    Evaluate model on validation set. Return average loss, confusion matrix and recall.
    """
    model.eval()
    running_avg = 0.0
    running_count = 0

    conf_mx = np.zeros((2,2), dtype=np.int64)

    with torch.no_grad():
        for X, Y in valDataLoader:
            outputs = model(X)
            loss = loss_fn(outputs, Y)


            outputs = outputs.squeeze()
            outputs = (outputs > 0.5).float()

            #print("outputs[:5, :]", outputs[:5])
            #print("Y[:5, :]: ", Y[:5, :])

            #print("outputs.shape: ", outputs.shape)
            #print("Y.shape: ", Y.shape)

            running_avg = running_avg*running_count + loss.item()*X.size(dim=0)
            running_count = running_count + X.size(dim=0)
            running_avg = running_avg/running_count

            mx = confusion_matrix(Y.squeeze().int().numpy(force=True), outputs.int().numpy(force=True), labels=[0, 1])
            conf_mx = conf_mx + mx
            

            
    return running_avg, conf_mx

#@wandbc.track_in_wandb()
def objective(trial):
    #hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_sizes = []
    for i in range(n_layers):
        num_neurons = trial.suggest_int('neurons_layer_{}'.format(i), 8, 64)
        hidden_sizes.append(num_neurons)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('bs', 32, 256, step=32)

    #loss_fn = nn.MSELoss()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #Data
    trainDataLoader = DataLoader(trainDataset, batch_size = batch_size, shuffle=True) 
    valDataLoader = DataLoader(valDataset, batch_size = batch_size, shuffle=True) 

    #training
    input_size = trainDataset[0][0].size(dim=0)
    myModel = GoalNet(input_size, hidden_sizes).to(device=device)
    trainedModel = train(trial, myModel, trainDataLoader, valDataLoader, loss_fn, lr)

    #save model
    #Should be saved in train function
    #model_path = storage_path + "{}_trial_{}".format(study_name, trial.number)
    #torch.save(trainedModel, model_path)

    #validation
    _, conf_mx = eval(trainedModel, valDataLoader, loss_fn)
    TN, FP, FN, TP = conf_mx.ravel()

    #print("type(TN): ", type(TN))
    #print("type(TN.item()): ", type(TN.item()))

    specificity = TN / (TN + FP)
    recall = TP / (TP + FN)
    BA = (recall + specificity) / 2 #balanced accuracy
    #print("type(BA): ", type(BA))

    trial.set_user_attr("TP", TP.item())
    trial.set_user_attr("FN", FN.item())
    trial.set_user_attr("TN", TN.item())
    trial.set_user_attr("FP", FP.item())

    return BA.item()

if __name__ == '__main__':


    path = "./data/n_eps-500-env-base_agent_env_2025-06-05_20-56-48.pkl"
    train_val_test_split = [0.6, 0.2, 0.2]
    num_epochs = 10

    trainDataset, valDataset, testDataset = load_datasets(path, train_val_test_split, horizon, device=device)

    accuracy = get_accuracy(path)
    print("accuracy from new function :", accuracy)

    pos_weight = get_pos_weight(trainDataset).to(device=device)
    print("pos_weight: ", pos_weight)


    #study_name = "bug_search_2"
    study_name = "test_new"
    #study_name = "goalNet"
    #study_name = "goalNet_BABCE_BA"
    study_name = "horizon_{}_".format(horizon) + study_name
    storage_path = "./models/"
    model_storage_path = storage_path + study_name

    number_of_trials = 1 #1 for testing only, I use 100 for training

    db_dir = "./databases/"
    os.makedirs(db_dir, exist_ok=True)
    db_path = "sqlite:///" + db_dir + "optuna_goalNet.db"
    print("db_path: ", db_path)
    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(model_storage_path, exist_ok=True)
    study = optuna.create_study(direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=1), 
                study_name=study_name, 
                storage=db_path, 
                load_if_exists=True)
    study.optimize(objective, n_trials=number_of_trials, n_jobs=1, callbacks=[wandbc])
