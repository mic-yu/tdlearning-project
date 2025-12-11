import numpy as np
import torch
import wandb
import copy
import time

from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

from utils import  get_ba_from_conf, get_td_train_data_inf
from train import  eval



def get_target_from_batch(batch, targetModel, cfg, params):
    """
    batch (Tensor): N x ((state s ), (state s'), k (for s), terminal_info (for s'), true_value)
    """
    assert (batch.size(dim=1) - 3) % 2 == 0
    state_dim = int((batch.size(dim=1) - 3) / 2)
    max_k = cfg.max_k
    with torch.no_grad():
        target_input = batch[:, state_dim:-2]
        target_input[:, -1:] = target_input[:, -1:] - 1
        target_input[:, -1:] = target_input[:, -1:] / max_k
        Y = targetModel(target_input)
    
    for idx in range(batch.size(dim=0)):
        if batch[idx][-2] == 1:
            Y[idx][0] = batch[idx][-1]
    
    X = torch.cat((batch[:, :state_dim], batch[:, -3:-2]), dim=1)
    return X, Y

def get_target_from_batch_inf(batch, targetModel, cfg, params):
    """
    Computes Y = td target (targetModel(s') or true_value if s' is terminal)

    batch (Tensor): N x ((state s features), (state s' features), terminal_info (for s'), true_value)
        terminal_info = 1 if s' is terminal and 0 otherwise.
        
    Returns:
        Tuple containing:
        - X (torch.tensor): state s features
        - Y (torch.tensor): td target
    """
    assert (batch.size(dim=1) - 2) % 2 == 0
    state_dim = int((batch.size(dim=1) - 2) / 2)
    with torch.no_grad():
        target_input = batch[:, state_dim:-2]
        Y = targetModel(target_input)
    
    for idx in range(batch.size(dim=0)):
        if batch[idx][-2] == 1:
            Y[idx][0] = batch[idx][-1]
    
    X = batch[:, :state_dim]
    return X, Y

def train_td(trial, wb_run, model, trainDataList, valDataLoader, cfg, params):
    #trainData = get_td_train_data(trainDataList, cfg)
    trainData = get_td_train_data_inf(trainDataList)
    trainData = trainData.to(device=params["device"])
    trainDataset = TensorDataset(trainData)
    generator = torch.Generator().manual_seed(37)
    #debugDataset, _ = torch.utils.data.random_split(trainDataset, [0.2, 0.8], generator=generator)
    trainDataLoader = DataLoader(trainDataset, batch_size=params["bs"], shuffle=True, generator=generator)

    lr = params["lr"]
    loss_fn = params["loss_fn"]
    num_epochs = cfg.num_epochs
    if num_epochs == 0:
        num_epochs = 1
    assert num_epochs > 0


    #Data
    model_path = params["model_storage_path"] #for saving final model
    best_model_path = params["best_model_storage_path"] #for saving best model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sig = nn.Sigmoid()

    #Training loop
    step = 0
    best_val_ba = 0.0
    best_val_loss = torch.tensor(float('inf'))
    train_avg_v = 0.0
    running_avg = 0.0
    running_count = 0
    gradient_avg = 0.0
    targetModel = copy.deepcopy(model)
    targetModel.eval()
    targetModel.to(device=params["device"])
    start_time = time.time()
    for epoch in range(num_epochs): 
        model.train()
        for batch, in trainDataLoader:
            X, Y = get_target_from_batch_inf(batch, targetModel, cfg, params)
            optimizer.zero_grad()
            outputs = model(X)
            if cfg.train_forward_sig:
                outputs = sig(outputs)

            loss = loss_fn(outputs, Y)
            #print(f"Step: {step} Loss: {loss.item()}")
            loss.backward()

            batch_avg_gradient = grad_norm(model)
            optimizer.step()
            with torch.no_grad():
                batch_avg_v = outputs.mean().item()
               
                
            
            running_avg = running_avg*running_count + loss.item()*X.size(dim=0)
            train_avg_v = train_avg_v * running_count + batch_avg_v*X.size(dim=0)
            gradient_avg = gradient_avg * running_count + batch_avg_gradient*X.size(dim=0)

            running_count = running_count + X.size(dim=0)
            running_avg = running_avg/running_count
            train_avg_v = train_avg_v / running_count
            gradient_avg = gradient_avg / running_count

            step = step + 1
            if (step + 1) % cfg.train_loss_frequency == 0:
                #print(f"Epoch: {epoch} Step: {step} Loss: {running_avg}")
                wb_run.log({"train/loss": running_avg,
                           "optimization step": step,
                           "train/avg_v": train_avg_v,
                           "train/gradient_avg": gradient_avg
                           })
                running_avg = 0.0
                train_avg_v = 0.0
                gradient_avg = 0.0
                running_count = 0
            if step % params["target_update_frequency"] == 0:
                targetModel = copy.deepcopy(model)
                targetModel.eval()
                targetModel.to(device=params["device"])
        if (epoch + 1) % cfg.eval_frequency == 0:
            val_loss, conf_mx, eval_avg_v = eval(model, valDataLoader, params["loss_fn"], nn_sig=False)
            if trial is not None:
                trial.report(-val_loss, epoch)
            BA = get_ba_from_conf(*conf_mx.ravel())
            elapsed_time = (time.time() - start_time) / 60
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), best_model_path)
                wb_run.summary["Best eval loss"] = val_loss
                wb_run.summary["Best Epoch"] = epoch
                best_val_loss = val_loss
            wandb_report = {"epoch": epoch,
                "optimization step": step,
                "train/loss": running_avg,
                "validation/loss": val_loss,
                "validation/Balanced Accuracy": BA,
                "validation/avg_v": eval_avg_v,
                "Elapsed Time Minutes": elapsed_time,
                "best_val_loss": best_val_loss
                }
            wb_run.log(wandb_report)
                

            
        #print("Epoch {} complete with avg train loss {}".format(epoch, running_avg))
        #train_loss_list.append(running_avg)



    #model_path = model_storage_path + "/{}_trial_{}".format(study_name, trial.number)
    torch.save(model.state_dict(), model_path)

    fma_name = wb_run.name + "_" + "final_model"
    final_model_artifact = wandb.Artifact(name=fma_name, type="model")

    bma_name = wb_run.name + "_" + "best_model"
    best_model_artifact = wandb.Artifact(name=bma_name, type="model")

    final_model_artifact.add_file(local_path=model_path)
    best_model_artifact.add_file(local_path=best_model_path)

    wb_run.log_artifact(final_model_artifact)
    wb_run.log_artifact(best_model_artifact)
    

    return model

def eval(model, valDataLoader, loss_fn, nn_sig=True):
    """
    Evaluate model on validation set. Return average loss, confusion matrix and recall.
    """
    model.eval()
    running_avg = 0.0
    running_count = 0
    avg_v = 0.0

    conf_mx = np.zeros((2,2), dtype=np.int64)

    with torch.no_grad():
        for X, Y in valDataLoader:
            outputs = model(X, sig=nn_sig)
            loss = loss_fn(outputs, Y)

            batch_avg_v = outputs.mean().item()

            outputs = outputs.squeeze()
            outputs = (outputs > 0.5).float()

            #print("outputs[:5, :]", outputs[:5])
            #print("Y[:5, :]: ", Y[:5, :])

            #print("outputs.shape: ", outputs.shape)
            #print("Y.shape: ", Y.shape)

            running_avg = running_avg*running_count + loss.item()*X.size(dim=0)
            avg_v = avg_v * running_count + batch_avg_v*X.size(dim=0)

            running_count = running_count + X.size(dim=0)
            running_avg = running_avg/running_count
            avg_v = avg_v / running_count

            
            size_y = Y.size()
            size_outputs = outputs.size()
            if size_y[0] <= 1 or size_outputs[0] <= 1:
                print("size_y: ", size_y)
                print("size_outputs: ", size_outputs)
            mx = confusion_matrix(Y.squeeze().int().numpy(force=True), outputs.int().numpy(force=True), labels=[0, 1])
            conf_mx = conf_mx + mx
            

            
    return running_avg, conf_mx, avg_v


def grad_norm(model):
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item()
            count += 1
    return total / count if count > 0 else 0.0
