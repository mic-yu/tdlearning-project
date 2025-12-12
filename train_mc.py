import torch
import copy
import time
import wandb

from torch import nn

from train_td import eval, grad_norm, update_target_network
from utils import  get_ba_from_conf



def train_mc(trial, wb_run, model, trainDataLoader, valDataLoader, cfg, params):

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
    start_time = time.time()
    for epoch in range(num_epochs): 
        model.train()
        for X, Y in trainDataLoader:
           
            optimizer.zero_grad()
            outputs = model(X, sig=False)
            if cfg.train_forward_sig:
                outputs = sig(outputs)

            loss = loss_fn(outputs, Y)
            #print(f"Step: {step} Loss: {loss.item()}")
            loss.backward()

            batch_avg_gradient = grad_norm(model)
            optimizer.step()
            with torch.no_grad():
                if cfg.loss_fn == "BCE" and cfg.train_forward_sig == False:
                    outputs = sig(outputs)
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

        if (epoch + 1) % cfg.eval_frequency == 0:
            if cfg.loss_fn == "BCE":
                nn_sig = True
            elif cfg.loss_fn == "MSE":
                nn_sig = False
            else:
                nn_sig = None
            val_loss, conf_mx, eval_avg_v = eval(model, valDataLoader, params["loss_fn"], nn_sig=nn_sig)
            if trial is not None:
                trial.report(-val_loss, epoch)
            BA = get_ba_from_conf(*conf_mx.ravel())
            elapsed_time = (time.time() - start_time) / 60
            if val_loss <= best_val_loss:
                if cfg.objective_value == "best_val_loss":
                    torch.save(model.state_dict(), best_model_path)
                    wb_run.summary["Best Epoch"] = epoch
                wb_run.summary["Best eval loss"] = val_loss
                best_val_loss = val_loss
            if BA >= best_val_ba:
                if cfg.objective_value == "best_val_ba":
                    torch.save(model.state_dict(), best_model_path)
                    wb_run.summary["Best Epoch"] = epoch
                best_val_ba = BA
                wb_run.summary["Best eval BA"] = best_val_ba
            wandb_report = {"epoch": epoch,
                "optimization step": step,
                "train/loss": running_avg,
                "validation/loss": val_loss,
                "validation/Balanced Accuracy": BA,
                "validation/avg_v": eval_avg_v,
                "Elapsed Time Minutes": elapsed_time,
                "best_val_loss": best_val_loss,
                "best_val_ba": best_val_ba
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