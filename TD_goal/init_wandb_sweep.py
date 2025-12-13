import wandb
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg):

    parameters = {}
    n_layers = cfg.wandb_sweep.n_layers
    neuron_min = cfg.search_space.num_neurons_range[0]
    neuron_max = cfg.search_space.num_neurons_range[1]
    for i in range(n_layers):
        parameters[f'neurons_layer_{i}'] = {"min": neuron_min, "max": neuron_max, "distribution": "int_uniform"}
    
    bs_min = cfg.search_space.bs_range[0]
    bs_max = cfg.search_space.bs_range[1]
    bs_step = cfg.search_space.bs_step
    bs_values = list(range(bs_min, bs_max+1, bs_step))
    parameters["bs"] = {"values": bs_values, "distribution": "categorical"}
    
    lr_min = cfg.search_space.lr_range[0]
    lr_max = cfg.search_space.lr_range[1]
    parameters["lr"] = {"min": lr_min, "max": lr_max, "distribution": "log_uniform_values"}
    tuf_min = cfg.search_space.target_update_frequency_range[0]
    tuf_max = cfg.search_space.target_update_frequency_range[1]
    tuf_step = cfg.search_space.tuf_step
    tuf_values = list(range(tuf_min, tuf_max + 1, tuf_step))
    #parameters["target_update_frequency"] = {"values": tuf_values, "distribution": "categorical"}
    alpha_ema_min = cfg.search_space.alpha_ema_range[0]
    alpha_ema_max = cfg.search_space.alpha_ema_range[1]
    #parameters["alpha_ema"] = {"min": alpha_ema_min, "max": alpha_ema_max, "distribution": "log_uniform_values"}

    sweep_configuration ={
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
        "method": "bayes",
        "name": cfg.wandb_sweep.name,
        "metric": {"goal": "maximize", "name": "best_val_ba"},
        "parameters": parameters
    }
    
    sweep_id = wandb.sweep(sweep=sweep_configuration)
    print("sweep_id: ", sweep_id)

if __name__ == '__main__':
    run()