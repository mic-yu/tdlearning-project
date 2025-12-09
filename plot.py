import os
import torch
import numpy as np
import optuna
import pickle

from GoalNet import GoalNet
from utils import get_relative_observation, get_hidden_sizes_from_optuna
from artifact_download import load_wandb_model
from cal_eval_dev import load_iso_reg

from matplotlib import pyplot as plt

def plot_fixed_agent_td(model, iso_reg, abs_agent_loc, horizon, scale=10):
    max_horizon = 100
    num_x = 9*scale
    num_y = 6*scale
    
    agent_x = abs_agent_loc[0]
    agent_y = abs_agent_loc[1]
    agent_z = abs_agent_loc[2]

    x = np.linspace(-4500, 4500, num=num_x)
    y = np.linspace(-3000, 3000, num=num_y)
    xv, yv = np.meshgrid(x,y)
    #print("xv.shape: ", xv.shape)



    abs_ball = np.stack((xv.flatten(), yv.flatten()), axis=1)
    rel_ball = []
    for i, ball_loc in enumerate(abs_ball):
        rel_ball.append(get_relative_observation(np.array(abs_agent_loc), ball_loc))


    rel_goal = get_relative_observation(np.array(abs_agent_loc), (4500, 0))
    #print("rel_goal: ", rel_goal)
    rel_goal = torch.tensor(rel_goal).to(dtype=torch.float32)
    rel_goal = rel_goal.repeat((num_x*num_y, 1))


    rel_ball = np.array(rel_ball)
    rel_ball = torch.from_numpy(rel_ball).to(dtype=torch.float32)
    rel_input = torch.cat((rel_ball, rel_goal), dim=1) #(num_x*num_y , 8)
    #print("rel_input.size(): ", rel_input.size())

    #adjustments to my td model that has last input feature as k/max_k
    tile_length = rel_input.size(dim=0)
    horizon_feature = torch.tensor(horizon / max_horizon).tile((tile_length, 1))
    rel_input = torch.cat((rel_input, horizon_feature), dim=1)

    output = model(rel_input)
    output = iso_reg.transform(output.numpy(force=True))
    # output = torch.from_numpy(output)
    # output = output.view(num_y, num_x)
    output = np.reshape(output, shape=(num_y, num_x))
    #print("output.size: ", output.shape)

    plt.imshow(output, extent=[-4500, 4500, -3000, 3000], origin='lower')
    plt.colorbar()
    plt.title(f"Horizon {horizon} \n Opponent goal is on the right.")
    plt.xlabel(f"Agent coordinates are ({agent_x}, {agent_y}, {agent_z})")
    plt.show()

def plot_fixed_abs_agent(model, iso_reg, abs_agent_loc, scale=10):
    num_x = 9*scale
    num_y = 6*scale
    
    agent_x = abs_agent_loc[0]
    agent_y = abs_agent_loc[1]
    agent_z = abs_agent_loc[2]

    x = np.linspace(-4500, 4500, num=num_x)
    y = np.linspace(-3000, 3000, num=num_y)
    xv, yv = np.meshgrid(x,y)
    #print("xv.shape: ", xv.shape)



    abs_ball_mesh = np.stack((xv.flatten(), yv.flatten()), axis=1)
    abs_ball = []
    for i, ball_loc in enumerate(abs_ball_mesh):
        abs_ball.append(np.append(ball_loc, 0))



    abs_goal = np.array((abs_agent_loc[0], 0))
    #print("rel_goal: ", rel_goal)
    abs_goal = torch.tensor(abs_goal).to(dtype=torch.float32)
    abs_goal = abs_goal.repeat((num_x*num_y, 1))

    print("abs_goal.size()", abs_goal.size())
    abs_ball = np.array(abs_ball)
    abs_ball = torch.from_numpy(abs_ball).to(dtype=torch.float32)
    print("abs_ball.size()", abs_ball.size())
    abs_input = torch.cat((abs_goal, abs_ball), dim=1) #(num_x*num_y , 8)
    #print("rel_input.size(): ", rel_input.size())
    print("abs_input.size()", abs_input.size())

    output = model(abs_input)
    output = iso_reg.transform(output.numpy(force=True))
    # output = torch.from_numpy(output)
    # output = output.view(num_y, num_x)
    output = np.reshape(output, shape=(num_y, num_x))
    #print("output.size: ", output.shape)

    plt.imshow(output, extent=[-4500, 4500, -3000, 3000], origin='lower')
    plt.colorbar()
    plt.title(f"Horizon {horizon} \n Opponent goal is on the right.")
    plt.xlabel(f"Agent coordinates are ({agent_x}, {agent_y}, {agent_z})")
    plt.show()

def plot_fixed_agent(model, iso_reg, abs_agent_loc, scale=10):
    num_x = 9*scale
    num_y = 6*scale
    
    agent_x = abs_agent_loc[0]
    agent_y = abs_agent_loc[1]
    agent_z = abs_agent_loc[2]

    x = np.linspace(-4500, 4500, num=num_x)
    y = np.linspace(-3000, 3000, num=num_y)
    xv, yv = np.meshgrid(x,y)
    #print("xv.shape: ", xv.shape)



    abs_ball = np.stack((xv.flatten(), yv.flatten()), axis=1)
    rel_ball = []
    for i, ball_loc in enumerate(abs_ball):
        rel_ball.append(get_relative_observation(np.array(abs_agent_loc), ball_loc))


    rel_goal = get_relative_observation(np.array(abs_agent_loc), (4500, 0))
    #print("rel_goal: ", rel_goal)
    rel_goal = torch.tensor(rel_goal).to(dtype=torch.float32)
    rel_goal = rel_goal.repeat((num_x*num_y, 1))


    rel_ball = np.array(rel_ball)
    rel_ball = torch.from_numpy(rel_ball).to(dtype=torch.float32)
    rel_input = torch.cat((rel_ball, rel_goal), dim=1) #(num_x*num_y , 8)
    #print("rel_input.size(): ", rel_input.size())

    output = model(rel_input)
    output = iso_reg.transform(output.numpy(force=True))
    # output = torch.from_numpy(output)
    # output = output.view(num_y, num_x)
    output = np.reshape(output, shape=(num_y, num_x))
    #print("output.size: ", output.shape)

    plt.imshow(output, extent=[-4500, 4500, -3000, 3000], origin='lower')
    plt.colorbar()
    plt.title(f"Horizon {horizon} \n Opponent goal is on the right.")
    plt.xlabel(f"Agent coordinates are ({agent_x}, {agent_y}, {agent_z})")
    plt.show()

def plot_agent_near_ball(model, iso_reg, delta, agent_z=0, scale=10):
    num_x = 9*scale
    num_y = 6*scale

    x = np.linspace(-4500, 4500, num=num_x)
    y = np.linspace(-3000, 3000, num=num_y)
    xv_agent, yv_agent = np.meshgrid(x,y)   

    dx = delta[0]
    dy = delta[1]
    xv_ball = xv_agent + dx
    yv_ball = yv_agent + dy

    rel_ball = get_relative_observation([0, 0, agent_z], [dx, dy])
    rel_ball = torch.tensor((rel_ball)).to(dtype=torch.float32)
    rel_ball = rel_ball.repeat((num_x*num_y, 1)) 

    abs_agent = np.stack((xv_agent.flatten(), yv_agent.flatten(), np.tile(agent_z, (num_x*num_y,))), axis=1)
    #abs_ball = np.stack((xv_ball.flatten(), yv_ball.flatten()), axis=1)
    rel_goal = []
    for agent_loc in abs_agent:
        rel_goal.append(get_relative_observation(agent_loc, [4500, 0]))
    rel_goal = torch.tensor(rel_goal, dtype=torch.float32)

    rel_input = torch.cat((rel_ball, rel_goal), dim=1) #(num_x*num_y, 8)

    

    output = model(rel_input)
    #print("output nn: ", output)
    output = iso_reg.transform(output.numpy(force=True))
    #print("output iso: ", output)

    output = np.reshape(output, shape=(num_y, num_x))
    print("output.size: ", output.shape)

    plt.imshow(output, extent=[-4500, 4500, -3000, 3000], origin='lower')
    plt.colorbar()
    plt.title(f"Horizon {horizon} \n Opponent goal is on the right.")
    plt.xlabel(f"Each point on the heatmap is at the location of the agent, and \n" + \
               f"the ball is {dx} to the right and {dy} above the agent. \n" + \
                f"Agent angle is {agent_z}.")
    plt.show()

def plot_agent_near_ball_td(model, iso_reg, delta, horizon, agent_z=0, scale=10):
    max_horizon = 100
    num_x = 9*scale
    num_y = 6*scale

    x = np.linspace(-4500, 4500, num=num_x)
    y = np.linspace(-3000, 3000, num=num_y)
    xv_agent, yv_agent = np.meshgrid(x,y)   

    dx = delta[0]
    dy = delta[1]
    xv_ball = xv_agent + dx
    yv_ball = yv_agent + dy

    rel_ball = get_relative_observation([0, 0, agent_z], [dx, dy])
    rel_ball = torch.tensor((rel_ball)).to(dtype=torch.float32)
    rel_ball = rel_ball.repeat((num_x*num_y, 1)) 

    abs_agent = np.stack((xv_agent.flatten(), yv_agent.flatten(), np.tile(agent_z, (num_x*num_y,))), axis=1)
    #abs_ball = np.stack((xv_ball.flatten(), yv_ball.flatten()), axis=1)
    rel_goal = []
    for agent_loc in abs_agent:
        rel_goal.append(get_relative_observation(agent_loc, [4500, 0]))
    rel_goal = torch.tensor(rel_goal, dtype=torch.float32)

    rel_input = torch.cat((rel_ball, rel_goal), dim=1) #(num_x*num_y, 8)

    #adjustments to my td model that has last input feature as k/max_k
    tile_length = rel_input.size(dim=0)
    horizon_feature = torch.tensor(horizon / max_horizon).tile((tile_length, 1))
    rel_input = torch.cat((rel_input, horizon_feature), dim=1)

    

    output = model(rel_input)
    #print("output nn: ", output)
    output = iso_reg.transform(output.numpy(force=True))
    #print("output iso: ", output)

    output = np.reshape(output, shape=(num_y, num_x))
    print("output.size: ", output.shape)

    plt.imshow(output, extent=[-4500, 4500, -3000, 3000], origin='lower')
    plt.colorbar()
    plt.title(f"Horizon {horizon} \n Opponent goal is on the right.")
    plt.xlabel(f"Each point on the heatmap is at the location of the agent, and \n" + \
               f"the ball is {dx} to the right and {dy} above the agent. \n" + \
                f"Agent angle is {agent_z}.")
    plt.show()

def plot_agent_near_ball_abs_inf(model, iso_reg, delta, agent_z=0, scale=10):
    num_x = 9*scale
    num_y = 6*scale

    x = np.linspace(-4500, 4500, num=num_x)
    y = np.linspace(-3000, 3000, num=num_y)
    xv_agent, yv_agent = np.meshgrid(x,y)   

    dx = delta[0]
    dy = delta[1]
    # xv_ball = xv_agent + dx
    # yv_ball = yv_agent + dy

    # rel_ball = get_relative_observation([0, 0, agent_z], [dx, dy])
    # rel_ball = torch.tensor((rel_ball)).to(dtype=torch.float32)
    # rel_ball = rel_ball.repeat((num_x*num_y, 1)) 

    abs_agent = np.stack((xv_agent.flatten(), yv_agent.flatten()), axis=1)
    #abs_ball = np.stack((xv_ball.flatten(), yv_ball.flatten()), axis=1)
    abs_obs = []
    for agent_loc in abs_agent:
        ball_loc = agent_loc + [dx, dy]
        angle = [1, 0]
        single_obs = np.concatenate((agent_loc, ball_loc, angle), axis=None)
        abs_obs.append(single_obs)

    # rel_goal = torch.tensor(rel_goal, dtype=torch.float32)

    # rel_input = torch.cat((rel_ball, rel_goal), dim=1) #(num_x*num_y, 8)
    abs_obs = torch.Tensor(abs_obs)
    abs_obs[:, :-2] = abs_obs[:, :-2]/4500

    
    model.training = False
    output = model(abs_obs)
    #print("output nn: ", output)
    #output = iso_reg.transform(output.numpy(force=True))
    #print("output iso: ", output)

    output = np.reshape(output.numpy(force=True), shape=(num_y, num_x))
    print("output.size: ", output.shape)

    plt.imshow(output, extent=[-4500, 4500, -3000, 3000], origin='lower')
    plt.colorbar()
    plt.title(f"Horizon {horizon} \n Opponent goal is on the right.")
    plt.xlabel(f"Each point on the heatmap is at the location of the agent, and \n" + \
               f"the ball is {dx} to the right and {dy} above the agent. \n" + \
                f"Agent angle is {agent_z}.")
    plt.show()

if __name__ == "__main__":
    horizon = -1 #iteratons
    #cluster = 4357100
    cluster = 4600982
    cluster = 4638940
    #temporary:
    #reflects personal file structure
    match horizon:
        case 10:
            job = 2
        case 100:
            job = 0
        case 50:
            job = 1

    job = 0

    storage_path = f"./../roboClassifier/abs_output_files/transfer_models_{cluster}_{job}_{horizon}/models/"
    db_dir = f"./../roboClassifier/abs_output_files/transfer_databases_{cluster}_{job}_{horizon}/databases/"
    db_path = f"sqlite:///" + db_dir + "optuna_goalNet.db"

    # storage_path = f"./transfer/transfer_models_4346754_{job}_{horizon}/models/"
    # db_dir = f"./transfer/transfer_databases_4346754_{job}_{horizon}/databases/"
    # db_path = f"sqlite:///" + db_dir + "optuna_goalNet.db"
    
    
    study_name = "horizon_{}_goalNet_BABCE_BA".format(horizon)
    iso_reg_save_path = f"./../roboClassifier/iso_reg_{cluster}_{horizon}.pkl"
    input_size = 6

    study = optuna.load_study(study_name=study_name, storage=db_path)
    trial = study.best_trial
    hidden_sizes = get_hidden_sizes_from_optuna(db_path, study_name, trial)
    model_path = f"{storage_path}{study_name}/{study_name}_trial_{trial.number}.pth"
    assert os.path.exists(model_path)
    model = GoalNet(input_size, hidden_sizes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()

    # with open(iso_reg_save_path, "rb") as f:
    #     iso_reg = pickle.load(f)

    print("horizon: ", horizon)
    print("job: ", job)
    print("input_size: ", input_size)
    print("hidden_sizes: ", hidden_sizes)
    print("model_path: ", model_path)

    delta = [100, 0]
    plot_agent_near_ball_abs_inf(model, None, delta, agent_z=0, scale=10)
    
    

    '''
    abs_agent_loc = [2000, 0, 0]
    plot_fixed_abs_agent(model, iso_reg, abs_agent_loc)
    '''

    # for x in [-4000, -2000, -1000, 0, 1000, 2000, 4000]:
    #     abs_agent_loc = [x, 0, 0]
    #     plot_fixed_agent(model, iso_reg, abs_agent_loc)
    # for a in [0, 45, 90, 135, 180, 225, 270, 315]:
    #     abs_agent_loc = [0,0, a]
    #     plot_fixed_agent(model, iso_reg, abs_agent_loc)

    # for dx in [-1000, -500, -200, -100, 0, 100, 200, 500, 1000]:
    #     delta = [dx, 0]
    #     plot_agent_near_ball(model, iso_reg, delta, agent_z=0, scale=10)

    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_path = "./artifact_download/tdgoal_h100_ep500_epoch200_l5_trial_2025-08-27_16-10-28_pv5kiybp_best.pth"
    # run_path = "dpinchuk-university-of-wisconsin-madison/tdgoal_h100_ep500_epoch200/pv5kiybp"
    model_path = "./artifact_download/ep_500_abs_l5_BCE_trial_2025-12-01_03-58-56_ssc8vz8m_best.pth"
    run_path = "dpinchuk-university-of-wisconsin-madison/td_abs_inf_500ep/ssc8vz8m"
    assert os.path.exists(model_path)
    model = load_wandb_model(run_path, model_path, device)
    # iso_reg = load_iso_reg("./temp_best_td_model_iso_reg.pth")
    delta = [100, 0]
    plot_agent_near_ball_abs_inf(model, None, delta, agent_z=0, scale=10)
    '''

    # for x in [-4000, -2000, -1000, 0, 1000, 2000, 4000]:
    #     abs_agent_loc = [x, 0, 180]
    #     plot_fixed_agent_td(model, iso_reg, abs_agent_loc, 100)


    # for dx in [-1000, -500, -200, -100, 0, 100, 200, 500, 1000]:
    #     delta = [dx, 0]
    #     plot_agent_near_ball_td(model, iso_reg, delta, 100, agent_z=0, scale=10)
    
    # delta = [1000, 0]
    # plot_agent_near_ball_td(model, iso_reg, delta, 100, agent_z=0, scale=10)






