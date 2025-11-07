import pickle
import numpy as np

from math import sqrt
from matplotlib import pyplot as plt


def plot_ground_truth(path):
    with open(path, 'rb') as f:
        gt = pickle.load(f)
    
    L = len(gt)
    scale = int(sqrt(L / 54))
    print("scale:", scale)
    num_x = 9*scale
    num_y = 6*scale

    gt = np.array(gt)
    gt = np.reshape(gt, shape=(num_y, num_x))

    plt.imshow(gt, extent=[-4500, 4500, -3000, 3000], origin='lower')
    plt.colorbar()
    #plt.title(f"Hor {horizon} \n Opponent goal is on the right.")
    # plt.xlabel(f"Each point on the heatmap is at the location of the agent, and \n" + \
    #            f"the ball is {dx} to the right and {dy} above the agent. \n" + \
    #             f"Agent angle is {agent_z}.")
    plt.show()


if __name__ == "__main__":
    #path = "./../abstract-sim/abstractSim/data/obs/near_ball_dx_100_scale_4_2025-09-21_22-22-56.pkl"
    #path = "./../abstract-sim/abstractSim/data/obs/near_ball_dx_100_scale_4_h_1002025-09-21_22-35-50.pkl"
    #path = "./../abstract-sim/abstractSim/data/obs/near_ball_dx_100_scale_6_h_1002025-09-21_22-39-25.pkl"
    #path = "./../abstract-sim/abstractSim/data/obs/near_ball_dx_100_scale_8_h_1002025-09-21_22-47-14.pkl"
    #path = "./../abstract-sim/abstractSim/data/obs/near_ball_dx_100_scale_10_h_1002025-09-21_22-55-15.pkl"
    path = "./../abstract-sim/abstractSim/data/obs/near_ball_dx_1000_scale_10_h_1002025-09-23_00-52-51.pkl"
    plot_ground_truth(path)