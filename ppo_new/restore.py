import os, sys
sys.path.append('..')
from baselines.common import tf_util as U
from ppo_new.run import train
import tensorflow as tf
from env.LaneChangeEnv import LaneChangeEnv
import numpy as np
import traci

def main():
    """
    restore latest model from ckpt
    """
    model_dir = '../tf_models/trial9'
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model_path = latest_checkpoint

    EP_MAX = 20
    EP_LEN_MAX = 1000

    # train flag check: train or animate trained results
    # animate trained results
    pi = train(max_iters=1, callback=None)
    U.load_state(model_path)

    env = LaneChangeEnv(gui=True, label='1', is_train=False)
    sumoseed = 45  #44
    randomseed = 45  # 6 9

    for ep in range(EP_MAX):
        # if env.is_collision:
        #     print('sumoseed:', sumoseed, 'randomseed:', randomseed)
        #     break
        sumoseed += 0
        randomseed += 0
        print('sumoseed:', sumoseed, 'randomseed:', randomseed)
        ob = env.reset(tlane=0, tfc=2, is_gui=True, sumoseed=sumoseed, randomseed=randomseed)
        # ob = env.reset(tlane=0, tfc=2, is_gui=True, sumoseed=None, randomseed=None)

        traci.vehicle.setColor(env.egoID, (255, 69, 0))
        ob_np = np.asarray(ob).flatten()
        speed_list = []
        lat_speed_list = []
        for t in range(EP_LEN_MAX):
            ac = pi.act(stochastic=False, ob=ob_np)[0]
            ob, reward, done, info = env.step(ac)  # need modification
            speed_list.append(env.ego.speed)
            lat_speed_list.append(env.ego.speed_lat)
            ob_np = np.asarray(ob).flatten()
            if done:
                break
        np_array = np.vstack([np.linspace(0, len(speed_list)-1, num=len(speed_list)), speed_list, lat_speed_list]).T
        if ep == 1:
            np.savetxt('../data/final.csv', np_array, delimiter=",")

if __name__ == '__main__':
    main()