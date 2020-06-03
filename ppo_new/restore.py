import os, sys
sys.path.append('..')
from baselines.common import tf_util as U
from ppo_new.run import train
import tensorflow as tf
from env.LaneChangeEnv import LaneChangeEnv
import numpy as np
os.environ["SUMO_HOME"] = "/usr/local/Cellar/sumo/1.6.0/share/sumo"

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    print(tools)
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

def main():
    """
    restore latest model from ckpt
    """
    model_dir = '../tf_models/trial11'
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model_path = latest_checkpoint

    EP_MAX = 100
    EP_LEN_MAX = 1000

    # train flag check: train or animate trained results
    # animate trained results
    pi = train(max_iters=1, callback=None)
    U.load_state(model_path)

    env = LaneChangeEnv(gui=True)
    sumoseed = 11
    randomseed = 11  # 6 9
    for ep in range(EP_MAX):
        sumoseed += 1
        randomseed += 1
        print('sumoseed:', sumoseed, 'randomseed:', randomseed)
        ob = env.reset(tlane=0, tfc=2, is_gui=True, sumoseed=sumoseed, randomseed=randomseed)
        traci.vehicle.setColor(env.egoID, (255, 69, 0))
        ob_np = np.asarray(ob).flatten()
        for t in range(EP_LEN_MAX):
            ac = pi.act(stochastic=False, ob=ob_np)[0]
            ob, reward, done, info = env.step(ac)  # need modification
            ob_np = np.asarray(ob).flatten()

            if done:
                break

if __name__ == '__main__':
    main()