import os, sys
sys.path.append('..')
from baselines.common import tf_util as U
from ppo_new.run import train
import tensorflow as tf
from env.LaneChangeEnv import LaneChangeEnv
import numpy as np
os.environ["SUMO_HOME"] = "/usr/share/sumo"
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
    model_dir = '../tf_models/test'
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model_path = latest_checkpoint

    EP_MAX = 10
    EP_LEN_MAX = 1000

    # train flag check: train or animate trained results
    # animate trained results
    pi = train(max_iters=1, callback=None)
    U.load_state(model_path)

    env = LaneChangeEnv()
    for ep in range(EP_MAX):
        ob = env.reset(tlane=0, tfc=2, is_gui=True, sumoseed=None, randomseed=None)
        traci.vehicle.setColor(env.egoID, (255, 69, 0))
        ob_np = np.asarray(ob).flatten()
        for t in range(EP_LEN_MAX):
            ac = pi.act(stochastic=False, ob=ob_np)[0]

            ob, reward, done, info = env.step(ac)  # need modification
            ob_np = np.asarray(ob).flatten()

            is_end_episode = done and info['resetFlag']
            if is_end_episode:
                break


if __name__ == '__main__':
    main()