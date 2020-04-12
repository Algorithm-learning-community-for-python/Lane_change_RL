#!/usr/bin/env python3
import os
import sys
os.environ["SUMO_HOME"] = "/usr/share/sumo"
sys.path.append('..')
from baselines.common import tf_util as U
from baselines import logger
from env.LaneChangeEnv import LaneChangeEnv
from ppo_new import ppo_sgd_simple
import random, sys, os
from baselines.logger import configure
from mpi4py import MPI
import numpy as np
import tensorflow as tf
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

os.environ['OPENAI_LOGDIR'] = '~/projects/Lane_change_RL/logs/test'
os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,tensorboard'


def train(max_iters, callback=None):
    from baselines.ppo1 import mlp_policy
    U.make_session(num_cpu=6).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = LaneChangeEnv()

    # pi = ppo_sgd.learn(env, policy_fn,
    #                    max_timesteps=num_timesteps,
    #                    timesteps_per_actorbatch=512,
    #                    clip_param=0.1, entcoeff=0.0,
    #                    optim_epochs=16,
    #                    optim_stepsize=1e-4,
    #                    optim_batchsize=64,
    #                    gamma=0.99,
    #                    lam=0.95,
    #                    schedule='constant',
    #                    is_train=is_train)
    pi = ppo_sgd_simple.learn(env, policy_fn,
                              max_iters=max_iters,
                              timesteps_per_actorbatch=1024,  # 4096 512
                              clip_param=0.2, entcoeff=0,  # 0.2 0.0
                              optim_epochs=2,
                              optim_stepsize=2e-4,  # 1e-4
                              optim_batchsize=256,  # 512
                              gamma=0.99,  # look forward 1.65s
                              lam=0.95,
                              callback=callback,
                              schedule='constant',  # constant',
                              )
    env.close()

    return pi

def callback(locals_, globals_):
    saver_ = locals_['saver']
    sess_ = U.get_session()
    timesteps_so_far_ = locals_['timesteps_so_far']
    iters_so_far_ = locals_['iters_so_far']
    model_dir = '../tf_models/test'
    if MPI.COMM_WORLD.Get_rank() == 0 and iters_so_far_ % 30 == 0:
        saver_.save(sess_, model_dir + '/model', global_step=timesteps_so_far_)
    return True


def main():
    configure()
    train(max_iters=2000, callback=callback)
    # # logger.configure()
    # is_train = True
    # model_dir = '../../tf_model/11'
    # latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    # model_path = latest_checkpoint
    # EP_MAX = 10
    # EP_LEN_MAX = 1000
    #
    # if is_train:
    #     # train the model
    #     train(num_timesteps=1000000, is_train=True)
    # else:
    #     # animate trained results
    #     pi = train(num_timesteps=1, is_train=False)
    #     U.load_state(model_path)
    #
    #     env = LaneChangeEnv()
    #     for ep in range(EP_MAX):
    #         egoid = 'lane1.' + str(random.randint(1, 5))
    #         # todo set sumoseed and randomseed as fixed
    #         ob = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=True, sumoseed=None, randomseed=None)
    #         traci.vehicle.setColor(egoid, (255, 69, 0))
    #         ob_np = np.asarray(ob).flatten()
    #         for t in range(EP_LEN_MAX):
    #             ac = pi.act(stochastic=False, ob=ob_np)[0]
    #
    #             ob, reward, done, info = env.step(ac)  # need modification
    #             ob_np = np.asarray(ob).flatten()
    #
    #             is_end_episode = done and info['resetFlag']
    #             if is_end_episode:
    #                 break


if __name__ == '__main__':
    main()
