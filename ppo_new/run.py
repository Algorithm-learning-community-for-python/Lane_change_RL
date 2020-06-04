#!/usr/bin/env python3
import os
import sys
# os.environ["SUMO_HOME"] = "/usr/share/sumo"
os.environ["SUMO_HOME"] = "/usr/local/Cellar/sumo/1.6.0/share/sumo"
sys.path.append('..')
from baselines.common import tf_util as U
from baselines import logger
from env.LaneChangeEnv import LaneChangeEnv
from ppo_new import ppo_sgd_simple
import random, sys, os
from mpi4py import MPI
import argparse

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


def train(max_iters, callback=None):
    from baselines.ppo1 import mlp_policy
    U.make_session(num_cpu=8).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=128, num_hid_layers=2)  # 64
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
                              timesteps_per_actorbatch=2048,  # 4096 512
                              clip_param=0.2, entcoeff=0,  # 0.2 0.0
                              optim_epochs=5,
                              optim_stepsize=5e-3,  # 1e-4 1e-3
                              optim_batchsize=512,  # 512
                              gamma=0.99,  # look forward 1.65s
                              lam=0.95,
                              callback=callback,
                              schedule='constant',  # constant',
                              continue_from=args.cont if 'args' in globals().keys() else None)
    env.close()

    return pi


def callback(locals_, globals_):
    saver_ = locals_['saver']
    sess_ = U.get_session()
    timesteps_so_far_ = locals_['timesteps_so_far']
    iters_so_far_ = locals_['iters_so_far']
    model_dir = '../tf_models/' + args.log
    if MPI.COMM_WORLD.Get_rank() == 0 and iters_so_far_ % 30 == 0:
        saver_.save(sess_, model_dir + '/model', global_step=timesteps_so_far_)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", type=str, default='',
                        help="name of saved log folder & model folder")  # eg. trial2
    parser.add_argument("-c", "--cont", type=str, default='',
                        help="folder to continue training from")  # eg. opt_gait_interp_trial3
    args = parser.parse_args()

    os.environ['OPENAI_LOGDIR'] = "~/Desktop/Lane_change_RL/tf_models/" + args.log
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,tensorboard'
    # if not os.path.exists('../tf_models/' + args.log):
    #     os.mkdir('../tf_models/' + args.log)
    # sys.stdout = open('../tf_models/' + args.log + '/logstdout.txt', 'a')

    logger.configure()
    train(max_iters=2000, callback=callback)

