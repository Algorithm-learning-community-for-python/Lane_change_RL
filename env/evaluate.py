import os, sys
os.environ["SUMO_HOME"] = "/usr/local/Cellar/sumo/1.2.0/share/sumo"
from env.LaneChangeEnv import LaneChangeEnv
import random
import numpy as np
import tensorflow as tf
from ppo_new.run import train
from baselines.common import tf_util as U


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


def episode_generator(pi, env, is_gui, sumoseed, randomseed):
    egoid = 'lane1.' + str(random.randint(1, 6))
    ob = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=is_gui, sumoseed=sumoseed, randomseed=randomseed)
    traci.vehicle.setColor(egoid, (255, 69, 0))

    cur_ep_ret = 0  # return in current episode
    cur_ep_ret_detail = 0
    cur_ep_len = 0  # len of current episode
    cur_ep_obs = []
    cur_ep_acs = []
    while True:
        # ac, _ = pi(ob=ob, safety_gap=sg)
        ac = pi.act(stochastic=False, ob=ob)[0]
        ob, rew, new, info = env.step(ac)

        cur_ep_ret += rew
        cur_ep_ret_detail += np.array(list(info['reward_dict'].values()))
        cur_ep_len += 1
        cur_ep_obs.append(ob)
        cur_ep_acs.append(ac)
        if new:
            return {"ep_obs": cur_ep_obs, "ep_acs": cur_ep_acs,
                   "ep_ret": cur_ep_ret, 'ep_rets_detail': cur_ep_ret_detail, "ep_len": cur_ep_len,
                   'ep_num_danger': info['num_danger'], 'ep_is_success': info['is_success']}


def pi_baseline(**kwargs):
    ob = kwargs['ob']
    safety_gap = kwargs['safety_gap']
    # safety gap set to seconds to collision
    TTC = (ob[3 * 4 + 0 + 1] - 5) / max(ob[1] - ob[3 * 4 + 1 + 1], 0.001)
    TTC2 = (abs(ob[4 * 4 + 0 + 1]) - 5) / max(ob[4 * 4 + 1 + 1] - ob[1], 0.001)
    if TTC > safety_gap and TTC2 > safety_gap:
        ac_lat = 1  # change lane
    else:
        ac_lat = 0  # abort
    if ob[2] < 3.2:
        # follow target leader
        ac_longi = 1
    else:
        # follow original leader
        ac_longi = 0
    ac = ac_longi * 3 + ac_lat
    return ac, None


# def evaluate(num_eps, is_gui, safety_gap):
#     sumoseed = 0
#     randomseed = 0
#     env = LaneChangeEnv()
#
#     ret_eval = 0
#     ret_det_eval = 0  # not a integer, will be broadcasted
#     collision_num = 0
#     success_num = 0
#     for i in range(num_eps):
#         ep_eval = episode_generator(pi_baseline, env, sg=safety_gap, is_gui=is_gui, sumoseed=sumoseed, randomseed=randomseed)
#
#         ret_eval += ep_eval['ep_ret']
#         # ep_rets_detail_np = np.vstack([ep_ret_detail for ep_ret_detail in ep_eval['ep_rets_detail']])
#         # ret_det_eval += np.mean(ep_rets_detail_np, axis=0)
#         ret_det_eval += ep_eval['ep_rets_detail']
#         collision_num += int(ep_eval['ep_is_collision'])
#         success_num += int(ep_eval['ep_is_success'])
#
#         f.write('%s,%s,%s,%s,' % (sumoseed, randomseed, ep_eval['ep_is_success'], ep_eval['ep_is_collision']))
#         for i_obs in range(21):
#             f.write(str(ep_eval['ep_obs'][-1][i_obs]) + ',')
#         f.write('\n')
#
#     ret_eval /= float(num_eps)
#     ret_det_eval /= float(num_eps)
#     collision_rate = collision_num / float(num_eps)
#     success_rate = success_num / float(num_eps)
#
#     print(ret_det_eval)
#     print('safety_gap:', safety_gap,
#           'reward:', ret_eval,
#           '\ncollision_rate:', collision_rate,
#           '\nsuccess_rate:', success_rate)
#     return ret_eval, collision_rate, success_rate


def evaluate_ppo(num_eps, is_gui):
    sumoseed = 0
    randomseed = 0
    env = LaneChangeEnv()

    model_dir = '../tf_models/trial6'
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model_path = latest_checkpoint
    pi = train(max_iters=1, callback=None)
    U.load_state(model_path)

    ret_eval = 0
    ret_det_eval = 0  # not a integer, will be broadcasted
    danger_list = []
    ep_len_list = []
    success_num = 0
    for i in range(num_eps):
        ep_eval = episode_generator(pi, env, is_gui=is_gui, sumoseed=sumoseed, randomseed=randomseed)

        ret_eval += ep_eval['ep_ret']
        # ep_rets_detail_np = np.vstack([ep_ret_detail for ep_ret_detail in ep_eval['ep_rets_detail']])
        # ret_det_eval += np.mean(ep_rets_detail_np, axis=0)
        ret_det_eval += ep_eval['ep_rets_detail']
        danger_list.append(1 if ep_eval['ep_num_danger'] > 0 else 0)
        success_num += int(ep_eval['ep_is_success'])
        if ep_eval['ep_is_success']:
            ep_len_list.append(ep_eval['ep_len'])
        sumoseed += 1
        randomseed += 1
        # f.write('%s,%s,%s,%s,' % (sumoseed, randomseed, ep_eval['ep_is_success'], ep_eval['ep_num_danger']))
        # for i_obs in range(21):
        #     f.write(str(ep_eval['ep_obs'][-1][i_obs]) + ',')
        # f.write('\n')

    ret_eval /= float(num_eps)
    ret_det_eval /= float(num_eps)
    danger_rate = np.mean(danger_list)
    success_rate = success_num / float(num_eps)
    sucess_len = np.mean(ep_len_list)
    print(ret_det_eval)
    print('reward:', ret_eval,
          '\ndanger_rate:', danger_rate,
          '\nsuccess_rate:', success_rate,
          '\nsucess_len', sucess_len)
    return ret_eval, danger_rate, success_rate


NUM_EPS = 100
IS_GUI = False
rw, coll_rate, succ_rate = evaluate_ppo(NUM_EPS, IS_GUI)

# f = open('../data/baseline_evaluation/testseed2.csv', 'w+')
# safety_gap = 2
constraints_list = [3.0]  # [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
# for safety_gap in constraints_list:
    # f.write('sumoseed,randomseed,is_sucess,is_collision,'
    #         'ego_pos_longi,ego_speed,ego_pos_lat,ego_acce,ego_speed_lat,'
    #         'ol_dis2ego,ol_speed,ol_pos_lat,ol_acce,'
    #         'of_dis2ego,of_speed,of_pos_lat,of_acce,'
    #         'tl_dis2ego,tl_speed,tl_pos_lat,tl_acce,'
    #         'tf_dis2ego,tf_speed,tf_pos_lat,tf_acce,\n')

        # f.write('safety_gap, reward, collision_rate, success_rate\n')
        # f.write('%s, %s, %s, %s\n' % (safety_gap, rw, coll_rate, succ_rate))

    # safety_gap_list = [1, 5, 10, 15, 20, 30]
    # for safety_gap in safety_gap_list:
    #     rw, coll_rate, succ_rate = evaluate(HORIZON, NUM_HORIZON, IS_GUI, safety_gap)
    #     f.write('%s, %s, %s, %s\n' % (safety_gap, rw, coll_rate, succ_rate))