import os, sys
from env.LaneChangeEnv import LaneChangeEnv
import random
import numpy as np
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


def episode_generator(pi, env, is_gui, ttc, gap, sumoseed, randomseed):
    egoid = 'lane1.' + str(random.randint(1, 6))
    ob = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=is_gui, sumoseed=sumoseed, randomseed=randomseed)
    traci.vehicle.setColor(egoid, (255, 69, 0))

    cur_ep_ret = 0  # return in current episode
    cur_ep_ret_detail = 0
    cur_ep_len = 0  # len of current episode
    cur_ep_obs = []
    cur_ep_acs = []
    while True:
        ac = pi(ob=ob, env=env, ttc=ttc, gap=gap)
        ob, rew, new, info = env.step(ac)

        cur_ep_ret += rew
        cur_ep_ret_detail += np.array(list(info['reward_dict'].values()))
        cur_ep_len += 1
        cur_ep_obs.append(ob)
        cur_ep_acs.append(ac)
        if new:
            return {"ep_obs": cur_ep_obs, "ep_acs": cur_ep_acs,
                   "ep_ret": cur_ep_ret, 'ep_rets_detail': cur_ep_ret_detail, "ep_len": cur_ep_len,
                   'ep_num_danger': info['num_danger'], 'ep_is_success': info['is_success'], 'ep_num_crash': info['num_crash'],
                    'ep_is_collision': info["is_collision"]}


def pi_baseline(ob, env, ttc, gap):
    # safety gap set to seconds to collision
    if env.ego.trgt_leader:
        leader_speed = env.ego.trgt_leader.speed
    else:
        leader_speed = env.ego.speed
    if env.ego.trgt_follower:
        follower_speed = env.ego.trgt_follower.speed
    else:
        follower_speed = env.ego.speed
    leader_dis = abs(ob[3 * 4 + 0 + 1])*239.8
    follower_dis = abs(ob[4 * 4 + 0 + 1])*239.8
    TTC = (leader_dis - 5) / max(env.ego.speed, 0.001)
    TTC2 = (follower_dis - 5) / max(follower_speed, 0.001)
    # print(TTC, TTC)
    if TTC > ttc and TTC2 > ttc and leader_dis > gap and follower_dis > gap:
        ac_lat = 1  # change lane
    else:
        ac_lat = 0  # abort
    ac = ac_lat * 3 + 1
    return ac


def evaluate_baseline(num_eps, ttc, gap, is_gui):
    sumoseed = 0
    randomseed = 0
    pi = pi_baseline

    env = LaneChangeEnv(is_train=False)
    ret_eval = 0
    ret_det_eval = 0  # not a integer, will be broadcasted
    danger_num = 0
    crash_num = 0
    level_1_danger = []
    level_2_danger = []
    collision_num = 0
    ep_len_list = []
    success_num = 0
    for i in range(num_eps):
        ep_eval = episode_generator(pi, env, is_gui=is_gui, ttc=ttc, gap=gap, sumoseed=sumoseed, randomseed=randomseed)

        ret_eval += ep_eval['ep_ret']
        ret_det_eval += ep_eval['ep_rets_detail']
        danger_num += ep_eval['ep_num_danger']
        crash_num += ep_eval['ep_num_crash']
        level_1_danger.append(1 if ep_eval['ep_num_danger'] > 0 else 0)
        level_2_danger.append((1 if ep_eval['ep_num_crash'] > 0 else 0))
        collision_num += ep_eval['ep_is_collision']
        success_num += int(ep_eval['ep_is_success'])
        if ep_eval['ep_is_success']:
            ep_len_list.append(ep_eval['ep_len'])
        sumoseed += 1
        randomseed += 1

    ret_eval /= float(num_eps)
    ret_det_eval /= float(num_eps)
    danger_rate = danger_num / num_eps
    crash_rate = crash_num / num_eps
    level_1_danger_rate = np.mean(level_1_danger)
    level_2_danger_rate = np.mean(level_2_danger)
    coll_rate = collision_num / num_eps
    success_rate = success_num / float(num_eps)
    success_len = np.mean(ep_len_list)
    print('reward_detail: ', ret_det_eval)
    print('reward: ', ret_eval,
          '\ndanger_rate: ', danger_rate,
          '\ncrash_rate: ', crash_rate,
          '\nlevel-1-danger_rate: ', level_1_danger_rate,
          '\nlevel-2-danger_rate: ', level_2_danger_rate,
          '\ncollision_rate: ', coll_rate,
          '\nsuccess_rate: ', success_rate,
          '\nsucess_len: ', success_len)
    env.close()
    return ret_eval, danger_rate, crash_rate, level_1_danger_rate, level_2_danger_rate, coll_rate, success_rate, success_len


NUM_EPS = 100
IS_GUI = False


# f = open('../data/baseline_evaluation/testseed2.csv', 'w+')
# safety_gap = 2
constraints_list = [3.0]  # [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
ttcs = [0.1, 0.3, 0.5, 1, 2, 3]
# ttcs = [2]
gap = 0

reward_list = []
danger_rate_list = []
crash_rate_list = []
level_1_danger_list = []
level_2_danger_list = []
coll_rate_list = []
succ_rate_list = []
succ_len_list = []
for ttc in ttcs:
    ret_eval, danger_rate, crash_rate, level_1_danger_rate, level_2_danger_rate, coll_rate, success_rate, success_len = evaluate_baseline(NUM_EPS, ttc, gap, IS_GUI)
    reward_list.append(ret_eval)
    danger_rate_list.append(danger_rate)
    crash_rate_list.append(crash_rate)
    level_1_danger_list.append(level_1_danger_rate)
    level_2_danger_list.append(level_2_danger_rate)
    coll_rate_list.append(coll_rate)
    succ_rate_list.append(success_rate)
    succ_len_list.append(success_len)

print('reward: ', reward_list)
print('danger rate: ', danger_rate_list)
print('crash rate: ', crash_rate_list)
print('level-1-danger_rate: ', level_1_danger_list)
print('level-2-danger_rate: ', level_2_danger_list)
print('collison rate: ', coll_rate_list)
print('success rate: ', succ_rate_list)
print('sucess len: ', succ_len_list)

# reward:  [-89.12552753359037, -69.84537459892903, -73.81562785829651, -148.23580687485645, -227.71842861064192, -229.9101089174337]
# danger rate:  [2.13, 0.88, 0.77, 1.88, 3.82, 3.82]
# crash rate:  [0.58, 0.33, 0.5, 1.24, 2.09, 2.09]
# level-1-danger_rate:  [0.23, 0.09, 0.05, 0.14, 0.25, 0.25]
# level-2-danger_rate:  [0.05, 0.03, 0.05, 0.12, 0.2, 0.2]
# collison rate:  [0.0, 0.0, 0.02, 0.09, 0.14, 0.14]
# success rate:  [0.99, 0.99, 0.9, 0.6, 0.08, 0.05]
# sucess len:  [55.656565656565654, 62.43434343434343, 67.5, 90.1, 66.625, 73.4]
