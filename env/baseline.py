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
    collision_num = 0
    ep_len_list = []
    success_num = 0
    for i in range(num_eps):
        ep_eval = episode_generator(pi, env, is_gui=is_gui, ttc=ttc, gap=gap, sumoseed=sumoseed, randomseed=randomseed)

        ret_eval += ep_eval['ep_ret']
        ret_det_eval += ep_eval['ep_rets_detail']
        danger_num += ep_eval['ep_num_danger']
        crash_num += ep_eval['ep_num_crash']
        collision_num += ep_eval['ep_is_collision']
        success_num += int(ep_eval['ep_is_success'])
        if ep_eval['ep_is_success']:
            ep_len_list.append(ep_eval['ep_len'])
        sumoseed += 1
        randomseed += 1
        # f.write('%s,%s,%s,%s,' % (sumoseed, randomseed, ep_eval['ep_is_success'], ep_eval['ep_num_danger']))
        # for i_obs in range(21):
        #     f.write(str(ep_eval['ep_obs'][-1][i_obs]) + ',')
        # f.write('\n')
    env.close()
    ret_eval /= float(num_eps)
    ret_det_eval /= float(num_eps)
    danger_rate = danger_num / num_eps
    crash_rate = crash_num / num_eps
    coll_rate = collision_num / num_eps
    success_rate = success_num / float(num_eps)
    success_len = np.mean(ep_len_list)
    print('reward_detail: ', ret_det_eval)
    print('reward: ', ret_eval,
          '\ndanger_rate: ', danger_rate,
          '\ncrash_rate: ', crash_rate,
          '\ncollision_rate: ', coll_rate,
          '\nsuccess_rate: ', success_rate,
          '\nsuccess_len: ', success_len)
    return ret_eval, danger_rate, crash_rate, coll_rate, success_rate, success_len


NUM_EPS = 100
IS_GUI = False


# f = open('../data/baseline_evaluation/testseed2.csv', 'w+')
# safety_gap = 2
constraints_list = [3.0]  # [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
ttcs = [0.5, 1, 2, 3, 4]
# ttcs = [2]
gap = 0

reward_list = []
danger_rate_list = []
crash_rate_list = []
coll_rate_list = []
succ_rate_list = []
succ_len_list = []
for ttc in ttcs:
    ret_eval, danger_rate, crash_rate, coll_rate, success_rate, success_len = evaluate_baseline(NUM_EPS, ttc, gap, IS_GUI)
    reward_list.append(ret_eval)
    danger_rate_list.append(danger_rate)
    crash_rate_list.append(crash_rate)
    coll_rate_list.append(coll_rate)
    succ_rate_list.append(success_rate)
    succ_len_list.append(success_len)

print('reward: ', reward_list)
print('danger rate: ', danger_rate_list)
print('crash rate: ', crash_rate_list)
print('collison rate: ', coll_rate_list)
print('success rate: ', succ_rate_list)
print('sucess len: ', succ_len_list)
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