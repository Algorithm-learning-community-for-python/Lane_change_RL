import gym
import sys
import os
import random
import LaneChangeEnv as lcEnv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


def normalSim():
    # randomly make lane change or abort/keep
    if env.timestep < 130:
        obs, rwd, done, info = env.step(2)
    else:
        obs, rwd, done, info = env.step(1)

    print(env.timestep)
    if done is True and info['resetFlag'] == 1:
        env.reset(egoid='lane2.1', tlane=0, tfc=1, is_gui=True)


def laneChange(low, high, origLane, tgtLane, rd):
    for vehID in env.veh_dict.keys():
        if int(vehID.split('.')[1]) % 2 == 0 and vehID.split('.')[0] == 'lane'+str(origLane):
        #if vehID == 'lane1.0':
            veh = env.veh_dict[vehID]
            if veh.lcPos is None:
                veh.lcPos = random.uniform(low, high)
            veh.targetLane = tgtLane

            if veh.dis2entrance < veh.lcPos and abs(veh.pos_lat - (0.5+veh.targetLane)*rd.laneWidth) > 0.01:

                if abs(veh.dis2entrance > 20):
                    veh.changeLane(False, veh.targetLane, rd)
                    traci.vehicle.setColor(veh.veh_id, (255, 69, 0))
                    '''
                    if veh.changeTimes < 15:
                        veh.changeLane(True, veh.targetLane, rd)
                        traci.vehicle.setColor(veh.veh_id, (255, 69, 0))
                    elif 15 < veh.changeTimes < 30:
                        #traci.vehicle.changeSublane(veh.veh_id, 0.0)
                        veh.changeLane(True, veh.origLane, rd)
                    else:
                        traci.vehicle.changeSublane(veh.veh_id, 0.0)
                    veh.changeTimes += 1
                    '''
                    # todo check only 1 cmd

                else:
                    if not veh.laneIndex == 0:
                        traci.vehicle.setRouteID(veh.veh_id, rd.highwayKeepRouteID)
                # todo order of cmd, step, write

            '''    
            if (abs(veh.pos_lat - (0.5+tgtLane)*3.2) < 0.01 or veh.dis2entrance < 1.0) \
               and veh.change_times == 1:

                #fe.write('%s, %s\n' % (vehID, veh.dis2entrance))
                fe.flush()
                veh.change_times += 1
'''

def writeInfo(testid='lane1.0'):
    assert testid in vehID_tuple_all, 'testid no in env'
    testLane = traci.vehicle.getLaneIndex(testid)
    ve = self.veh_dict_list[testLane][testid]

    f = open(file_name, 'a')

    data = '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (ve.veh_id,
                                                         bin(traci.vehicle.getLaneChangeState(testid, 1)[0]),
                                                         bin(traci.vehicle.getLaneChangeState(testid, 1)[1]),
                                                         bin(traci.vehicle.getLaneChangeMode(testid)),
                                                         ve.pos_x, ve.pos_y, ve.pos_lat, ve.speed, ve.acce,
                                                         ve.latSpeed, ve.yawAngle, ve.dis2entrance)
    if ve.leader is not None:
        data += ', %s, %s' % (ve.leaderDis, ve.speed - traci.vehicle.getSpeed(ve.leaderID))
    data += '\n'

    #f.write(data)
    f.flush()
    f.close()


if __name__ == '__main__':
    #fs = open('data/data6.csv', 'a')
    #fe = open('data/data7.csv', 'a')
    # fs.write('egoid, startDis\n')
    # fe.write('egoid, endDis\n')
    '''
    f.write('egoid, lcStateM, lcStateR, lcMode, posX, posY, posLat, speed, acce, latSpeed, yaw_angle, dis2entrance, '
            'leader_dis, leader_delta_v'
            '\n')'''
    f = open('data/data15.csv', 'a')
    f.write('egoid, lanePos, latPos, speed, latSpeed, lcState, latAcce, action\n')

    env = lcEnv.LaneChangeEnv()
    # env.reset(egoid='lane2.1', tlane=0, tfc=1, is_gui=True)
    env.reset(None, tfc=2, sumoseed=3, randomseed=3)
    testid = 'lane1.0'
    for step in range(10000):
        # todo random --completed ?
        # todo emergency braking
        laneChange(300, 350, 1, 0, env.rd)
        env.preStep()

        if testid in env.vehID_tuple_all:
            veh = env.veh_dict[testid]
            # todo complete action extraction
            if veh.latAcce < 0 or veh.latSpeed < 0 and abs(veh.latAcce) < 0.1:
                action = 1
            else:
                action = 0

            f.write('%s, %s, %s, %s, %s, %s, %s, %s\n' % (veh.veh_id, veh.lanePos, veh.pos_lat, veh.speed, veh.latSpeed,
                                                  bin(traci.vehicle.getLaneChangeState(veh.veh_id, -1)[1]), veh.latAcce, action))
            f.flush()
