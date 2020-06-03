import os, sys, random, datetime, gym, math
from gym import spaces
import numpy as np
from env.Road import Road
from env.Vehicle import Vehicle
from env.Ego import Ego
from env.IDM import IDM
# add sumo/tools into python environment

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
######################################################################
# lane change environment


class LaneChangeEnv(gym.Env):
    def __init__(self, id=None, traffic=1, gui=False, seed=None):
        # todo check traffic flow density
        if traffic == 0:
            # average 9 vehicles
            self.cfg = '../map/ramp3/mapFree.sumo.cfg'
        elif traffic == 2:
            # average 19 vehicles
            self.cfg = '../map/ramp3/mapDense.sumo.cfg'
        else:
            # average 14 vehicles
            self.cfg = '../map/ramp3/map.sumo.cfg'

        # arguments must be string, if float/int, must be converted to str(float/int), instead of '3.0'
        self.sumoBinary = "/usr/local/Cellar/sumo/1.6.0/bin/sumo"
        self.sumoCmd = ['-c', self.cfg,
                        # '--lanechange.duration', str(3),     # using 'Simple Continuous lane-change model'
                        '--lateral-resolution', str(0.8),  # using 'Sublane-Model'
                        '--step-length', str(0.1),
                        '--default.action-step-length', str(0.1),
                        '--no-warnings', str(True)]
        # randomness
        if seed is None:
            self.sumoCmd += ['--random']  # initialize with current system time
        else:

            self.sumoCmd += ['--seed', str(seed)]  # initialize with given seed
        # gui
        if gui is True:
            self.sumoBinary += '-gui'
            self.sumoCmd = [self.sumoBinary] + self.sumoCmd + ['--quit-on-end', str(True),
                                                               '--start', str(True)]
        else:
            self.sumoCmd = [self.sumoBinary] + self.sumoCmd
        # start Traci
        traci.start(self.sumoCmd)

        self.rd = Road()
        self.timestep = 0
        self.dt = traci.simulation.getDeltaT()

        self.veh_dict = {}
        self.vehID_tuple_all = ()

        self.egoID = id
        self.ego = None

        self.is_success = False
        self.is_final_success = False
        self.success_timer = 0

        self.info = {}
        self.observation = np.empty(21)
        self.is_collision = False
        self.num_danger = 0
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21, ))

    def update_veh_dict(self, veh_id_tuple):
        for veh_id in veh_id_tuple:
            if veh_id not in self.veh_dict.keys():
                if veh_id == self.egoID:
                    self.veh_dict[veh_id] = Ego(veh_id, self.rd)
                else:
                    self.veh_dict[veh_id] = Vehicle(veh_id, self.rd)

        for veh_id in list(self.veh_dict.keys()):
            if veh_id not in veh_id_tuple:
                self.veh_dict.pop(veh_id)

        for veh_id in list(self.veh_dict.keys()):
            self.veh_dict[veh_id].update_info(self.rd, self.veh_dict)

    def _updateObservationSingle(self, name, veh):
        """
        :param name: 0:ego; 1:leader; 2:target leader; 3:target follower
        :param id: vehicle id corresponding to name
        :return:
        """
        # todo difference
        if veh is not None:
            self.observation[name*4+0+1] = (veh.pos_longi - self.ego.pos_longi) / 239.8  # todo 239.8
            self.observation[name*4+1+1] = (veh.speed - self.ego.speed) / 30.0
            self.observation[name*4+2+1] = (veh.pos_lat - self.ego.pos_lat) / 3.2
            self.observation[name*4+3+1] = veh.acce
        else:
            assert name != self.egoID
            self.observation[name*4+0+1] = (800 - self.ego.pos_longi) / 239.8
            self.observation[name*4+1+1] = 0
            if name == 1 or name == 2:
                self.observation[name*4+2+1] = (4.8 - self.ego.pos_lat) / 3.2
            else:
                self.observation[name*4+2+1] = (1.6 - self.ego.pos_lat) / 3.2
            self.observation[name*4+3+1] = 0

    def updateObservation(self):
        self.observation[0] = (self.ego.pos_longi - 239.8) / 239.8
        self.observation[1] = self.ego.speed / 30.0
        self.observation[2] = (self.ego.pos_lat - 2.4) / 2.4
        self.observation[3] = self.ego.acce
        self.observation[4] = self.ego.speed_lat

        #self._updateObservationSingle(0, self.ego)
        self._updateObservationSingle(1, self.ego.orig_leader)
        self._updateObservationSingle(2, self.ego.orig_follower)
        self._updateObservationSingle(3, self.ego.trgt_leader)
        self._updateObservationSingle(4, self.ego.trgt_follower)
        # self.observation = np.array(self.observation).flatten()
        # print(self.observation.shape)

    def updateReward2(self, act_lat, act_longi):
        names = ["comfort", "efficiency", "time", "speed", "safety"]
        w_comf = 0.3
        w_effi = 1
        w_time = 0.1
        w_speed = 0.02
        w_safety = 1.5
        weights = np.array([w_comf, w_effi, w_time, w_speed, w_safety])
        # reward for comfort
        if act_longi == 1:
            r_comf = 0
        else:
            r_comf = -1
        # reward for efficiency
        r_effi = - self.ego.dis2tgtLane
        if act_lat == 2:
            r_effi = 1
        else:
            r_effi = -1
        if self.is_final_success:
            r_effi += 50
        # reward for elapsed time
        r_time = -1
        # reward for desired speed
        r_speed = -abs(self.ego.speed - self.ego.speedLimit)
        # reward for safety
        _, _, _, is_danger = self.is_to_crash(act_longi, act_lat, longi_safety_dis=15, lat_safety_dis=1)
        if is_danger:
            r_safety = -15
        else:
            r_safety = 0


        rewards = np.array([r_comf, r_effi, r_time, r_speed, r_safety])
        total_reward = np.sum(weights * rewards)
        reward_dict = dict(zip(names, weights * rewards))
        return total_reward, reward_dict

    def updateReward(self, act_lat, act_longi):
        names = ["comfort", "efficiency", "time", "speed", "safety"]
        w_comf = 0  # 0.2 0
        w_effi = 1
        w_time = 0.0  # 0.01 0
        w_speed = 0.01
        w_safety = 1.5  # 1.5 0.15
        weights = np.array([w_comf, w_effi, w_time, w_speed, w_safety])
        # reward for comfort
        if act_longi == 1:
            r_comf = 0
        else:
            r_comf = -1
        # reward for efficiency
        r_effi = - self.ego.dis2tgtLane
        # if act_lat == 2:
        #     r_effi = 1
        # else:
        #     r_effi = -1
        if self.is_final_success:
            r_effi += 50
        # reward for elapsed time
        r_time = -self.timestep
        # reward for desired speed
        r_speed = -abs(self.ego.speed - self.ego.speedLimit)
        # reward for safety
        _, _, _, is_danger = self.is_to_crash(act_longi, act_lat, longi_safety_dis=15, lat_safety_dis=1)
        if is_danger:
            r_safety = -15
        else:
            r_safety = 0
        if self.ego.trgt_leader and self.ego.trgt_follower:
            dis2leader = abs(self.ego.pos_longi - self.ego.trgt_leader.pos_longi)
            dis2follower = abs(self.ego.pos_longi - self.ego.trgt_follower.pos_longi)
            dis_diff = abs(dis2leader - dis2follower)
            gap = abs(self.ego.trgt_leader.pos_longi - self.ego.trgt_follower.pos_longi)
            # print(gap)
            r_safety += -1 + np.exp(-0.08 * dis_diff)
            # print(0.05 * dis_diff)
            r_safety += -1 + np.tanh(gap/70)
        elif self.ego.trgt_leader:
            dis2leader = abs(self.ego.pos_longi - self.ego.trgt_leader.pos_longi)
            r_safety += -1 + np.tanh(dis2leader/40)
            r_safety += -1
        elif self.ego.trgt_follower:
            dis2follower = abs(self.ego.pos_longi - self.ego.trgt_follower.pos_longi)
            r_safety += -1 + np.tanh(dis2follower/40)
            r_safety += -1
        else:
            r_safety += -2

        rewards = np.array([r_comf, r_effi, r_time, r_speed, r_safety])
        total_reward = np.sum(weights * rewards)
        reward_dict = dict(zip(names, weights * rewards))
        return total_reward, reward_dict

    def is_done(self):
        done = False
        if self.is_success and self.success_timer*self.dt > 0.5:
            done = True
            self.is_final_success = True
        if self.ego.dis2entrance < 5.0:
            done = True
        collision_num = traci.simulation.getCollidingVehiclesNumber()
        if collision_num > 0:
            print("collision ids: ", traci.simulation.getCollidingVehiclesIDList())
            print("egoid: ", self.ego.veh_id)
            done = True
            self.is_collision = True
            done = True
        return done

    def is_to_crash(self, action_longi, action_lat, longi_safety_dis=5, lat_safety_dis=0.2):
        longi_danger_list = [False, False, False, False]
        lat_danger_list = [False, False, False, False]

        is_danger = False
        after_which = None
        count = 0
        for veh in [self.ego.trgt_leader, self.ego.trgt_follower, self.ego.orig_leader, self.ego.orig_follower]:
            if veh:
                sum_width = (veh.width + self.ego.width) / 2
                if sum_width < abs(veh.pos_lat - self.ego.pos_lat) <= sum_width + lat_safety_dis and \
                        abs(veh.pos_longi - self.ego.pos_longi) < (veh.length + self.ego.length) / 2 + longi_safety_dis:
                    lat_danger_list[count] = True
                if sum_width >= abs(veh.pos_lat - self.ego.pos_lat) and \
                        abs(veh.pos_longi - self.ego.pos_longi) < (veh.length + self.ego.length) / 2 + longi_safety_dis:
                    longi_danger_list[count] = True
            count += 1

        if (action_lat == 2 and (lat_danger_list[0] or lat_danger_list[1])) or \
           (action_lat == 1 and (lat_danger_list[2] or lat_danger_list[3])):
            action_lat = 1
            is_danger = True
        if action_lat == 0 and (lat_danger_list[2] or lat_danger_list[3]):
            action_lat = 1
            is_danger = True
        if (longi_danger_list[0] or longi_danger_list[2]):  # action_longi == 2 and
            action_longi = -1
            is_danger = True
            if longi_danger_list[2]:
                after_which = 'orig'
                if longi_danger_list[0]:
                    after_which = 'both'
            elif longi_danger_list[0]:
                after_which = 'tgt'

        return action_longi, action_lat, after_which, is_danger

    def preStep(self):
        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)

    def step(self, action):
        action_lateral = action // 3
        action_longi = action % 3
        # print("longi: ", action_longi, "lateral: ", action_lateral)
        assert action is not None, 'action is None'
        assert self.egoID in self.vehID_tuple_all, 'vehicle not in env'

        self.timestep += 1

        # lateral control-------------------------
        # episode in progress; 0:change back to original line; 1:lane change to target lane; 2:keep current
        # lane change to target lane

        action_longi, action_lateral, after_which, is_danger = self.is_to_crash(action_longi, action_lateral)
        if is_danger: self.num_danger += 1

        # lateral control
        if action_lateral == 2:
            # perform lane change
            self.is_success = self.ego.changeLane(True, self.ego.trgt_laneIndex, self.rd)
        if action_lateral == 0:
            # abort lane change, change back to ego's original lane
            self.is_success = self.ego.changeLane(True, self.ego.orig_laneIndex, self.rd)
        if action_lateral == 1:
            # keep current lateral position
            self.is_success = self.ego.changeLane(True, -1, self.rd)

        if self.is_success:
            self.success_timer += 1
        if True:
            # longitudinal control
            if action_longi == -1:
                # print("danger")
                if after_which == 'orig':
                    vNext = self.ego.orig_leader.speed
                elif after_which == 'tgt':
                    vNext = self.ego.trgt_leader.speed
                else:
                    assert after_which == 'both'
                    vNext = min(self.ego.trgt_leader.speed, self.ego.orig_leader.speed)
            else:
                if action_longi == 2:
                    acceNext = 1.5  # accelerate
                elif action_longi == 1:
                    acceNext = 0
                else:
                    acceNext = -1.5
                vNext = max(self.ego.speed + acceNext * 0.1, 0.1)
            traci.vehicle.setSpeed(self.egoID, vNext)

        # target follower control
        if self.ego.trgt_follower and np.random.randint(0, 2):
            # print(self.ego.trgt_follower.veh_id)
            # print(self.ego.trgt_leader.veh_id)
            # print(self.ego.veh_id)
            idm = IDM(v0=self.ego.trgt_follower.speedLimit)
            if self.ego.trgt_leader:
                accNext = idm.calc_acce(self.ego.trgt_follower.speed,
                                        max(self.ego.trgt_leader.pos_longi - self.ego.trgt_follower.pos_longi, 0.1),
                                        self.ego.trgt_leader.speed)
            else:
                accNext = 0
            vNext = max(self.ego.trgt_follower.speed + accNext * 0.1, 0.1)
            traci.vehicle.setSpeed(self.ego.trgt_follower.veh_id, vNext)

        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)
        # check if episode ends
        done = self.is_done()

        self.updateObservation()
        reward, reward_dict = self.updateReward(action_lateral, action_longi)

        self.info['reward_dict'] = reward_dict
        self.info["num_danger"] = self.num_danger
        if self.is_final_success:
            self.info["is_success"] = 1
        else:
            self.info["is_success"] = 0
        if self.is_collision:
            self.info["is_collision"] = 1
        else:
            self.info["is_collision"] = 0
        return self.observation, reward, done, self.info

    def seed(self, seed=None):
        if seed is None:
            self.randomseed = datetime.datetime.now().microsecond
        else:
            self.randomseed = seed
        random.seed(self.randomseed)

    def reset(self, egoid=None, tlane=0, tfc=2, is_gui=False, sumoseed=None, randomseed=None):
        """
        reset env
        :param id: ego vehicle id
        :param tfc: int. 0:light; 1:medium; 2:dense
        :return: initial observation
        """
        self.seed(randomseed)
        if egoid is None:
            egoid = 'lane1.' + str(random.randint(1, 6))
        traci.close()
        self.__init__(id=egoid, traffic=tfc, gui=is_gui, seed=sumoseed)
        # continue step until ego appears in env
        if self.egoID is not None:
            while self.egoID not in self.veh_dict.keys():
                # must ensure safety in preStpe
                self.preStep()
                for id in traci.edge.getLastStepVehicleIDs(self.rd.warmupEdge):
                    traci.vehicle.setLaneChangeMode(id, 0)
                if self.timestep > 1000:
                    raise Exception('cannot find ego after 1000 timesteps')

            assert self.egoID in self.vehID_tuple_all, "cannot start training while ego is not in env"

            self.ego = self.veh_dict[self.egoID]
            self.ego.setTrgtLane(tlane)
            self.ego.update_info(self.rd, self.veh_dict)

            self.updateObservation()
            return self.observation
        return

    def close(self):
        traci.close()
