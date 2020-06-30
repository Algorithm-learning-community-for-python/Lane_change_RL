import os, sys, random, datetime, gym, math
from gym import spaces
import numpy as np
from env.Road import Road
from env.Vehicle import Vehicle
from env.Ego import Ego
from env.IDM import IDM
import traci


class LaneChangeEnv(gym.Env):
    def __init__(self, gui=False, max_timesteps=250, label='default', is_train=True):
        self.max_timesteps = max_timesteps
        self.is_train = is_train
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21,))

        self.sumoBinary = os.environ["SUMO_HOME"] + '/bin/sumo'
        self.sumoCmd_base = ['--lateral-resolution', str(0.8),  # using 'Sublane-Model'
                             '--step-length', str(0.1),
                             '--default.action-step-length', str(0.1),
                             '--no-warnings', str(True),
                             '--no-step-log', str(True)]
        if gui:
            self.sumoBinary += '-gui'
            self.sumoCmd_base += ['--quit-on-end', str(True), '--start', str(True)]
        sumoCmd = [self.sumoBinary] + ['-c', '../map/ramp3/mapDense.sumo.cfg'] + self.sumoCmd_base
        traci.start(sumoCmd, label=label)
        self.rd = Road()

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
        """
        # todo difference
        if veh is not None:
            self.observation[name*4+0+1] = (veh.pos_longi - self.ego.pos_longi) / 239.8
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
        self.observation[1] = (self.ego.speed - 15) / 30.0
        self.observation[2] = (self.ego.pos_lat - 2.4) / 2.4
        self.observation[3] = self.ego.acce
        self.observation[4] = self.ego.speed_lat

        self._updateObservationSingle(1, self.ego.orig_leader)
        self._updateObservationSingle(2, self.ego.orig_follower)
        self._updateObservationSingle(3, self.ego.trgt_leader)
        self._updateObservationSingle(4, self.ego.trgt_follower)

    def updateReward(self, act_lat, act_longi):
        names = ["comfort", "efficiency", "time", "speed", "safety"]
        w_comf = 0.2  # 0.2 0
        w_effi = 1
        w_time = 0.0  # 0.01 0
        w_speed = 0.1
        w_safety = 1  # 1.5 0.15
        weights = np.array([w_comf, w_effi, w_time, w_speed, w_safety])
        weights = weights / np.sum(weights)

        # reward for comfort
        r_comf = -1 + np.exp(- (0.1*abs(self.ego.acce) + abs(self.ego.delta_acce)))

        # reward for efficiency
        r_effi = -1 + np.exp(-self.ego.dis2tgtLane)

        # reward for elapsed time
        r_time = 0

        # reward for desired speed
        r_speed = -1 + np.exp(-abs(self.ego.speed - self.ego.speedLimit))

        # reward for safety
        _, _, _, _, is_danger = self.is_to_crash(act_longi, act_lat, longi_safety_dis=10, lat_safety_dis=0.8)
        _, _, _, _, self.is_crash = self.is_to_crash(act_longi, act_lat)
        # print("danger: ", is_danger, "crash: ", self.is_crash)
        # print(abs(self.ego.pos_longi - self.ego.trgt_leader.pos_longi))
        if self.is_crash:
            r_safety = self.timestep - 250  # -15
            self.num_crash += 1
        elif is_danger:

            r_safety = -1
            self.num_danger += 1
        else:
            r_safety = 0
            if self.ego.trgt_leader and self.ego.trgt_follower:
                dis2leader = abs(self.ego.pos_longi - self.ego.trgt_leader.pos_longi)
                dis2follower = abs(self.ego.pos_longi - self.ego.trgt_follower.pos_longi)
                # print(dis2leader, dis2follower, self.ego.speedLimit, self.ego.speed)
                dis_diff = abs(dis2leader - dis2follower)
                # gap = abs(self.ego.trgt_leader.pos_longi - self.ego.trgt_follower.pos_longi)
                r_safety += -1 + np.exp(-0.1 * dis_diff)
            elif self.ego.trgt_leader:
                dis2leader = abs(self.ego.pos_longi - self.ego.trgt_leader.pos_longi)
                ttc_l = dis2leader / max(self.ego.speed, 0.1)
                r_safety += -1 + np.tanh(ttc_l/2.5)
            elif self.ego.trgt_follower:
                dis2follower = abs(self.ego.pos_longi - self.ego.trgt_follower.pos_longi)
                ttc_f = dis2follower / max(self.ego.trgt_follower.speed, 0.1)
                r_safety += -1 + np.tanh(ttc_f/2.5)

            if self.ego.curr_leader and self.ego.trgt_leader:
                dis2currleader = abs(self.ego.pos_longi - self.ego.curr_leader.pos_longi)
                dis2trgtleader = abs(self.ego.pos_longi - self.ego.trgt_leader.pos_longi)
                ttc_cl = dis2currleader / max(self.ego.speed, 0.1)
                ttc_tl = dis2trgtleader / max(self.ego.speed, 0.1)
                r_safety = 0.7*r_safety + 0.3*(-1 + np.tanh(min(ttc_cl, ttc_tl)/2.5))
            elif self.ego.curr_leader:
                dis2currleader = abs(self.ego.pos_longi - self.ego.curr_leader.pos_longi)
                ttc_cl = dis2currleader / max(self.ego.speed, 0.1)
                r_safety = 0.7*r_safety + 0.3*(-1 + np.tanh(ttc_cl /2.5))
            elif self.ego.trgt_leader:
                dis2trgtleader = abs(self.ego.pos_longi - self.ego.trgt_leader.pos_longi)
                ttc_tl = dis2trgtleader / max(self.ego.speed, 0.1)
                r_safety = 0.7 * r_safety + 0.3 * (-1 + np.tanh(ttc_tl / 2.5))
            else:
                r_safety = 0.7 * r_safety - 0.15

        rewards = np.array([r_comf, r_effi, r_time, r_speed, r_safety])
        total_reward = np.sum(weights * rewards)
        reward_dict = dict(zip(names, weights * rewards))
        return total_reward, reward_dict

    def is_done(self):
        done = False
        if self.is_success and self.success_timer*self.dt > 1:
            done = True
            self.is_final_success = True
        if self.ego.dis2entrance < 5.0:
            done = True
        if self.is_crash and self.is_train:
            done = True
        collision_num = traci.simulation.getCollidingVehiclesNumber()
        if collision_num > 0:
            print("collision ids: ", traci.simulation.getCollidingVehiclesIDList())
            print("egoid: ", self.ego.veh_id)
            print("trgt_follower: ", self.ego.trgt_follower.veh_id if self.ego.trgt_follower else None)
            print("trgt_leader: ", self.ego.trgt_leader.veh_id if self.ego.trgt_leader else None)
            self.is_collision = True
            done = True
        if self.timestep > self.max_timesteps:
            done = True
        if self.egoID not in self.vehID_tuple_all:
            done = True
        return done

    def is_to_crash(self, action_longi, action_lat, longi_safety_dis=5, lat_safety_dis=0.3):
        longi_danger_list = [False, False, False, False]
        lat_danger_list = [False, False, False, False]

        is_danger = False
        after_which = None
        before_which = None
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
           (action_lat == 0 and (lat_danger_list[2] or lat_danger_list[3])):
            action_lat = 1
            is_danger = True
        # if action_lat == 0 and (lat_danger_list[2] or lat_danger_list[3]):
        #     action_lat = 1
        #     is_danger = True
        if (longi_danger_list[0] or longi_danger_list[2]):  # action_longi == 2 and
            action_longi = -1
            is_danger = True
            if longi_danger_list[2]:
                after_which = 'orig'
                if longi_danger_list[0]:
                    after_which = 'both'
            elif longi_danger_list[0]:
                after_which = 'tgt'

        if (longi_danger_list[1] or longi_danger_list[3]):  # action_longi == 2 and
            action_longi = -2
            is_danger = True
            if longi_danger_list[3]:
                before_which = 'orig'
                if longi_danger_list[1]:
                    before_which = 'both'
            elif longi_danger_list[1]:
                before_which = 'tgt'

        return action_longi, action_lat, after_which, before_which, is_danger

    def preStep(self):
        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)
        warmupID = traci.edge.getLastStepVehicleIDs(self.rd.warmupEdge)
        [traci.vehicle.setLaneChangeMode(id, 0) for id in warmupID]

    def step(self, action):
        action_lateral = action // 3  # 0, 1
        assert action_lateral in [0, 1]
        action_lateral += 1
        action_longi = action % 3  # 0, 1, 2
        # print("longi: ", action_longi, "lateral: ", action_lateral)
        assert action is not None, 'action is None'
        assert self.egoID in self.vehID_tuple_all, 'vehicle not in env'

        if self.is_train:
            action_longi, action_lateral, after_which, before_which, self.is_crash = self.is_to_crash(action_longi, action_lateral)

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

        # longitudinal safety constraint
        if action_longi == -1:
            if after_which == 'orig':
                vNext = self.ego.orig_leader.speed
            elif after_which == 'tgt':
                vNext = self.ego.trgt_leader.speed
            else:
                assert after_which == 'both'
                vNext = min(self.ego.trgt_leader.speed, self.ego.orig_leader.speed)
        elif action_longi == -2:
            if before_which == 'orig':
                vNext = self.ego.orig_follower.speed
            elif before_which == 'tgt':
                vNext = self.ego.trgt_follower.speed
            else:
                assert before_which == 'both'
                vNext = max(self.ego.trgt_follower.speed, self.ego.orig_follower.speed)
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
        if self.ego.trgt_follower and self.polite_flag:
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
        warmupID = traci.edge.getLastStepVehicleIDs(self.rd.warmupEdge)
        [traci.vehicle.setLaneChangeMode(id, 0) for id in warmupID]

        self.updateObservation()
        reward, reward_dict = self.updateReward(action_lateral, action_longi)
        done = self.is_done()

        self.info['reward_dict'] = reward_dict
        self.info["num_danger"] = self.num_danger
        self.info["num_crash"] = self.num_crash
        if self.is_final_success:
            self.info["is_success"] = 1
        else:
            self.info["is_success"] = 0
        if self.is_collision:
            self.info["is_collision"] = 1
        else:
            self.info["is_collision"] = 0
        self.timestep += 1
        return self.observation, reward, done, self.info

    def seed(self, seed=None):
        if not seed:
            random.seed(datetime.datetime.now().microsecond)
            np.random.seed(datetime.datetime.now().microsecond)
        else:
            random.seed(seed)
            np.random.seed(seed)

    def reset(self, egoid=None, tlane=0, tfc=None, is_gui=False, sumoseed=None, randomseed=None):
        egoid = 'lane1.' + str(random.randint(1, 6)) if not egoid else egoid
        tfc = np.random.choice([0, 1, 2]) if tfc is None else tfc
        if tfc == 0:
            cfg = '../map/ramp3/mapDenseSlow.sumo.cfg'
        elif tfc == 1:
            cfg = '../map/ramp3/mapDense.sumo.cfg'
        else:
            assert tfc == 2
            cfg = '../map/ramp3/mapDenseFast.sumo.cfg'
        sumoCmd_load = ['-c', cfg] + self.sumoCmd_base
        self.seed(randomseed)
        if not sumoseed:
            sumoCmd_load += ['--random']  # initialize with current system time
        else:
            sumoCmd_load += ['--seed', str(sumoseed)]  # initialize with given seed

        traci.load(sumoCmd_load)

        self.timestep = 0
        self.dt = traci.simulation.getDeltaT()

        self.veh_dict = {}
        self.vehID_tuple_all = ()

        self.egoID = egoid
        self.ego = None

        self.is_success = False
        self.is_final_success = False
        self.success_timer = 0
        self.is_collision = False
        self.is_crash = False
        self.polite_flag = np.random.choice([0, 1])
        self.num_danger = 0
        self.num_crash = 0

        self.info = {}
        self.observation = np.empty(21)

        # step simulation until ego appears
        if self.egoID is not None:
            while self.egoID not in self.veh_dict.keys():
                self.preStep()
                for id in traci.edge.getLastStepVehicleIDs(self.rd.warmupEdge):
                    traci.vehicle.setLaneChangeMode(id, 0)
                if self.timestep > 500:
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
