import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
from numpy.core.numeric import base_repr
import cityflow
import numpy as np
import os
import json as json
class CityFlow_1x1_LowTraffic(gym.Env):
    """
    Description:
        A single intersection with low traffic.
        8 roads, 1 intersection (plus 4 virtual intersections).
    State:
        Type: array[16]
        The number of vehicless and waiting vehicles on each lane.
    Actions:
        Type: Discrete(9)
        index of one of 9 light phases.
        Note:
            Below is a snippet from "roadnet.json" file which defines lightphases for "intersection_1_1".
            "lightphases": [
              {"time": 5, "availableRoadLinks": []},
              {"time": 30, "availableRoadLinks": [ 0, 4 ] },
              {"time": 30, "availableRoadLinks": [ 2, 7 ] },
              {"time": 30, "availableRoadLinks": [ 1, 5 ] },
              {"time": 30,"availableRoadLinks": [3,6]},
              {"time": 30,"availableRoadLinks": [0,1]},
              {"time": 30,"availableRoadLinks": [4,5]},
              {"time": 30,"availableRoadLinks": [2,3]},
              {"time": 30,"availableRoadLinks": [6,7]}]
    Reward:
        The total amount of time -- in seconds -- that all the vehicles in the intersection
        waitied for.
        Todo: as a way to ensure fairness -- i.e. not a single lane gets green lights for too long --
        instead of simply summing up the waiting time, we could weigh the waiting time of each car by how
        much it had waited so far.
    """

    metadata = {'render.modes':['human']}
    def __init__(self):
        #super(CityFlow_1x1_LowTraffic, self).__init__()
        # hardcoded settings from "config.json" file
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1x1_config")
        self.cityflow = cityflow.Engine(os.path.join(self.config_dir, "config.json"), thread_num=1)
        self.intersection_ids = \
            ["intersection_1_1",
             "intersection_2_1",
             "intersection_3_1",
             "intersection_4_1",
             "intersection_1_2",
             "intersection_2_2",
             "intersection_3_2",
             "intersection_4_2",
             "intersection_1_3",
             "intersection_2_3",
             "intersection_3_3",
             "intersection_4_3",
             "intersection_1_4",
             "intersection_2_4",
             "intersection_3_4",
             "intersection_4_4"]

        self.sec_per_step = 1.0
        self.steps_per_episode = 100
        self.current_step = 0
        self.is_done = False
        self.reward_range = (-float('inf'), float('inf'))
        with open("/content/drive/MyDrive/gym_cityflow-master/gym_cityflow/envs/1x1_config/roadnet_4X4.json", "r") as read_content:
            self.roadnet = json.load(read_content)
        #self.roadnet['intersections'][3]['trafficLight']['lightphases']
        #print('HERE: ',self.roadnet['intersections'])
        self.lightphases = []
        self.road_phases = []
        for j in range(len(self.intersection_ids)):
          for i in range(len(self.roadnet['intersections'])):
            if self.roadnet['intersections'][i]['id'] == self.intersection_id[j]:
              idty = i
              self.lightphases.append(self.roadnet['intersections'][i]['trafficLight']['lightphases'])
              self.road_phases.append(self.roadnet['intersections'][idty]['roadLinks'])
        self.start_lane_ids = \
            ["road_0_2_0_0",
             "road_0_2_0_1",
             "road_1_1_1_0",
             "road_1_1_1_1",
             "road_2_2_2_0",
             "road_2_2_2_1",
             "road_1_3_3_0",
             "road_1_3_3_1",
             "road_1_2_0_0",
             "road_1_2_0_1",
             "road_2_1_1_0",
             "road_2_1_1_1",
             "road_3_2_2_0",
             "road_3_2_2_1",
             "road_2_3_3_0",
             "road_2_3_3_1",
             "road_2_2_0_0",
             "road_2_2_0_1",
             "road_3_1_1_0",
             "road_3_1_1_1",
             "road_4_2_2_0",
             "road_4_2_2_1",
             "road_3_3_3_0",
             "road_3_3_3_1",
             "road_0_1_0_0",
             "road_0_1_0_1",
             "road_1_0_1_0",
             "road_1_0_1_1",
             "road_2_1_2_0",
             "road_2_1_2_1",
             "road_1_2_3_0",
             "road_1_2_3_1",
             "road_1_1_0_0",
             "road_1_1_0_1",
             "road_2_0_1_0",
             "road_2_0_1_1",
             "road_3_1_2_0",
             "road_3_1_2_1",
             "road_2_2_3_0",
             "road_2_2_3_1",
             "road_2_1_0_0",
             "road_2_1_0_1",
             "road_3_0_1_0",
             "road_3_0_1_1",
             "road_4_1_2_0",
             "road_4_1_2_1",
             "road_3_2_3_0",
             "road_3_2_3_1",
             "road_3_1_0_0",
             "road_3_1_0_1",
             "road_4_0_1_0",
             "road_4_0_1_1",
             "road_5_1_2_0",
             "road_5_1_2_1",
             "road_4_2_3_0",
             "road_4_2_3_1",
             "road_3_2_0_0",
             "road_3_2_0_1",
             "road_4_1_1_0",
             "road_4_1_1_1",
             "road_5_2_2_0",
             "road_5_2_2_1",
             "road_4_3_3_0",
             "road_4_3_3_1",
             "road_0_3_0_0",
             "road_0_3_0_1",
             "road_1_2_1_0",
             "road_1_2_1_1",
             "road_2_3_2_0",
             "road_2_3_2_1",
             "road_1_4_3_0",
             "road_1_4_3_1",
             "road_1_3_0_0",
             "road_1_3_0_1",
             "road_2_2_1_0",
             "road_2_2_1_1",
             "road_3_3_2_0",
             "road_3_3_2_1",
             "road_2_4_3_0",
             "road_2_4_3_1",
             "road_2_3_0_0",
             "road_2_3_0_1",
             "road_3_2_1_0",
             "road_3_2_1_1",
             "road_4_3_2_0",
             "road_4_3_2_1",
             "road_3_4_3_0",
             "road_3_4_3_1",
             "road_3_3_0_0",
             "road_3_3_0_1",
             "road_4_2_1_0",
             "road_4_2_1_1",
             "road_5_3_2_0",
             "road_5_3_2_1",
             "road_4_4_3_0",
             "road_4_4_3_1",
             "road_0_4_0_0",
             "road_0_4_0_1",
             "road_1_3_1_0",
             "road_1_3_1_1",
             "road_2_4_2_0",
             "road_2_4_2_1",
             "road_1_5_3_0",
             "road_1_5_3_1",
             "road_1_4_0_0",
             "road_1_4_0_1",
             "road_2_3_1_0",
             "road_2_3_1_1",
             "road_3_4_2_0",
             "road_3_4_2_1",
             "road_2_5_3_0",
             "road_2_5_3_1",
             "road_2_4_0_0",
             "road_2_4_0_1",
             "road_3_3_1_0",
             "road_3_3_1_1",
             "road_4_4_2_0",
             "road_4_4_2_1",
             "road_3_5_3_0",
             "road_3_5_3_1",
             "road_3_4_0_0",
             "road_3_4_0_1",
             "road_4_3_1_0",
             "road_4_3_1_1",
             "road_5_4_2_0",
             "road_5_4_2_1",
             "road_4_5_3_0",
             "road_4_5_3_1",]
        
        self.mode = "start_waiting"
        assert self.mode == "all_all" or self.mode == "start_waiting", "mode must be one of 'all_all' or 'start_waiting'"
        """
        `mode` variable changes both reward and state.
        
        "all_all":
            - state: waiting & running vehicle count from all lanes (incoming & outgoing)
            - reward: waiting vehicle count from all lanes
            
        "start_waiting" - 
            - state: only waiting vehicle count from only start lanes (only incoming)
            - reward: waiting vehicle count from start lanes
        """
        """
        if self.mode == "all_all":
            self.state_space = len(self.all_lane_ids) * 2
        if self.mode == "start_waiting":
            self.state_space = len(self.start_lane_ids)
        """
        
        self.action_space = spaces.Discrete(9*16) #multiplied by 16 for num_intersections
        if self.mode == "all_all":
            self.observation_space = spaces.MultiDiscrete([100]*16*16) #multiplied by 16 for num_intersections
        else:
            self.observation_space = spaces.MultiDiscrete([100]*8*16) #multiplied by 16 for num_intersections

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        '''
        We can take this action(number between 0 and num_phases*num_intersections(9x16) and convert this into a base 9 series of numbers using
        numpy.base_repr library to return string of length num_intersections(16). For example, if we want to use phase 7 for 1st intersection, and 0 for the rest, 
        we use a string of 16 numbers as follows: 0000000000000007. We then convert this to an array
        With this series of actions, we use a for loop in the step function along with cityflow.set_tl_phase, which takes intersections[i] and actions[i] as arguments.
        After the for loop is over, we run self.cityflow.next_step()
        8 instead of 9 because apparently 9 is out of range for something
        '''
        actions = np.base_repr(action,8)
        action_list = list(actions)
        pad = len(self.intersection_ids) - len(action_list)
        actions = np,base_repr(action, 8, padding=pad)
        #print('1: ', actions[1])
        action_list = list(actions[1])
        #print('Here: ',action)
        #print('There: ', action_list)
        #print('One more: ',int(action_list[0]))
        for i in range(len(action_list)):
            #print(self.intersection_ids[i])
            #print(int(action_list[i]))
            self.cityflow.set_tl_phase(self.intersection_ids[i], int(action_list[i]))
        self.cityflow.next_step()
        #print('Step')
        state = self._get_state()
        reward = self._get_reward()

        self.current_step += 1

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                        "You should always call 'reset()' once you receive 'done = True' "
                        "-- any further steps are undefined behavior.")
            reward = 0.0

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        return state, reward, self.is_done, {}


    def reset(self):
        self.cityflow.reset()
        self.is_done = False
        self.current_step = 0

        return self._get_state()

    def render(self, mode='human'):
        print("Current time: " + self.cityflow.get_current_time())

    def _get_state(self):
        lane_vehicles_dict = self.cityflow.get_lane_vehicle_count()
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()

        state = None

        if self.mode=="all_all":
            state = np.zeros(len(self.all_lane_ids) * 2, dtype=np.float32)
            for i in range(len(self.all_lane_ids)):
                state[i*2] = lane_vehicles_dict[self.all_lane_ids[i]]
                state[i*2 + 1] = lane_waiting_vehicles_dict[self.all_lane_ids[i]]

        if self.mode=="start_waiting":
            state = np.zeros(len(self.start_lane_ids), dtype=np.float32)
            for i in range(len(self.start_lane_ids)):
                state[i] = lane_waiting_vehicles_dict[self.start_lane_ids[i]]

        return state

    def _get_reward(self):
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()
        reward = 0.0
        reward_time = 0.0
        total_reward = 0.0
        lamb = 0.1
        if self.mode == "all_all":
            for (road_id, num_vehicles) in lane_waiting_vehicles_dict.items():
                if road_id in self.all_lane_ids:
                    reward_time -= self.sec_per_step * num_vehicles

        if self.mode == "start_waiting":
            for (road_id, num_vehicles) in lane_waiting_vehicles_dict.items():
                if road_id in self.start_lane_ids:
                    reward_time -= self.sec_per_step * num_vehicles
        for k in range(len(self.intersection_ids)):
          for i in range(len(self.road_phases[k])):
            start = self.road_phases[k][i]['startRoad']
            end   = self.road_phases[k][i]['endRoad']
            for j in range(len(self.road_phases[k][i]['laneLinks'])):
              start_lane = self.road_phases[k][i]['laneLinks'][j]['startLaneIndex']
              end_lane = self.road_phases[k][i]['laneLinks'][j]['endLaneIndex']
              start_lane_id = start + '_' + str(start_lane)
              end_lane_id = end + '_' + str(end_lane)
              pressure = abs(lane_waiting_vehicles_dict[start_lane_id] - lane_waiting_vehicles_dict[end_lane_id])
              if pressure>-reward:
                reward = -pressure
          total_reward += reward
        total_reward = total_reward + lamb*reward_time
        return total_reward 

    def set_replay_path(self, path):
        travel_time = self.cityflow.set_replay_file(path)

    def results(self):
        return self.cityflow.get_average_travel_time()

    def seed(self, seed=None):
        self.cityflow.set_random_seed(seed)

    def get_path_to_config(self):
        return self.config_dir

    def set_save_replay(self, save_replay):
        self.cityflow.set_save_replay(save_replay)