#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend.py
# =====================================

import warnings
from collections import OrderedDict
from math import cos, sin, pi, sqrt
import random
from random import choice

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding

# gym.envs.user_defined.toyota_env.
from dynamics_and_models import VehicleDynamics, ReferencePath, EnvironmentModel
from endtoend_env_utils import *
from traffic import Traffic

warnings.filterwarnings("ignore")


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class CrossroadEnd2endAllRela(gym.Env):
    def __init__(self,
                 mode='training',
                 multi_display=False,
                 state_mode='fix',  # 'dyna'
                 future_point_num=25,
                 **kwargs):
        self.mode = mode
        self.dynamics = VehicleDynamics()
        self.interested_other = None
        self.detected_vehicles = None
        self.all_other = None
        self.ego_dynamics = None
        self.state_mode = state_mode
        self.init_state = {}
        self.action_number = 2
        self.ego_l, self.ego_w = Para.L, Para.W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.light_phase = None
        self.light_encoding = None
        self.task_encoding = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.init_state = None
        self.obs = None
        self.action = None

        self.done_type = 'not_done_yet'
        self.reward_info = None
        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.track_info_dim = Para.TRACK_ENCODING_DIM
        self.light_info_dim = Para.LIGHT_ENCODING_DIM
        self.task_info_dim = Para.TASK_ENCODING_DIM
        self.ref_info_dim = Para.REF_ENCODING_DIM
        self.per_other_info_dim = Para.PER_OTHER_INFO_DIM
        self.other_start_dim = sum([self.ego_info_dim, self.track_info_dim, self.light_info_dim,
                                    self.task_info_dim, self.ref_info_dim])
        self.veh_num = Para.MAX_VEH_NUM
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.other_number = sum([self.veh_num, self.bike_num, self.person_num])

        self.veh_mode_dict = None
        self.training_task = None
        self.env_model = None
        self.ref_path = None
        self.future_n_point = None
        self.future_point_num = future_point_num

        if not multi_display:
            self.traffic = Traffic(self.step_length,
                                   mode=self.mode,
                                   init_n_ego_dict=self.init_state)
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)
            plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):  # kwargs include three keys
        self.light_phase = self.traffic.init_light()
        self.training_task = choice(['left', 'straight', 'right'])
        self.task_encoding = TASK_ENCODING[self.training_task]
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        self.ref_path = ReferencePath(self.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.training_task]
        self.env_model = EnvironmentModel()
        self.init_state = self._reset_init_state()
        self.traffic.init_traffic(self.init_state, self.training_task)
        self.traffic.sim_step()
        ego_dynamics = self._get_ego_dynamics([self.init_state['ego']['v_x'],
                                               self.init_state['ego']['v_y'],
                                               self.init_state['ego']['r'],
                                               self.init_state['ego']['x'],
                                               self.init_state['ego']['y'],
                                               self.init_state['ego']['phi']],
                                              [0,
                                               0,
                                               self.dynamics.vehicle_params['miu'],
                                               self.dynamics.vehicle_params['miu']]
                                              )
        self._get_all_info(ego_dynamics)
        self.obs, other_mask_vector, self.future_n_point = self._get_obs()
        self.action = None
        self.reward_info = None
        self.done_type = 'not_done_yet'
        all_info = dict(future_n_point=self.future_n_point, mask=other_mask_vector)
        return self.obs, all_info

    def close(self):
        del self.traffic

    def step(self, action):
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_info = self._compute_reward(self.obs, self.action)
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()
        all_info = self._get_all_info(ego_dynamics)
        self.obs, other_mask_vector, self.future_n_point = self._get_obs()
        self.done_type, done = self._judge_done()
        self.reward_info.update({'final_rew': reward})
        all_info.update({'reward_info': self.reward_info, 'future_n_point': self.future_n_point, 'mask': other_mask_vector})
        return self.obs, reward, done, all_info

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_ego_dynamics(self, next_ego_state, next_ego_params):
        out = dict(v_x=next_ego_state[0],
                   v_y=next_ego_state[1],
                   r=next_ego_state[2],
                   x=next_ego_state[3],
                   y=next_ego_state[4],
                   phi=next_ego_state[5],
                   l=self.ego_l,
                   w=self.ego_w,
                   alpha_f=next_ego_params[0],
                   alpha_r=next_ego_params[1],
                   miu_f=next_ego_params[2],
                   miu_r=next_ego_params[3], )
        miu_f, miu_r = out['miu_f'], out['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / (abs(out['v_x']) + 1e-8)

        l, w, x, y, phi = out['l'], out['w'], out['x'], out['y'], out['phi']

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

        corner_point = cal_corner_point_of_ego_car()
        out.update(dict(alpha_f_bound=alpha_f_bound,
                        alpha_r_bound=alpha_r_bound,
                        r_bound=r_bound,
                        corner_point=corner_point))

        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_other = self.traffic.n_ego_vehicles['ego']  # coordination 2
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.light_phase = self.traffic.v_light

        # all_vehicles
        # dict(x=x, y=y, v=v, phi=a, l=length,
        #      w=width, route=route)

        all_info = dict(all_other=self.all_other,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.light_phase)
        return all_info

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done
        """
        if self.traffic.collision_flag:
            return 'collision', 1
        if self._break_road_constrain():
            return 'break_road_constrain', 1
        elif self._deviate_too_much():
            return 'deviate_too_much', 1
        elif self._break_stability():
            return 'break_stability', 1
        elif self._break_red_light():   # todo
            return 'break_red_light', 1
        elif self._is_achieve_goal():
            return 'good_done', 1
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_longi, delta_lateral, delta_phi, delta_v = self.obs[self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
        return True if abs(delta_lateral) > 15 else False

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['corner_point']))
        return not all(results)

    def _break_stability(self):
        alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
                                         self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics['r_bound']
        # if -alpha_f_bound < alpha_f < alpha_f_bound \
        #         and -alpha_r_bound < alpha_r < alpha_r_bound and \
        #         -r_bound < self.ego_dynamics['r'] < r_bound:
        if -r_bound < self.ego_dynamics['r'] < r_bound:
            return False
        else:
            return True

    def _break_red_light(self):
        return True if self.light_phase != 0 and self.ego_dynamics['y'] > -Para.CROSSROAD_SIZE/2 and self.training_task != 'right' else False

    def _is_achieve_goal(self):
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -Para.CROSSROAD_SIZE/2 - 10 and 0 < y < Para.LANE_NUMBER*Para.LANE_WIDTH else False
        elif self.training_task == 'right':
            return True if x > Para.CROSSROAD_SIZE/2 + 10 and -Para.LANE_NUMBER*Para.LANE_WIDTH < y < 0 else False
        else:
            assert self.training_task == 'straight'
            return True if y > Para.CROSSROAD_SIZE/2 + 10 and 0 < x < Para.LANE_NUMBER*Para.LANE_WIDTH else False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = 0.4 * steer_norm
        scaled_a_x = 2.25 * a_x_norm - 0.75  # [-3, 1.5]
        # if self.light_phase != 0 and self.ego_dynamics['y'] < -25 and self.training_task != 'right':
        #     scaled_steer = 0.
        #     scaled_a_x = -3.
        scaled_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        return scaled_action

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        steer, a_x = trans_action
        state = np.array([[current_v_x, current_v_y, current_r, current_x, current_y, current_phi]], dtype=np.float32)
        action = np.array([[steer, a_x]], dtype=np.float32)
        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, 10)
        next_ego_state, next_ego_params = next_ego_state.numpy()[0], next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_params

    def _get_obs(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_v_x = self.ego_dynamics['v_x']

        other_vector, other_mask_vector = self._construct_other_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        track_vector = self.ref_path.tracking_error_vector_vectorized(ego_x, ego_y, ego_phi, ego_v_x)
        future_n_point = self.ref_path.get_future_n_point(ego_x, ego_y, self.future_point_num)
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        vector = np.concatenate((ego_vector, track_vector, self.light_encoding, self.task_encoding,
                                 self.ref_path.ref_encoding, other_vector), axis=0)
        vector = vector.astype(np.float32)
        # vector = self._convert_to_rela(vector)

        return vector, other_mask_vector, future_n_point

    def _convert_to_rela(self, obs_abso):
        obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_other = self._split_all(obs_abso)
        obs_other_reshape = self._reshape_other(obs_other)
        ego_x, ego_y = obs_ego[3], obs_ego[4]
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_other_info_dim - 2)), dtype=np.float32)
        ego = ego[np.newaxis, :]
        rela = obs_other_reshape - ego
        rela_obs_other = self._reshape_other(rela, reverse=True)
        return np.concatenate([obs_ego, obs_track, obs_light, obs_task, obs_ref, rela_obs_other], axis=0)

    def _convert_to_abso(self, obs_rela):
        obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_other = self._split_all(obs_rela)
        obs_other_reshape = self._reshape_other(obs_other)
        ego_x, ego_y = obs_ego[3], obs_ego[4]
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_other_info_dim - 2)), dtype=np.float32)
        ego = ego[np.newaxis, :]
        abso = obs_other_reshape + ego
        abso_obs_other = self._reshape_other(abso, reverse=True)
        return np.concatenate([obs_ego, obs_track, obs_light, obs_task, obs_ref, abso_obs_other])

    def _split_all(self, obs):
        obs_ego = obs[:self.ego_info_dim]
        obs_track = obs[self.ego_info_dim:
                        self.ego_info_dim + self.track_info_dim]
        obs_light = obs[self.ego_info_dim + self.track_info_dim:
                        self.ego_info_dim + self.track_info_dim + self.light_info_dim]
        obs_task = obs[self.ego_info_dim + self.track_info_dim + self.light_info_dim:
                       self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim]
        obs_ref = obs[self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim:
                      self.other_start_dim]
        obs_other = obs[self.other_start_dim:]

        return obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_other

    def _split_other(self, obs_other):
        obs_bike = obs_other[:self.bike_num * self.per_other_info_dim]
        obs_person = obs_other[self.bike_num * self.per_other_info_dim:
                               (self.bike_num + self.person_num) * self.per_other_info_dim]
        obs_veh = obs_other[(self.bike_num + self.person_num) * self.per_other_info_dim:]
        return obs_bike, obs_person, obs_veh

    def _reshape_other(self, obs_other, reverse=False):
        if reverse:
            return np.reshape(obs_other, (self.other_number * self.per_other_info_dim,))
        else:
            return np.reshape(obs_other, (self.other_number, self.per_other_info_dim))

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
        return np.array(ego_feature, dtype=np.float32)

    def _construct_other_vector_short(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        other_vector = []
        other_mask_vector = []

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_other(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            for v in vs:
                v.update(exist=True)
                route_list = v['route']
                start = route_list[0]
                end = route_list[1]
                if start == name_setting['do'] and end == name_setting['li']:
                    v.update(turn_rad=1 / (Para.CROSSROAD_SIZE / 2 + 0.5 * Para.LANE_WIDTH))
                    dl.append(v)
                elif start == name_setting['do'] and end == name_setting['ui']:
                    v.update(turn_rad=0.)
                    du.append(v)
                elif start == name_setting['do'] and end == name_setting['ri']:
                    v.update(turn_rad=-1 / (Para.CROSSROAD_SIZE / 2 - 2.5 * Para.LANE_WIDTH))
                    dr.append(v)

                elif start == name_setting['ro'] and end == name_setting['di']:
                    v.update(turn_rad=1 / (Para.CROSSROAD_SIZE / 2 + 0.5 * Para.LANE_WIDTH))
                    rd.append(v)
                elif start == name_setting['ro'] and end == name_setting['li']:
                    v.update(turn_rad=0.)
                    rl.append(v)
                elif start == name_setting['ro'] and end == name_setting['ui']:
                    v.update(turn_rad=-1 / (Para.CROSSROAD_SIZE / 2 - 2.5 * Para.LANE_WIDTH))
                    ru.append(v)

                elif start == name_setting['uo'] and end == name_setting['ri']:
                    v.update(turn_rad=1 / (Para.CROSSROAD_SIZE / 2 + 0.5 * Para.LANE_WIDTH))
                    ur.append(v)
                elif start == name_setting['uo'] and end == name_setting['di']:
                    v.update(turn_rad=0.)
                    ud.append(v)
                elif start == name_setting['uo'] and end == name_setting['li']:
                    v.update(turn_rad=-1 / (Para.CROSSROAD_SIZE / 2 - 2.5 * Para.LANE_WIDTH))
                    ul.append(v)

                elif start == name_setting['lo'] and end == name_setting['ui']:
                    v.update(turn_rad=1 / (Para.CROSSROAD_SIZE / 2 + 0.5 * Para.LANE_WIDTH))
                    lu.append(v)
                elif start == name_setting['lo'] and end == name_setting['ri']:
                    v.update(turn_rad=0.)
                    lr.append(v)
                elif start == name_setting['lo'] and end == name_setting['di']:
                    v.update(turn_rad=-1 / (Para.CROSSROAD_SIZE / 2 - 2.5 * Para.LANE_WIDTH))
                    ld.append(v)

            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -Para.CROSSROAD_SIZE/2-10 and v['y'] > ego_y-2, dl))  # interest of left straight
            du = list(filter(lambda v: ego_y-2 < v['y'] < Para.CROSSROAD_SIZE/2+10 and v['x'] < ego_x+5, du))  # interest of left straight

            dr = list(filter(lambda v: v['x'] < Para.CROSSROAD_SIZE/2+10 and v['y'] > ego_y, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < Para.CROSSROAD_SIZE/2+10 and v['y'] < Para.CROSSROAD_SIZE/2+10, ru))  # interest of straight

            if task == 'straight':
                ur = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < Para.CROSSROAD_SIZE/2+10, ur))  # interest of straight
            elif task == 'right':
                ur = list(filter(lambda v: v['x'] < Para.CROSSROAD_SIZE/2+10 and v['y'] < Para.CROSSROAD_SIZE/2, ur))  # interest of right
            ud = list(filter(lambda v: max(ego_y-2, -Para.CROSSROAD_SIZE/2) < v['y'] < Para.CROSSROAD_SIZE/2 and ego_x > v['x'], ud))  # interest of left
            ul = list(filter(lambda v: -Para.CROSSROAD_SIZE/2-10 < v['x'] < ego_x and v['y'] < Para.CROSSROAD_SIZE/2, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -Para.CROSSROAD_SIZE/2-10 < v['x'] < Para.CROSSROAD_SIZE/2+10, lr))  # interest of right
            ld = ld  # not interest in case of traffic light

            # sort
            dl = sorted(dl, key=lambda v: (v['y'], -v['x']))
            du = sorted(du, key=lambda v: v['y'])
            dr = sorted(dr, key=lambda v: (v['y'], v['x']))

            ru = sorted(ru, key=lambda v: (-v['x'], v['y']), reverse=True)

            if task == 'straight':
                ur = sorted(ur, key=lambda v: v['y'])
            elif task == 'right':
                ur = sorted(ur, key=lambda v: (-v['y'], v['x']), reverse=True)

            ud = sorted(ud, key=lambda v: v['y'])
            ul = sorted(ul, key=lambda v: (-v['y'], -v['x']), reverse=True)

            lr = sorted(lr, key=lambda v: -v['x'])

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            mode2fillvalue = dict(
                dl=dict(x=Para.LANE_WIDTH/2, y=-(Para.CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '4i'), turn_rad=0., exist=False),
                du=dict(x=Para.LANE_WIDTH*1.5, y=-(Para.CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '3i'), turn_rad=0., exist=False),
                dr=dict(x=Para.LANE_WIDTH*(Para.LANE_NUMBER-0.5), y=-(Para.CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '2i'), turn_rad=0., exist=False),
                ru=dict(x=(Para.CROSSROAD_SIZE/2+15), y=Para.LANE_WIDTH*(Para.LANE_NUMBER-0.5), v=0, phi=180, w=2.5, l=5, route=('2o', '3i'), turn_rad=0., exist=False),
                ur=dict(x=-Para.LANE_WIDTH/2, y=(Para.CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'), turn_rad=0., exist=False),
                ud=dict(x=-Para.LANE_WIDTH*1.5, y=(Para.CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '1i'), turn_rad=0., exist=False),
                ul=dict(x=-Para.LANE_WIDTH*(Para.LANE_NUMBER-0.5), y=(Para.CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '4i'), turn_rad=0., exist=False),
                lr=dict(x=-(Para.CROSSROAD_SIZE/2+20), y=-Para.LANE_WIDTH*1.5, v=0, phi=0, w=2.5, l=5, route=('4o', '2i'), turn_rad=0., exist=False))

            tmp_v = []
            if self.state_mode == 'fix':
                for mode, num in VEHICLE_MODE_DICT[task].items():
                    tmp_v_mode = slice_or_fill(eval(mode), mode2fillvalue[mode], num)
                    tmp_v.extend(tmp_v_mode)
            elif self.state_mode == 'dyna':
                for mode, num in VEHICLE_MODE_DICT[task].items():
                    tmp_v.extend(eval(mode))
                while len(tmp_v) < self.veh_num:
                    if self.training_task == 'left':
                        tmp_v.append(mode2fillvalue['dl'])
                    elif self.training_task == 'straight':
                        tmp_v.append(mode2fillvalue['du'])
                    else:
                        tmp_v.append(mode2fillvalue['dr'])
                if len(tmp_v) > self.veh_num:
                    tmp_v = sorted(tmp_v, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2), -v['x']))
                    tmp_v = tmp_v[:self.veh_num]

            return tmp_v

        self.interested_other = filter_interested_other(self.all_other, self.training_task)

        for other in self.interested_other:
            other_x, other_y, other_v, other_phi, other_l, other_w, other_turn_rad, other_mask = \
                other['x'], other['y'], other['v'], other['phi'], other['l'], other['w'], other[
                    'turn_rad'], other['exist']
            other_vector.extend(
                [other_x, other_y, other_v, other_phi, other_l, other_w] + [other_turn_rad])
            other_mask_vector.append(other_mask)
        return np.array(other_vector, dtype=np.float32), np.array(other_mask_vector, dtype=np.float32)

    def _reset_init_state(self):
        if self.training_task == 'left':
            random_index = int(np.random.random()*(900+500)) + 700
        elif self.training_task == 'straight':
            random_index = int(np.random.random()*(1200+500)) + 700
        else:
            random_index = int(np.random.random()*(420+500)) + 700

        x, y, phi, exp_v = self.ref_path.idx2point(random_index)
        v = exp_v * np.random.random()
        routeID = TASK2ROUTEID[self.training_task]
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x,
                             y=y,
                             phi=phi,
                             l=self.ego_l,
                             w=self.ego_w,
                             routeID=routeID,
                             ))

    def _compute_reward(self, obs, action):
        obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
        reward, _, _, _, _, _, reward_dict = self.env_model.compute_rewards(obses, actions)
        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        return reward.numpy()[0], reward_dict

    def render(self, mode='human', weights=None):
        if mode == 'human':
            # plot basic map
            square_length = Para.CROSSROAD_SIZE
            extension = 40
            lane_width = Para.LANE_WIDTH
            light_line_width = 3
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax.axis("equal")
            # ax.add_patch(plt.Rectangle((-square_length / 2 - extension, -square_length / 2 - extension),
            #                            square_length + 2 * extension, square_length + 2 * extension, edgecolor='black',
            #                            facecolor='none', linewidth=2))

            # ----------arrow--------------
            plt.arrow(lane_width/2, -square_length / 2-10, 0, 5, color='b')
            plt.arrow(lane_width/2, -square_length / 2-10+5, -0.5, 0, color='b', head_width=1)
            plt.arrow(lane_width*1.5, -square_length / 2-10, 0, 4, color='b', head_width=1)
            plt.arrow(lane_width*2.5, -square_length / 2 - 10, 0, 5, color='b')
            plt.arrow(lane_width*2.5, -square_length / 2 - 10+5, 0.5, 0, color='b', head_width=1)

            # ----------horizon--------------

            plt.plot([-square_length / 2 - extension, -square_length / 2], [0.3, 0.3], color='orange')
            plt.plot([-square_length / 2 - extension, -square_length / 2], [-0.3, -0.3], color='orange')
            plt.plot([square_length / 2 + extension, square_length / 2], [0.3, 0.3], color='orange')
            plt.plot([square_length / 2 + extension, square_length / 2], [-0.3, -0.3], color='orange')

            #
            for i in range(1, Para.LANE_NUMBER + 1):
                linestyle = dotted_line_style if i < Para.LANE_NUMBER else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER else 2
                plt.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # ----------vertical----------------
            plt.plot([0.3, 0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
            plt.plot([-0.3, -0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
            plt.plot([0.3, 0.3], [square_length / 2 + extension, square_length / 2], color='orange')
            plt.plot([-0.3, -0.3], [square_length / 2 + extension, square_length / 2], color='orange')

            #
            for i in range(1, Para.LANE_NUMBER + 1):
                linestyle = dotted_line_style if i < Para.LANE_NUMBER else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER else 2
                plt.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # ----------stop line--------------
            # plt.plot([0, 2 * lane_width], [-square_length / 2, -square_length / 2],
            #          color='black')
            # plt.plot([-2 * lane_width, 0], [square_length / 2, square_length / 2],
            #          color='black')
            # plt.plot([-square_length / 2, -square_length / 2], [0, -2 * lane_width],
            #          color='black')
            # plt.plot([square_length / 2, square_length / 2], [2 * lane_width, 0],
            #          color='black')
            v_light = self.light_phase
            if v_light == 0:
                v_color, h_color = 'green', 'red'
            elif v_light == 1:
                v_color, h_color = 'orange', 'red'
            elif v_light == 2:
                v_color, h_color = 'red', 'green'
            else:
                v_color, h_color = 'red', 'orange'

            plt.plot([0, (Para.LANE_NUMBER-1)*lane_width], [-square_length / 2, -square_length / 2],
                     color=v_color, linewidth=light_line_width)
            plt.plot([(Para.LANE_NUMBER-1)*lane_width, Para.LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
                     color='green', linewidth=light_line_width)

            plt.plot([-Para.LANE_NUMBER * lane_width, -(Para.LANE_NUMBER-1)*lane_width], [square_length / 2, square_length / 2],
                     color='green', linewidth=light_line_width)
            plt.plot([-(Para.LANE_NUMBER-1)*lane_width, 0], [square_length / 2, square_length / 2],
                     color=v_color, linewidth=light_line_width)

            plt.plot([-square_length / 2, -square_length / 2], [0, -(Para.LANE_NUMBER-1)*lane_width],
                     color=h_color, linewidth=light_line_width)
            plt.plot([-square_length / 2, -square_length / 2], [-(Para.LANE_NUMBER-1)*lane_width, -Para.LANE_NUMBER * lane_width],
                     color='green', linewidth=light_line_width)

            plt.plot([square_length / 2, square_length / 2], [(Para.LANE_NUMBER-1)*lane_width, 0],
                     color=h_color, linewidth=light_line_width)
            plt.plot([square_length / 2, square_length / 2], [Para.LANE_NUMBER * lane_width, (Para.LANE_NUMBER-1)*lane_width],
                     color='green', linewidth=light_line_width)

            # ----------Oblique--------------
            plt.plot([Para.LANE_NUMBER * lane_width, square_length / 2], [-square_length / 2, -Para.LANE_NUMBER * lane_width],
                     color='black', linewidth=2)
            plt.plot([Para.LANE_NUMBER * lane_width, square_length / 2], [square_length / 2, Para.LANE_NUMBER * lane_width],
                     color='black', linewidth=2)
            plt.plot([-Para.LANE_NUMBER * lane_width, -square_length / 2], [-square_length / 2, -Para.LANE_NUMBER * lane_width],
                     color='black', linewidth=2)
            plt.plot([-Para.LANE_NUMBER * lane_width, -square_length / 2], [square_length / 2, Para.LANE_NUMBER * lane_width],
                     color='black', linewidth=2)

            def is_in_plot_area(x, y, tolerance=5):
                if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                        -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                    return True
                else:
                    return False

            def draw_rotate_rec(x, y, a, l, w, color, linestyle='-'):
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
                ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
                ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
                ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

            def plot_phi_line(x, y, phi, color):
                line_length = 5
                x_forw, y_forw = x + line_length * cos(phi*pi/180.),\
                                 y + line_length * sin(phi*pi/180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            # plot cars
            for veh in self.all_other:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

            # plot interested participants
            if weights is not None:
                assert weights.shape == (self.other_number,), print(weights.shape)
            index_top_k_in_weights = weights.argsort()[-4:][::-1]
            for i in range(len(self.interested_other)):
                veh = self.interested_other[i]
                mask = veh['exist']
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    color = 'm'
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')
                if i in index_top_k_in_weights:
                    plt.text(veh_x, veh_y, "{:.2f}".format(weights[i]), color='red', fontsize=15)

            # plot own car
            abso_obs = self._convert_to_abso(self.obs)
            obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_other = self._split_all(abso_obs)
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = obs_ego
            devi_longi, devi_lateral, devi_phi, devi_v = obs_track

            plot_phi_line(ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, self.ego_l, self.ego_w, 'red')

            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
            _, point = self.ref_path._find_closest_point(ego_x, ego_y)
            path_x, path_y, path_phi, path_v = point[0], point[1], point[2], point[3]
            plt.plot(path_x, path_y, 'g.')
            plt.plot(self.future_n_point[0], self.future_n_point[1], 'g.')

            # plot real time traj
            # try:
            #     color = ['b', 'lime']
            #     for i, item in enumerate(real_time_traj):
            #         if i == path_index:
            #             plt.plot(item.path[0], item.path[1], color=color[i], alpha=1.0)
            #         else:
            #             plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
            #         indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
            #         path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
            #         plt.plot(path_x, path_y,  color=color[i])
            # except Exception:
            #     pass

            # for j, item_point in enumerate(self.real_path.feature_points_all):
            #     for k in range(len(item_point)):
            #         plt.scatter(item_point[k][0], item_point[k][1], c='g')

            # text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
            plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
            plt.text(text_x, text_y_start - next(ge), 'devi_longi: {:.2f}m'.format(devi_longi))
            plt.text(text_x, text_y_start - next(ge), 'devi_lateral: {:.2f}m'.format(devi_lateral))
            plt.text(text_x, text_y_start - next(ge), 'devi_v: {:.2f}m/s'.format(devi_v))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
            plt.text(text_x, text_y_start - next(ge), r'devi_phi: ${:.2f}\degree$'.format(devi_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(path_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))

            if self.action is not None:
                steer, a_x = self.action[0], self.action[1]
                plt.text(text_x, text_y_start - next(ge), r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 80, 60
            ge = iter(range(0, 1000, 4))

            # done info
            plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.done_type))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

            # indicator for trajectory selection
            # text_x, text_y_start = -25, -65
            # ge = iter(range(0, 1000, 6))
            # if traj_return is not None:
            #     for i, value in enumerate(traj_return):
            #         if i==path_index:
            #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=14, color=color[i], fontstyle='italic')
            #         else:
            #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=12, color=color[i], fontstyle='italic')

            plt.show()
            plt.pause(0.001)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


def test_end2end():
    import random
    env = CrossroadEnd2endAllRela()
    env_model = EnvironmentModel()
    obs, all_info = env.reset()
    i = 0
    while i < 100000:
        for j in range(40):
            i += 1
            # action=2*np.random.random(2)-1
            if obs[4]<-18:
                action = np.array([0, 1], dtype=np.float32)
            else:
                action = np.array([0.0, 0.33], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            obses = np.tile(obses, (2, 1))
            ref_points = np.tile(info['future_n_point'], (2, 1, 1))
            # obses_ego[:, (-env.task_info_dim-env.light_info_dim)] = random.randint(0, 2), random.randint(0, 2)
            # obses_ego[:, (-env.task_info_dim)] = list(TASK_DICT.values())[random.randint(0, 2)], list(TASK_DICT.values())[random.randint(0, 2)]
            env_model.reset(obses)
            for i in range(5):
                obses, rewards, punish_term_for_training, real_punish_term, veh2veh4real, \
                veh2road4real, veh2line4real = env_model.rollout_out(np.tile(actions, (2, 1)), ref_points[:, :, i])
                # print(obses[:, env.ego_info_dim + env.track_info_dim: env.ego_info_dim+env.track_info_dim+env.light_info_dim])
            # print(env.training_task, obs[env.ego_info_dim + env.track_info_dim: env.ego_info_dim+env.track_info_dim+env.light_info_dim], env.light_phase)
            # print('task:', obs[env.ego_info_dim + env.track_info_dim + env.per_path_info_dim * env.num_future_data + env.light_info_dim])
            env.render(weights=np.zeros(env.other_number,))
            # if done:
            #     break
        obs, _ = env.reset()
        env.render(weights=np.zeros(env.other_number,))


if __name__ == '__main__':
    test_end2end()