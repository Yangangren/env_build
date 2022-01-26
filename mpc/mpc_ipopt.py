#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/01/13
# @Author  : Yang Guan, Yangang Ren, Jianhua Jiang (Tsinghua Univ.)
# @FileName: mpc_ipopt.py
# @Function: compare ADP and MPC
# =====================================

import math
import datetime

import casadi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.transforms import Affine2D
from matplotlib.collections import PatchCollection
from casadi import *

from endtoend import CrossroadEnd2endMixPI
from hierarchical_decision.multi_path_generator import MultiPathGenerator
from dynamics_and_models import ReferencePath, EnvironmentModel
from endtoend_env_utils import Para, REF_ENCODING, rotate_coordination, LIGHT_PHASE_TO_GREEN_OR_RED
from mpc.main import TimerStat
from utils.load_policy import LoadPolicy
from utils.recorder import Recorder


def deal_with_phi_casa(phi):
    phi = if_else(phi > 180, phi - 360, if_else(phi < -180, phi + 360, phi))
    return phi


def deal_with_phi(phi):
    phi = if_else(phi > 180, phi - 360, if_else(phi < -180, phi + 360, phi))
    return phi


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, x, u, tau):
        v_x, v_y, r, x, y, phi = x[0], x[1], x[2], x[3], x[4], x[5]
        phi = phi * np.pi / 180.
        steer, a_x = u[0], u[1]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * power(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (power(a, 2) * C_f + power(b, 2) * C_r) - I_z * v_x),
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return next_state


class Dynamics(object):
    def __init__(self, x_init, num_future_data, task, tau):
        self.task = task
        self.tau = tau
        self.vd = VehicleDynamics()
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.veh_num = Para.MAX_VEH_NUM
        self.num_future_data = num_future_data
        self.participants = x_init[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + self.num_future_data * Para.PER_PATH_INFO_DIM + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM + Para.REF_ENCODING_DIM + Para.HIS_ACT_ENCODING_DIM:]
        self.bikes = self.participants[:self.bike_num * Para.PER_OTHER_INFO_DIM]
        self.persons = self.participants[self.bike_num * Para.PER_OTHER_INFO_DIM: self.bike_num * Para.PER_OTHER_INFO_DIM + self.person_num * Para.PER_OTHER_INFO_DIM]
        self.vehs = self.participants[self.bike_num * Para.PER_OTHER_INFO_DIM + self.person_num * Para.PER_OTHER_INFO_DIM:]
        self.x_init = x_init

    def tracking_error_pred(self, next_ego, next_ref_index):
        ref_points = self.x_init[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM: Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + self.num_future_data * Para.PER_PATH_INFO_DIM]

        # todo -----  find the closest ref_point
        # next_ref_index_, next_ref_point = self.find_closest_ref_point(next_ego[3], next_ego[4], ref_points, k)
        next_ref_point = ref_points[(next_ref_index - 1) * Para.PER_PATH_INFO_DIM: next_ref_index * Para.PER_PATH_INFO_DIM]

        v_x, v_y, r, x, y, phi = next_ego[0], next_ego[1], next_ego[2], next_ego[3], next_ego[4], next_ego[5]
        ref_x, ref_y, ref_phi, ref_v = next_ref_point[0], next_ref_point[1], next_ref_point[2], next_ref_point[3]
        ref_phi_rad = ref_phi * pi / 180

        vector_ref_phi = np.array([np.cos(ref_phi_rad), np.sin(ref_phi_rad)])
        vector_ref_phi_ccw_90 = np.array([-np.sin(ref_phi_rad), np.cos(ref_phi_rad)]) # ccw for counterclockwise
        vector_ego2ref = np.array([ref_x - x, ref_y - y])

        delta_x = -1 * dot(vector_ego2ref, vector_ref_phi)
        delta_y = -1 * dot(vector_ego2ref, vector_ref_phi_ccw_90)
        # delta_x = -1 * (cos(ref_phi_rad) * (x - ref_x) + sin(ref_phi_rad) * (y - ref_y))  # todo check +-
        # delta_y = -1 * (-sin(ref_phi_rad) * (x - ref_x) + cos(ref_phi_rad) * (y - ref_y))  # todo check +-
        delta_phi = deal_with_phi_casa(phi - ref_phi)
        delta_v = v_x - ref_v

        return [delta_x, delta_y, delta_phi, delta_v]

    def find_closest_ref_point(self, ego_x, ego_y, ref_points, k):
        dists = []
        for i in range(self.num_future_data):
            path_x, path_y = ref_points[i * 4], ref_points[i * 4 + 1]
            dis = pow((ego_x - path_x), 2) + pow((ego_y - path_y), 2)
            dists.append(dis)
        ref_index = casadi.mmin(dists)  # todo
        if ref_index < k:
            ref_index = k
        ref_point = ref_points[ref_index]
        return ref_index, ref_point

    def bikes_pred(self):
        predictions = []
        for bikes_index in range(self.bike_num):
            predictions += \
                self.predict_for_bike_mode(
                    self.bikes[bikes_index * Para.PER_OTHER_INFO_DIM: (bikes_index + 1) * Para.PER_OTHER_INFO_DIM])
        self.bikes = predictions

    def persons_pred(self):
        predictions = []
        for persons_index in range(self.person_num):
            predictions += \
                self.predict_for_person_mode(
                    self.persons[persons_index * Para.PER_OTHER_INFO_DIM: (persons_index + 1) * Para.PER_OTHER_INFO_DIM])
        self.persons = predictions

    def vehs_pred(self):
        predictions = []
        for vehs_index in range(self.veh_num):
            predictions += \
                self.predict_for_veh_mode(
                    self.vehs[vehs_index * Para.PER_OTHER_INFO_DIM: (vehs_index + 1) * Para.PER_OTHER_INFO_DIM])
        self.vehs = predictions

    def predict_for_bike_mode(self, bikes):
        bike_x, bike_y, bike_v, bike_phi, bike_turn_rad = bikes[0], bikes[1], bikes[2], bikes[3], bikes[9]
        bike_phis_rad = bike_phi * np.pi / 180.
        bike_x_delta = bike_v * self.tau * math.cos(bike_phis_rad)
        bike_y_delta = bike_v * self.tau * math.sin(bike_phis_rad)
        bike_phi_rad_delta = bike_v * self.tau * bike_turn_rad

        next_bike_x, next_bike_y, next_bike_v, next_bike_phi_rad = \
            bike_x + bike_x_delta, bike_y + bike_y_delta, bike_v, bike_phis_rad + bike_phi_rad_delta
        next_bike_phi = next_bike_phi_rad * 180 / np.pi
        next_bike_phi = deal_with_phi(next_bike_phi)

        return [next_bike_x, next_bike_y, next_bike_v, next_bike_phi, bikes[4], bikes[5], bikes[6], bikes[7], bikes[8], bikes[9]]

    def predict_for_person_mode(self, persons):
        person_x, person_y, person_v, person_phi = persons[0], persons[1], persons[2], persons[3]
        person_phis_rad = person_phi * np.pi / 180.
        person_x_delta = person_v * self.tau * math.cos(person_phis_rad)
        person_y_delta = person_v * self.tau * math.sin(person_phis_rad)

        next_person_x, next_person_y, next_person_v, next_person_phi_rad = \
            person_x + person_x_delta, person_y + person_y_delta, person_v, person_phis_rad
        next_person_phi = next_person_phi_rad * 180 / np.pi
        next_person_phi = deal_with_phi(next_person_phi)

        return [next_person_x, next_person_y, next_person_v, next_person_phi, persons[4], persons[5], persons[6], persons[7], persons[8], persons[9]]

    def predict_for_veh_mode(self, vehs):
        veh_x, veh_y, veh_v, veh_phi, veh_turn_rad = vehs[0], vehs[1], vehs[2], vehs[3], vehs[9]
        veh_phis_rad = veh_phi * np.pi / 180.
        veh_x_delta = veh_v * self.tau * math.cos(veh_phis_rad)
        veh_y_delta = veh_v * self.tau * math.sin(veh_phis_rad)
        veh_phi_rad_delta = veh_v * self.tau * veh_turn_rad

        next_veh_x, next_veh_y, next_veh_v, next_veh_phi_rad = \
            veh_x + veh_x_delta, veh_y + veh_y_delta, veh_v, veh_phis_rad + veh_phi_rad_delta
        next_veh_phi = next_veh_phi_rad * 180 / np.pi
        next_veh_phi = deal_with_phi(next_veh_phi)

        return [next_veh_x, next_veh_y, next_veh_v, next_veh_phi, vehs[4], vehs[5], vehs[6], vehs[7], vehs[8], vehs[9]]

    def f_xu(self, x, u, k):
        next_ego = self.vd.f_xu(x, u, self.tau)           # Unit of heading angle is degree
        next_ref_index = k
        next_tracking = self.tracking_error_pred(next_ego, next_ref_index)
        his_action = [x[12], x[13], u[0], u[1]]
        return next_ego + next_tracking + his_action

    def g_x(self, x):
        ego_x, ego_y, ego_phi, delta_v = x[3], x[4], x[5], x[9]
        g_list = []
        ego_lws = (Para.L - Para.W) / 2.
        ego_front_points = ego_x + ego_lws * cos(ego_phi * np.pi / 180.), \
                           ego_y + ego_lws * sin(ego_phi * np.pi / 180.)
        ego_rear_points = ego_x - ego_lws * cos(ego_phi * np.pi / 180.), \
                          ego_y - ego_lws * sin(ego_phi * np.pi / 180.)

        for bikes_index in range(self.bike_num):
            bike = self.bikes[bikes_index * Para.PER_OTHER_INFO_DIM: (bikes_index + 1) * Para.PER_OTHER_INFO_DIM]
            bike_x, bike_y, bike_phi, bike_l, bike_w = bike[0], bike[1], bike[3], bike[4], bike[5]
            bike_lws = (bike_l - bike_w) / 2.
            bike_front_points = bike_x + bike_lws * math.cos(bike_phi * np.pi / 180.), \
                               bike_y + bike_lws * math.sin(bike_phi * np.pi / 180.)
            bike_rear_points = bike_x - bike_lws * math.cos(bike_phi * np.pi / 180.), \
                              bike_y - bike_lws * math.sin(bike_phi * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for bike_point in [bike_front_points, bike_rear_points]:
                    veh2bike_dist = sqrt(power(ego_point[0] - bike_point[0], 2) + power(ego_point[1] - bike_point[1], 2)) - 2.5
                    g_list.append(veh2bike_dist)

        for persons_index in range(self.person_num):
            person = self.persons[persons_index * Para.PER_OTHER_INFO_DIM: (persons_index + 1) * Para.PER_OTHER_INFO_DIM]
            person_x, person_y, person_phi = person[0], person[1], person[3]
            person_point = person_x, person_y
            for ego_point in [ego_front_points, ego_rear_points]:
                veh2person_dist = sqrt(power(ego_point[0] - person_point[0], 2) + power(ego_point[1] - person_point[1], 2)) - 2.5
                g_list.append(veh2person_dist)

        for vehs_index in range(self.veh_num):
            veh = self.vehs[vehs_index * Para.PER_OTHER_INFO_DIM: (vehs_index + 1) * Para.PER_OTHER_INFO_DIM]
            veh_x, veh_y, veh_phi, veh_l, veh_w = veh[0], veh[1], veh[3], veh[4], veh[5]
            veh_lws = (veh_l - veh_w) / 2.
            veh_front_points = veh_x + veh_lws * math.cos(veh_phi * np.pi / 180.), \
                               veh_y + veh_lws * math.sin(veh_phi * np.pi / 180.)
            veh_rear_points = veh_x - veh_lws * math.cos(veh_phi * np.pi / 180.), \
                              veh_y - veh_lws * math.sin(veh_phi * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = sqrt(power(ego_point[0] - veh_point[0], 2) + power(ego_point[1] - veh_point[1], 2)) - 2.5
                    g_list.append(veh2veh_dist)

        # g_list.append(-1 * delta_v)

        # for ego_point in [ego_front_points]:
        #     g_list.append(if_else(logic_and(ego_point[1]<-18, ego_point[0]<1), ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[1]<-18, 3.75-ego_point[0]<1), 3.75-ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]>0, 0-ego_point[1]<0), 0-ego_point[1], 1))
        #     g_list.append(if_else(logic_and(ego_point[1]>-18, 3.75-ego_point[0]<1), 3.75-ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]<0, 7.5-ego_point[1]<1), 7.5-ego_point[1]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]<-18, ego_point[1]-0<1), ego_point[1]-0-1, 1))

        return g_list


class ModelPredictiveControl(object):
    def __init__(self, horizon, task, num_future_data):
        self.horizon = horizon
        self.base_frequency = 10.
        self.num_future_data = num_future_data
        self.task = task
        self.DYNAMICS_DIM = 14      # ego_info + track_error + his_action
        self.ACTION_DIM = 2
        self.dynamics = None
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.veh_num = Para.MAX_VEH_NUM
        self.his_action_before = Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.TRACK_ENCODING_DIM * self.num_future_data + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM + Para.REF_ENCODING_DIM
        self._sol_dic = {'ipopt.print_level': 0,
                         'ipopt.sb': 'yes',
                         'print_time': 0}

    def mpc_solver(self, x_init, XO):
        self.dynamics = Dynamics(x_init, self.num_future_data, self.task, 1 / self.base_frequency)

        x = SX.sym('x', self.DYNAMICS_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        # Create empty NLP
        w = []
        lbw = []                 # lower bound for state and action constraints
        ubw = []                 # upper bound for state and action constraints
        lbg = []                 # lower bound for distance constraint
        ubg = []                 # upper bound for distance constraint
        G = []                   # dynamic constraints
        J = 0                    # accumulated cost

        # Initial conditions
        Xk = MX.sym('X0', self.DYNAMICS_DIM)
        w += [Xk]
        lbw += x_init[:Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM] + x_init[self.his_action_before: self.his_action_before + Para.HIS_ACT_ENCODING_DIM]
        ubw += x_init[:Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM] + x_init[self.his_action_before: self.his_action_before + Para.HIS_ACT_ENCODING_DIM]

        for k in range(1, self.horizon + 1):
            f = vertcat(*self.dynamics.f_xu(x, u, k))  # next_ego + next_tracking
            F = Function("F", [x, u], [f])
            g = vertcat(*self.dynamics.g_x(x))  # constraints
            G_f = Function('Gf', [x], [g])

            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += [-0.4, -3.]      # action constraints
            ubw += [0.4, 1.5]

            Fk = F(Xk, Uk)
            Gk = G_f(Xk)
            self.dynamics.bikes_pred()  # todo check
            self.dynamics.persons_pred()
            self.dynamics.vehs_pred()
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.DYNAMICS_DIM)

            # Dynamic Constraints
            G += [Fk - Xk]                                         # ego vehicle dynamic constraints
            lbg += [0.0] * self.DYNAMICS_DIM
            ubg += [0.0] * self.DYNAMICS_DIM

            G += [Gk]                                              # surrounding bike constraints
            lbg += [0.0] * (self.bike_num * 4)
            ubg += [inf] * (self.bike_num * 4)
            #                                                      # surrounding person constraints
            lbg += [0.0] * (self.person_num * 2)
            ubg += [inf] * (self.person_num * 2)
            #                                                      # surrounding vehicle constraints
            lbg += [0.0] * (self.veh_num * 4)
            ubg += [inf] * (self.veh_num * 4)

            w += [Xk]
            lbw += [0.] + [-inf] * (self.DYNAMICS_DIM - 1)         # speed constraints
            ubw += [8.33] + [inf] * (self.DYNAMICS_DIM - Para.HIS_ACT_ENCODING_DIM -2) + [0.] + [inf] * Para.HIS_ACT_ENCODING_DIM

            # Cost function
            # x[6]:devi_longi   x[7]:devi_lateral   x[9]:devi_v   x[8]:devi_phi
            # x[12]:action[steer]_last_time   x[13]:action[a]_last_time
            # x[2]:punish_yaw_rate   u[0]:punish_steer0   u[1]:punish_a_x0
            F_cost = Function('F_cost', [x, u], [0.8 * power(x[6], 2) + 0.5 * power(x[7], 2) +
                                                 0.1 * power(x[9], 2) + 30 * power(x[8] * np.pi / 180., 2) +
                                                 0.02 * power(x[2], 2) +
                                                 0.5 * power(u[0], 2) + 0.05 * power((x[12]-u[0]), 2) +
                                                 0.05 * power(u[1], 2) + 0.01 * power((x[13]-u[1]), 2)])
            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # load constraints and solve NLP
        r = S(lbx=vertcat(*lbw), ubx=vertcat(*ubw), x0=XO, lbg=vertcat(*lbg), ubg=vertcat(*ubg))
        state_all = np.array(r['x'])
        g_all = np.array(r['g'])
        state = np.zeros([self.horizon, self.DYNAMICS_DIM])
        control = np.zeros([self.horizon, self.ACTION_DIM])
        nt = self.DYNAMICS_DIM + self.ACTION_DIM  # total variable per step
        cost = np.array(r['f']).squeeze(0)

        # save trajectories
        for i in range(self.horizon):
            state[i] = state_all[nt * i: nt * (i + 1) - self.ACTION_DIM].reshape(-1)
            control[i] = state_all[nt * (i + 1) - self.ACTION_DIM: nt * (i + 1)].reshape(-1)
        return state, control, state_all, g_all, cost


class HierarchicalMpc(object):
    def __init__(self, logdir):
        # if self.task == 'left':
        #     self.policy = LoadPolicy('../utils/models/left/experiment-2021-03-15-16-39-00', 180000)
        # elif self.task == 'right':
        #     self.policy = LoadPolicy('G:\\env_build\\utils\\models\\right', 145000)
        # elif self.task == 'straight':
        #     self.policy = LoadPolicy('G:\\env_build\\utils\\models\\straight', 95000)

        self.logdir = logdir
        self.episode_counter = 0
        self.horizon = 25
        self.num_future_data = 25
        self.env = CrossroadEnd2endMixPI()
        self.obs, _ = self.env.reset()  # todo check
        self.task = self.env.training_task
        # self.ref_path = ReferencePath(self.task)
        self.stg = MultiPathGenerator()
        self.mpc_cal_timer = TimerStat()
        self.adp_cal_timer = TimerStat()
        self.recorder = Recorder()
        self.hist_posi = []

    def reset(self):
        self.obs, _ = self.env.reset()
        # self.stg = StaticTrajectoryGenerator_origin(mode='static_traj')
        self.recorder.reset()
        self.hist_posi = []

        if self.logdir is not None:
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))
            self.recorder.save(self.logdir)
            self.episode_counter += 1
            # if self.episode_counter >= 1:
            #     self.recorder.plot_mpc_rl(self.episode_counter-1,
            #                               self.logdir + '/episode{}/figs'.format(self.episode_counter-1), isshow=False)
        return self.obs

    def convert_vehs_to_abso(self, obs_rela):
        ego_infos, tracking_infos, encodings, veh_rela = obs_rela[:Para.EGO_ENCODING_DIM], \
                                              obs_rela[Para.EGO_ENCODING_DIM: Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM * (1 + self.num_future_data)], \
                                              obs_rela[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM * (1 + self.num_future_data): Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM * (1 + self.num_future_data) + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM + Para.REF_ENCODING_DIM + Para.HIS_ACT_ENCODING_DIM], \
                                              obs_rela[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM * (1 + self.num_future_data) + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM + Para.REF_ENCODING_DIM + Para.HIS_ACT_ENCODING_DIM:]
        ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_infos
        ego = np.array(([ego_x, ego_y] + [0] * (Para.PER_OTHER_INFO_DIM - 2)) * int(len(veh_rela) / Para.PER_OTHER_INFO_DIM), dtype=np.float32)
        vehs_abso = veh_rela + ego
        out = np.concatenate((ego_infos, tracking_infos, encodings, vehs_abso), axis=0)
        return out

    def step(self):
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        ADP_traj_return_value, MPC_traj_return_value = [], []
        action_total = []
        state_total = []

        with self.mpc_cal_timer:
            i = 0
            weight = [1.0, 1.0, 1.0]
            for path in self.path_list:
                self.env.set_traj(path)
                self.obs, _, _ = self.env._get_obs()
                mpc = ModelPredictiveControl(self.horizon, self.task, self.num_future_data)
                his_action = self.obs[Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.TRACK_ENCODING_DIM * self.num_future_data + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM + Para.REF_ENCODING_DIM: Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM + Para.TRACK_ENCODING_DIM * self.num_future_data + Para.LIGHT_ENCODING_DIM + Para.TASK_ENCODING_DIM + Para.REF_ENCODING_DIM + Para.HIS_ACT_ENCODING_DIM]
                state_all = np.array((list(self.obs[:Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM]) + list(his_action) + [0, 0]) * self.horizon +
                                      list(self.obs[:Para.EGO_ENCODING_DIM + Para.TRACK_ENCODING_DIM]) + list(his_action)).reshape((-1, 1))
                state, control, state_all, g_all, cost = mpc.mpc_solver(list(self.obs), state_all)    #todo check
                # state, control, state_all, g_all, cost = mpc.mpc_solver(list(self.convert_vehs_to_abso(self.obs)), state_all)
                state_total.append(state)
                if any(g_all < -1):
                    print('optimization fail')
                    mpc_action = np.array([0., -1.])
                    state_all = np.array((list(self.obs[:10]) + list(his_action) + [0, 0]) * self.horizon + list(self.obs[:10]) + list(his_action)).reshape((-1, 1))
                else:
                    state_all = np.array((list(self.obs[:10]) + list(his_action) + [0, 0]) * self.horizon + list(self.obs[:10]) + list(his_action)).reshape((-1, 1))
                    mpc_action = control[0]

                MPC_traj_return_value.append(weight[i] * cost.squeeze().tolist())
                i += 1
                action_total.append(mpc_action)

            MPC_traj_return_value = np.array(MPC_traj_return_value, dtype=np.float32)
            MPC_path_index = np.argmin(MPC_traj_return_value)                           # todo: minimize
            MPC_action = action_total[MPC_path_index]

        self.obs, rew, done, _ = self.env.step(MPC_action)
        self.render(MPC_traj_return_value, MPC_path_index, method='MPC')
        state = state_total[MPC_path_index]
        plt.plot([state[i][3] for i in range(1, self.horizon - 1)], [state[i][4] for i in range(1, self.horizon - 1)], 'r*')
        plt.pause(0.001)

        return done

    def render(self, MPC_traj_return_value, MPC_path_index, method='MPC'):
        extension = 40
        dotted_line_style = '--'
        solid_line_style = '-'

        plt.clf()
        ax = plt.axes([-0.05, -0.05, 1.1, 1.1], facecolor='white')
        ax.axis("equal")
        patches = []

        # ----------arrow--------------
        # plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_1 * 0.5 + 0.4, -Para.CROSSROAD_SIZE_LON / 2 - 10, 0, 3, color='b', zorder=1)
        # plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_1 * 0.5 + 0.4, -Para.CROSSROAD_SIZE_LON / 2 - 10 + 3, -0.5, 1, color='b', head_width=0.7, zorder=1)
        # plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_1 + Para.LANE_WIDTH_1 * 0.5, -Para.CROSSROAD_SIZE_LON / 2 - 10, 0, 4, color='b', head_width=0.7, zorder=1)
        # plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_1 + Para.LANE_WIDTH_1 * 1.5, -Para.CROSSROAD_SIZE_LON / 2 - 10, 0, 4, color='b', head_width=0.7, zorder=1)
        # plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_1 + Para.LANE_WIDTH_1 * 2.5 - 0.3, -Para.CROSSROAD_SIZE_LON / 2 - 10, 0, 3, color='b', zorder=1)
        # plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_1 + Para.LANE_WIDTH_1 * 2.5 - 0.3, -Para.CROSSROAD_SIZE_LON / 2 - 10 + 3, 0.5, 1, color='b', head_width=0.7, zorder=1)

        # green belt
        ax.add_patch(plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2,
                                    Para.OFFSET_R - Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 - Para.GREEN_BELT),
                                   extension, Para.GREEN_BELT, edgecolor='white', facecolor='green',
                                   angle=Para.ANGLE_R, linewidth=1, alpha=0.7, zorder=1))

        # ax.add_patch(plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R - Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 - Para.GREEN_BELT - Para.BIKE_LANE_WIDTH_1),
        #                            extension, Para.BIKE_LANE_WIDTH_1, edgecolor='white', facecolor='tomato',
        #                            angle=Para.ANGLE_R, linewidth=1, alpha=0.1, zorder=1))
        # ax.add_patch(plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R - Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 - Para.GREEN_BELT - Para.BIKE_LANE_WIDTH_1 - Para.PERSON_LANE_WIDTH_2),
        #                            extension, Para.PERSON_LANE_WIDTH_2, edgecolor='white', facecolor='silver',
        #                            angle=Para.ANGLE_R, linewidth=1, alpha=0.2, zorder=1))

        plt.plot(
            [-Para.CROSSROAD_SIZE_LAT / 2 - extension * cos(Para.ANGLE_L / 180 * pi), -Para.CROSSROAD_SIZE_LAT / 2],
            [Para.OFFSET_L + 0.2 - extension * sin(Para.ANGLE_L / 180 * pi), Para.OFFSET_L + 0.2], color='orange',
            zorder=1)
        plt.plot(
            [-Para.CROSSROAD_SIZE_LAT / 2 - extension * cos(Para.ANGLE_L / 180 * pi), -Para.CROSSROAD_SIZE_LAT / 2],
            [Para.OFFSET_L - 0.2 - extension * sin(Para.ANGLE_L / 180 * pi), Para.OFFSET_L - 0.2], color='orange',
            zorder=1)
        plt.plot(
            [Para.CROSSROAD_SIZE_LAT / 2 + extension * cos(Para.ANGLE_R / 180 * pi), Para.CROSSROAD_SIZE_LAT / 2],
            [Para.OFFSET_R + 0.2 + extension * sin(Para.ANGLE_R / 180 * pi), Para.OFFSET_R + 0.2], color='orange',
            zorder=1)
        plt.plot(
            [Para.CROSSROAD_SIZE_LAT / 2 + extension * cos(Para.ANGLE_R / 180 * pi), Para.CROSSROAD_SIZE_LAT / 2],
            [Para.OFFSET_R - 0.2 + extension * sin(Para.ANGLE_R / 180 * pi), Para.OFFSET_R - 0.2], color='orange',
            zorder=1)

        plt.plot([Para.OFFSET_U + 0.2, Para.OFFSET_U + 0.2],
                 [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2], color='orange', zorder=1)
        plt.plot([Para.OFFSET_U - 0.2, Para.OFFSET_U - 0.2],
                 [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2], color='orange', zorder=1)
        plt.plot([Para.OFFSET_D + 0.2, Para.OFFSET_D + 0.2],
                 [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2], color='orange', zorder=1)
        plt.plot([Para.OFFSET_D - 0.2, Para.OFFSET_D - 0.2],
                 [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2], color='orange', zorder=1)

        # Left out lane
        for i in range(1, Para.LANE_NUMBER_LAT_OUT + 3):
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.BIKE_LANE_WIDTH_2, Para.PERSON_LANE_WIDTH_2]
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension * cos(Para.ANGLE_L / 180 * pi),
                      -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L - extension * sin(Para.ANGLE_L / 180 * pi) + sum(lane_width_flag[:i]) / cos(
                         Para.ANGLE_L / 180 * pi), Para.OFFSET_L + sum(lane_width_flag[:i])],
                     linestyle=linestyle, color='black', linewidth=linewidth, zorder=1)

        # Left in lane
        for i in range(1, Para.LANE_NUMBER_LAT_IN + 3):
            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_2, Para.LANE_WIDTH_2, Para.LANE_WIDTH_2,
                               Para.BIKE_LANE_WIDTH_2, Para.PERSON_LANE_WIDTH_2]
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension * cos(Para.ANGLE_L / 180 * pi),
                      -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L - extension * sin(Para.ANGLE_L / 180 * pi) - sum(lane_width_flag[:i]) / cos(
                         Para.ANGLE_L / 180 * pi), Para.OFFSET_L - sum(lane_width_flag[:i])],
                     linestyle=linestyle, color='black', linewidth=linewidth, zorder=1)

        # Right out lane
        for i in range(1, Para.LANE_NUMBER_LAT_OUT + 4):
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.GREEN_BELT, Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                      Para.CROSSROAD_SIZE_LAT / 2 + extension * cos(Para.ANGLE_R / 180 * pi)],
                     [Para.OFFSET_R - sum(lane_width_flag[:i]),
                      Para.OFFSET_R + extension * sin(Para.ANGLE_R / 180 * pi) - sum(lane_width_flag[:i]) / cos(
                          Para.ANGLE_R / 180 * pi)],
                     linestyle=linestyle, color='black', linewidth=linewidth, zorder=1)

        # Right in lane
        for i in range(1, Para.LANE_NUMBER_LAT_IN + 3):
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                      Para.CROSSROAD_SIZE_LAT / 2 + extension * cos(Para.ANGLE_R / 180 * pi)],
                     [Para.OFFSET_R + sum(lane_width_flag[:i]),
                      Para.OFFSET_R + extension * sin(Para.ANGLE_R / 180 * pi) + sum(lane_width_flag[:i]) / cos(
                          Para.ANGLE_R / 180 * pi)],
                     linestyle=linestyle, color='black', linewidth=linewidth, zorder=1)

        # Up in lane
        for i in range(1, Para.LANE_NUMBER_LON_IN_U + 3):
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN_U else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LON_IN_U else 1
            plt.plot([Para.OFFSET_U - sum(lane_width_flag[:i]), Para.OFFSET_U - sum(lane_width_flag[:i])],
                     [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth, zorder=1)

        # Up out lane
        for i in range(1, Para.LANE_NUMBER_LON_OUT + 3):
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.BIKE_LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH_1]
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
            plt.plot([Para.OFFSET_U + sum(lane_width_flag[:i]), Para.OFFSET_U + sum(lane_width_flag[:i])],
                     [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth, zorder=1)

        # Down in lane
        for i in range(1, Para.LANE_NUMBER_LON_IN_D + 3):
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN_D else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LON_IN_D else 1
            plt.plot([Para.OFFSET_D + sum(lane_width_flag[:i]), Para.OFFSET_D + sum(lane_width_flag[:i])],
                     [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth, zorder=1)

        # Down out lane
        for i in range(1, Para.LANE_NUMBER_LON_OUT + 3):
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.BIKE_LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH_2]
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
            plt.plot([Para.OFFSET_D - sum(lane_width_flag[:i]), Para.OFFSET_D - sum(lane_width_flag[:i])],
                     [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth, zorder=1)

        # Oblique
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2,
                  Para.OFFSET_U - Para.LANE_NUMBER_LON_IN_U * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH_1 - Para.PERSON_LANE_WIDTH_2],
                 [
                     Para.OFFSET_L + Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH_2 + Para.PERSON_LANE_WIDTH_2,
                     Para.CROSSROAD_SIZE_LON / 2],
                 color='black', linewidth=1, zorder=1)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2,
                  Para.OFFSET_D - Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH_1 - Para.PERSON_LANE_WIDTH_2],
                 [
                     Para.OFFSET_L - Para.LANE_NUMBER_LAT_IN * Para.LANE_WIDTH_2 - Para.BIKE_LANE_WIDTH_2 - Para.PERSON_LANE_WIDTH_2,
                     -Para.CROSSROAD_SIZE_LON / 2],
                 color='black', linewidth=1, zorder=1)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                  Para.OFFSET_D + Para.LANE_NUMBER_LON_IN_D * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH_1 + Para.PERSON_LANE_WIDTH_2],
                 [
                     Para.OFFSET_R - Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 - Para.GREEN_BELT - Para.BIKE_LANE_WIDTH_1 - Para.PERSON_LANE_WIDTH_2,
                     -Para.CROSSROAD_SIZE_LON / 2],
                 color='black', linewidth=1, zorder=1)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                  Para.OFFSET_U + Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH_1 + Para.PERSON_LANE_WIDTH_1],
                 [
                     Para.OFFSET_R + Para.LANE_NUMBER_LAT_IN * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH_1 + Para.PERSON_LANE_WIDTH_2,
                     Para.CROSSROAD_SIZE_LON / 2],
                 color='black', linewidth=1, zorder=1)

        # stop line
        lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                           Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]  # Down
        plt.plot([Para.OFFSET_D, Para.OFFSET_D + sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN_D])],
                 [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2], color='gray', zorder=2)
        lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                           Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]  # Up
        plt.plot([-sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN_U]) + Para.OFFSET_U, Para.OFFSET_U],
                 [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2], color='gray', zorder=2)
        lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_2, Para.LANE_WIDTH_2, Para.LANE_WIDTH_2,
                           Para.BIKE_LANE_WIDTH_2, Para.PERSON_LANE_WIDTH_2]
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:Para.LANE_NUMBER_LAT_IN])],
                 color='gray', zorder=2)  # left
        lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                           Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2], [Para.OFFSET_R,
                                                                              Para.OFFSET_R + sum(lane_width_flag[
                                                                                                  :Para.LANE_NUMBER_LAT_IN])],
                 color='gray', zorder=2)

        v_light = self.env.light_phase
        light_line_width = 2
        if v_light == 0 or v_light == 1:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'green', 'green', 'red', 'red'
        elif v_light == 2:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'orange', 'orange', 'red', 'red'
        elif v_light == 3:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'red'
        elif v_light == 4:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'green'
        elif v_light == 5:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'orange'
        elif v_light == 6:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'red'
        elif v_light == 7:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'green', 'red'
        elif v_light == 8:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'orange', 'red'
        else:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'red'

        lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                           Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]  # Down
        plt.plot([Para.OFFSET_D, Para.OFFSET_D + sum(lane_width_flag[:1])],
                 [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                 color=v_color_1, linewidth=light_line_width, zorder=3)
        plt.plot([Para.OFFSET_D + sum(lane_width_flag[:1]), Para.OFFSET_D + sum(lane_width_flag[:3])],
                 [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                 color=v_color_2, linewidth=light_line_width, zorder=3)
        plt.plot([Para.OFFSET_D + sum(lane_width_flag[:3]), Para.OFFSET_D + sum(lane_width_flag[:4])],
                 [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                 color='green', linewidth=light_line_width, zorder=3)

        lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                           Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]  # Up
        plt.plot([-sum(lane_width_flag[:1]) + Para.OFFSET_U, Para.OFFSET_U],
                 [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                 color=v_color_1, linewidth=light_line_width, zorder=3)
        plt.plot([-sum(lane_width_flag[:2]) + Para.OFFSET_U, -sum(lane_width_flag[:1]) + Para.OFFSET_U],
                 [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                 color=v_color_2, linewidth=light_line_width, zorder=3)
        plt.plot([-sum(lane_width_flag[:3]) + Para.OFFSET_U, -sum(lane_width_flag[:2]) + Para.OFFSET_U],
                 [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                 color='green', linewidth=light_line_width, zorder=3)

        lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_2, Para.LANE_WIDTH_2, Para.LANE_WIDTH_2,
                           Para.BIKE_LANE_WIDTH_2, Para.PERSON_LANE_WIDTH_2]  # left
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:2])],
                 color=h_color_1, linewidth=light_line_width, zorder=3)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L - sum(lane_width_flag[:2]),
                  Para.OFFSET_L - sum(lane_width_flag[:3]) - lane_width_flag[3] / 2],
                 color=h_color_2, linewidth=light_line_width, zorder=3)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L - sum(lane_width_flag[:3]) - lane_width_flag[3] / 2,
                  Para.OFFSET_L - sum(lane_width_flag[:4])],
                 color='green', linewidth=light_line_width, zorder=3)

        lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                           Para.BIKE_LANE_WIDTH_1, Para.PERSON_LANE_WIDTH_2]  # right
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R,
                  Para.OFFSET_R + sum(lane_width_flag[:2])],
                 color=h_color_1, linewidth=light_line_width, zorder=3)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R + sum(lane_width_flag[:2]),
                  Para.OFFSET_R + sum(lane_width_flag[:3])],
                 color=h_color_2, linewidth=light_line_width, zorder=3)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R + sum(lane_width_flag[:3]),
                  Para.OFFSET_R + sum(lane_width_flag[:4])],
                 color='green', linewidth=light_line_width, zorder=3)

        # zebra crossing
        j1, j2 = 0.5, 6.75
        for ii in range(18):
            if ii <= 3:
                continue
            ax.add_patch(plt.Rectangle(
                (-Para.CROSSROAD_SIZE_LON / 2 + j1 + 0.6 + ii * 1.6, -Para.CROSSROAD_SIZE_LON / 2 + 0.5), 0.8, 4,
                color='lightgray', alpha=0.5, zorder=1))
            ii += 1
        for ii in range(17):
            if ii <= 3:
                continue
            ax.add_patch(plt.Rectangle(
                (-Para.CROSSROAD_SIZE_LON / 2 + j1 + 1.6 + ii * 1.6, Para.CROSSROAD_SIZE_LON / 2 - 0.5 - 4), 0.8, 4,
                color='lightgray', alpha=0.5, zorder=1))
            ii += 1
        for ii in range(21):
            if ii <= 3:
                continue
            ax.add_patch(plt.Rectangle(
                (-Para.CROSSROAD_SIZE_LAT / 2 + 0.5, Para.CROSSROAD_SIZE_LAT / 2 - j2 + 10.5 - ii * 1.6), 4, 0.8,
                color='lightgray', alpha=0.5, zorder=1))
            ii += 1
        for ii in range(21):
            if ii <= 3:
                continue
            ax.add_patch(plt.Rectangle(
                (Para.CROSSROAD_SIZE_LAT / 2 - 0.5 - 4, Para.CROSSROAD_SIZE_LAT / 2 - j2 + 10.5 - ii * 1.6), 4, 0.8,
                color='lightgray', alpha=0.5, zorder=1))
            ii += 1

        def is_in_plot_area(x, y, tolerance=5):
            if -Para.CROSSROAD_SIZE_LAT / 2 - extension + tolerance < x < Para.CROSSROAD_SIZE_LAT / 2 + extension - tolerance and \
                    -Para.CROSSROAD_SIZE_LON / 2 - extension + tolerance < y < Para.CROSSROAD_SIZE_LON / 2 + extension - tolerance:
                return True
            else:
                return False

        def draw_rotate_rec(type, x, y, a, l, w, color, linestyle='-', patch=False):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            if patch:
                if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    item_color = 'purple'
                elif type == 'DEFAULT_PEDTYPE':
                    item_color = 'lime'
                else:
                    item_color = 'gainsboro'
                patches.append(
                    plt.Rectangle((x + LU_x, y + LU_y), w, l, edgecolor=item_color, facecolor=item_color, linewidth=0.8,
                                  angle=-(90 - a)))
            else:
                patches.append(matplotlib.patches.Rectangle((-l / 2 + x, -w / 2 + y),
                                                            width=l, height=w,
                                                            # fill=False,
                                                            facecolor='white',
                                                            edgecolor=color,
                                                            linestyle=linestyle,
                                                            linewidth=1.0,
                                                            transform=Affine2D().rotate_deg_around(*(x, y), a)))

        def draw_rotate_batch_rec(x, y, a, l, w, type):
            for i in range(len(x)):
                plot_phi_line(type[i], x[i], y[i], a[i], color='k')
                patches.append(matplotlib.patches.Rectangle((-l[i] / 2 + x[i], -w[i] / 2 + y[i]),
                                                            width=l[i], height=w[i],
                                                            # fill=False,
                                                            facecolor='white',
                                                            edgecolor='k',
                                                            linewidth=1.0,
                                                            transform=Affine2D().rotate_deg_around(*(x[i], y[i]),
                                                                                                   a[i])))

        def plot_phi_line(type, x, y, phi, color):
            if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                line_length = 1.5
            elif type == 'DEFAULT_PEDTYPE':
                line_length = 0.5
            else:
                line_length = 3.2
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                             y + line_length * sin(phi * pi / 180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5, zorder=3.5)

        # plot others
        filted_all_other = [item for item in self.env.all_other if is_in_plot_area(item['x'], item['y'])]
        other_xs = np.array([item['x'] for item in filted_all_other], np.float32)
        other_ys = np.array([item['y'] for item in filted_all_other], np.float32)
        other_as = np.array([item['phi'] for item in filted_all_other], np.float32)
        other_ls = np.array([item['l'] for item in filted_all_other], np.float32)
        other_ws = np.array([item['w'] for item in filted_all_other], np.float32)
        other_type = np.array([item['type'] for item in filted_all_other])

        draw_rotate_batch_rec(other_xs, other_ys, other_as, other_ls, other_ws, other_type)

        # plot interested others
        # if weights is not None:
        #     assert weights.shape == (self.other_number,), print(weights.shape)
        # index_top_k_in_weights = weights.argsort()[-4:][::-1]
        for i in range(len(self.env.interested_other)):
            item = self.env.interested_other[i]
            item_mask = item['exist']
            item_x = item['x']
            item_y = item['y']
            item_phi = item['phi']
            item_l = item['l']
            item_w = item['w']
            item_type = item['type']
            if is_in_plot_area(item_x, item_y):
                # plot_phi_line(item_type, item_x, item_y, item_phi, 'black')
                draw_rotate_rec(item_type, item_x, item_y, item_phi, item_l, item_w, color='g', linestyle=':',
                                patch=True)
                plt.text(item_x, item_y, str(item_mask)[0])
            # if i in index_top_k_in_weights:
            #     plt.text(item_x, item_y, "{:.2f}".format(weights[i]), color='red', fontsize=15)

        ego_v_x = self.env.ego_dynamics['v_x']
        ego_v_y = self.env.ego_dynamics['v_y']
        ego_r = self.env.ego_dynamics['r']
        ego_x = self.env.ego_dynamics['x']
        ego_y = self.env.ego_dynamics['y']
        ego_phi = self.env.ego_dynamics['phi']
        ego_l = self.env.ego_dynamics['l']
        ego_w = self.env.ego_dynamics['w']
        ego_alpha_f = self.env.ego_dynamics['alpha_f']
        ego_alpha_r = self.env.ego_dynamics['alpha_r']
        alpha_f_bound = self.env.ego_dynamics['alpha_f_bound']
        alpha_r_bound = self.env.ego_dynamics['alpha_r_bound']
        r_bound = self.env.ego_dynamics['r_bound']

        plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'fuchsia')
        draw_rotate_rec('self_car', ego_x, ego_y, ego_phi, ego_l, ego_w, 'fuchsia')
        self.hist_posi.append((ego_x, ego_y))

        # plot history
        xs = [pos[0] for pos in self.hist_posi]
        ys = [pos[1] for pos in self.hist_posi]
        plt.scatter(np.array(xs), np.array(ys), color='fuchsia', alpha=0.1)


        # plot future data
        # todo 参考路径不对应 估计是初始化多次
        # obs_abso = self.convert_vehs_to_abso(self.obs)
        tracking_info = self.obs[
                        self.env.ego_info_dim:self.env.ego_info_dim + Para.TRACK_ENCODING_DIM * (self.num_future_data + 1)]
        future_path = tracking_info[Para.TRACK_ENCODING_DIM:]
        for i in range(self.num_future_data):
            delta_x, delta_y, delta_phi, _ = future_path[i * Para.TRACK_ENCODING_DIM:
                                                      (i + 1) * Para.TRACK_ENCODING_DIM]
            path_x, path_y, path_phi = delta_x, delta_y, ego_phi - delta_phi
            plt.plot(path_x, path_y, 'g.')
            # plot_phi_line(path_x, path_y, path_phi, 'g')

        # delta_, _, _ = tracking_info[:3]
        # indexs, points = self.ref_path._find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
        # path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
        # # plt.plot(path_x, path_y, 'g.')
        # delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

        # plot real time traj
        color = ['blue', 'coral', 'darkcyan', 'pink']
        # for i, item in enumerate(self.ref_path.path_list['green']):  #todo
        #
        #     print(self.ref_path.path_list['green'] == path_list)
        #     if REF_ENCODING[i] == self.ref_path.ref_encoding:
        #         plt.plot(item[0], item[1], color=color[i], alpha=1.0)
        #     else:
        #         plt.plot(item[0], item[1], color=color[i], alpha=0.3)

        for i, path in enumerate(self.path_list):
            # for i, (path_x, path_y, _, _) in enumerate(path.path):
            # print('画图路径长为', len(path.path[0]))
            if i == MPC_path_index:
                plt.plot(path.path[0], path.path[1], color=color[i], alpha=1.0)
            else:
                plt.plot(path.path[0], path.path[1], color=color[i], alpha=0.3)


                # indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
                # path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                # plt.plot(path_x, path_y,  color=color[i])

        # plot ego dynamics
        text_x, text_y_start = -100, 60
        ge = iter(range(0, 1000, 4))
        plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
        plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
        # plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
        # plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
        # plt.text(text_x, text_y_start - next(ge), 'delta_: {:.2f}m'.format(delta_))
        # plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
        # plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
        plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
        # plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
        # plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))
        plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
        # plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
        plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))

        plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
                                                                                                alpha_f_bound))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
                                                                                                alpha_r_bound))
        if self.env.action is not None:
            steer, a_x = self.env.action[0], self.env.action[1]
            plt.text(text_x, text_y_start - next(ge),
                     r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
            plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

        text_x, text_y_start = 70, 60
        ge = iter(range(0, 1000, 4))

        # done info
        plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.env.done_type))

        # reward info
        if self.env.reward_info is not None:
            for key, val in self.env.reward_info.items():
                plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

        text_x, text_y_start = 25, -30
        ge = iter(range(0, 1000, 6))
        plt.text(text_x, text_y_start - next(ge), 'MPC', fontsize=14, color='r', fontstyle='italic')
        color = ['blue', 'coral', 'darkcyan']
        if MPC_traj_return_value is not None:
            for i, value in enumerate(MPC_traj_return_value):
                if i == MPC_path_index:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=14,
                             color=color[i], fontstyle='italic')
                else:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=12,
                             color=color[i], fontstyle='italic')

        ax.add_collection(PatchCollection(patches, match_original=True, zorder=4))
        plt.show()
        plt.pause(0.001)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    hier_decision = HierarchicalMpc(logdir)
    for i in range(15):
        done = 0
        for _ in range(3):
            done = hier_decision.step()
            if done:
                break
        # np.save('mpc.npy', np.array(hier_decision.data2plot))
        hier_decision.reset()


def plot_data(epi_num, logdir):
    recorder = Recorder()
    recorder.load(logdir)
    recorder.plot_mpc_rl(epi_num, logdir, sample=True)


if __name__ == '__main__':
    main()
    # plot_data(epi_num=6, logdir='./results/2021-03-17-22-33-09')  # 6 or 3
