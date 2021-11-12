#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: dynamics_and_models.py
# =====================================

from math import pi, cos, sin

import bezier
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import logical_and

# gym.envs.user_defined.toyota_env.
from endtoend_env_utils import *

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
                                   C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
                                   a=1.19,  # distance from CG to front axle [m]
                                   b=1.46,  # distance from CG to rear axle [m]
                                   mass=1520.,  # mass [kg]
                                   I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        phi = phi * np.pi / 180.
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
        miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
        g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x))
        F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
        miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
        alpha_f = tf.atan((v_y + a * r) / (v_x + 1e-8)) - steer
        alpha_r = tf.atan((v_y - b * r) / (v_x + 1e-8))

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * tf.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
                      x + tau * (v_x * tf.cos(phi) - v_y * tf.sin(phi)),
                      y + tau * (v_x * tf.sin(phi) + v_y * tf.cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return tf.stack(next_state, 1), tf.stack([alpha_f, alpha_r, miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


class EnvironmentModel(object):  # all tensors
    def __init__(self, future_point_num=25, mode='training'):
        self.mode = mode
        self.future_point_num = future_point_num
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obses = None
        self.actions = None
        self.reward_info = None
        self.veh_num = Para.MAX_VEH_NUM
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.track_info_dim = Para.TRACK_ENCODING_DIM
        self.light_info_dim = Para.LIGHT_ENCODING_DIM
        self.task_info_dim = Para.TASK_ENCODING_DIM
        self.ref_info_dim = Para.REF_ENCODING_DIM
        self.other_number = sum([self.veh_num, self.bike_num, self.person_num])
        self.per_other_info_dim = Para.PER_OTHER_INFO_DIM
        self.future_point_num = future_point_num
        self.per_path_info_dim = Para.PER_PATH_INFO_DIM
        self.other_start_dim = sum([self.ego_info_dim, self.track_info_dim, self.future_point_num * self.per_path_info_dim,
                                    self.light_info_dim, self.task_info_dim, self.ref_info_dim])
        self.steer_store = []
        self.ref = ReferencePath(task='left', green_or_red='green')
        self.path_all = self.ref.path_all
        self.batch_path = None
        self.path_len = None

    def reset(self, obses, ref_index=None):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None
        self.steer_store = []
        self.batch_path = tf.gather(self.path_all, indices=ref_index)
        self.path_len = tf.shape(self.batch_path)[-1]

    def rollout_out(self, actions):  # ref_points [#batch, 4]
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)

            self.obses = self.compute_next_obses(self.obses, self.actions)

            rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2line4real, _ \
                = self.compute_rewards(self.obses, self.actions)

        return self.obses, rewards, punish_term_for_training, real_punish_term, veh2veh4real, \
               veh2road4real, veh2line4real

    # def ss(self, obses, actions, lam=0.1):
    #     actions = self._action_transformation_for_end2end(actions)
    #     next_obses = self.compute_next_obses(obses, actions)
    #     ego_infos, veh_infos = obses[:, :self.ego_info_dim], \
    #                            obses[:, self.ego_info_dim + self.per_tracking_info_dim * (
    #                                    self.num_future_data + 1):]
    #     next_ego_infos, next_veh_infos = next_obses[:, :self.ego_info_dim], \
    #                                      next_obses[:, self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                self.num_future_data + 1):]
    #     ego_lws = (L - W) / 2.
    #     ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.),
    #                                dtype=tf.float32), \
    #                        tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
    #     ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
    #                       tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
    #
    #     next_ego_front_points = tf.cast(next_ego_infos[:, 3] + ego_lws * tf.cos(next_ego_infos[:, 5] * np.pi / 180.),
    #                                dtype=tf.float32), \
    #                        tf.cast(next_ego_infos[:, 4] + ego_lws * tf.sin(next_ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
    #     next_ego_rear_points = tf.cast(next_ego_infos[:, 3] - ego_lws * tf.cos(next_ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
    #                       tf.cast(next_ego_infos[:, 4] - ego_lws * tf.sin(next_ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
    #
    #     veh2veh4real = tf.zeros_like(veh_infos[:, 0])
    #     for veh_index in range(self.veh_num):
    #         vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
    #         ego2veh_dist = tf.sqrt(tf.square(ego_infos[:, 3] - vehs[:, 0]) + tf.square(ego_infos[:, 4] - vehs[:, 1]))
    #
    #         next_vehs = next_veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
    #
    #         veh_lws = (L - W) / 2.
    #         veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
    #                            tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
    #         veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
    #                           tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
    #
    #         next_veh_front_points = tf.cast(next_vehs[:, 0] + veh_lws * tf.cos(next_vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
    #                            tf.cast(next_vehs[:, 1] + veh_lws * tf.sin(next_vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
    #         next_veh_rear_points = tf.cast(next_vehs[:, 0] - veh_lws * tf.cos(next_vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
    #                           tf.cast(next_vehs[:, 1] - veh_lws * tf.sin(next_vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
    #
    #         for ego_point in [(ego_front_points, next_ego_front_points), (ego_rear_points, next_ego_rear_points)]:
    #             for veh_point in [(veh_front_points, next_veh_front_points), (veh_rear_points, next_veh_rear_points)]:
    #                 veh2veh_dist = tf.sqrt(
    #                     tf.square(ego_point[0][0] - veh_point[0][0]) + tf.square(ego_point[0][1] - veh_point[0][1]))
    #                 next_veh2veh_dist = tf.sqrt(
    #                     tf.square(ego_point[1][0] - veh_point[1][0]) + tf.square(ego_point[1][1] - veh_point[1][1]))
    #                 next_g = next_veh2veh_dist - 2.5
    #                 g = veh2veh_dist - 2.5
    #                 veh2veh4real += tf.where(logical_and(next_g - (1-lam)*g < 0, ego2veh_dist < 10), tf.square(next_g - (1-lam)*g),
    #                                          tf.zeros_like(veh_infos[:, 0]))
    #     return veh2veh4real

    def compute_rewards(self, obses, actions):
        # obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_future_point, obses_light, obses_task, obses_ref, obses_other = self._split_all(obses)

        with tf.name_scope('compute_reward') as scope:
            veh_infos = tf.stop_gradient(obses_other)

            steers, a_xs = actions[:, 0], actions[:, 1]
            # rewards related to action
            # if len(self.steer_store) > 2:
            #     steers_1st_order = (self.steer_store[-1] - self.steer_store[-2]) * self.base_frequency
            #     steers_2st_order = (self.steer_store[-1] - 2 * self.steer_store[-2] + self.steer_store[-3]) * (
            #                 self.base_frequency ** 2)
            # elif len(self.steer_store) == 2:
            #     steers_1st_order = (self.steer_store[-1] - self.steer_store[-2]) * self.base_frequency
            #     steers_2st_order = tf.zeros_like(steers)
            # else:
            #     steers_1st_order = tf.zeros_like(steers)
            #     steers_2st_order = tf.zeros_like(steers)
            punish_steer = -tf.square(steers)  # - tf.square(steers_1st_order) - tf.square(steers_2st_order) todo
            punish_a_x = -tf.square(a_xs)

            # rewards related to ego stability
            punish_yaw_rate = -tf.square(obses_ego[:, 2])

            # rewards related to tracking error
            devi_lon = -tf.square(obses_track[:, 0])
            devi_lat = -tf.square(obses_track[:, 1])
            devi_phi = -tf.cast(tf.square(obses_track[:, 2] * np.pi / 180.), dtype=tf.float32)
            devi_v = -tf.square(obses_track[:, 3])

            # rewards related to veh2veh collision
            ego_lws = (Para.L - Para.W) / 2.
            ego_front_points = tf.cast(obses_ego[:, 3] + ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32), \
                               tf.cast(obses_ego[:, 4] + ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32)
            ego_rear_points = tf.cast(obses_ego[:, 3] - ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32), \
                              tf.cast(obses_ego[:, 4] - ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32)
            veh2veh4real = tf.zeros_like(veh_infos[:, 0])
            veh2veh4training = tf.zeros_like(veh_infos[:, 0])

            for veh_index in range(self.veh_num):
                vehs = veh_infos[:, veh_index * self.per_other_info_dim:(veh_index + 1) * self.per_other_info_dim]
                veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
                veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1]))
                        veh2veh4training += tf.where(veh2veh_dist-3.5 < 0, tf.square(veh2veh_dist-3.5), tf.zeros_like(veh_infos[:, 0]))
                        veh2veh4real += tf.where(veh2veh_dist-2.5 < 0, tf.square(veh2veh_dist-2.5), tf.zeros_like(veh_infos[:, 0]))

            veh2road4real = tf.zeros_like(veh_infos[:, 0])
            veh2road4training = tf.zeros_like(veh_infos[:, 0])
            # if self.task == 'left':
            #     for ego_point in [ego_front_points, ego_rear_points]:
            #         veh2road4training += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE/2, ego_point[0] < 1),
            #                              tf.square(ego_point[0]-1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE/2, Para.LANE_WIDTH-ego_point[0] < 1),
            #                              tf.square(Para.LANE_WIDTH-ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[0] < 0, Para.LANE_WIDTH*Para.LANE_NUMBER - ego_point[1] < 1),
            #                              tf.square(Para.LANE_WIDTH*Para.LANE_NUMBER - ego_point[1] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[0] < -Para.CROSSROAD_SIZE/2, ego_point[1] - 0 < 1),
            #                              tf.square(ego_point[1] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))
            #
            #         veh2road4real += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE/2, ego_point[0] < 1),
            #                              tf.square(ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE/2, Para.LANE_WIDTH - ego_point[0] < 1),
            #                              tf.square(Para.LANE_WIDTH - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(logical_and(ego_point[0] < -Para.CROSSROAD_SIZE/2, Para.LANE_WIDTH*Para.LANE_NUMBER - ego_point[1] < 1),
            #                              tf.square(Para.LANE_WIDTH*Para.LANE_NUMBER - ego_point[1] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(logical_and(ego_point[0] < -Para.CROSSROAD_SIZE/2, ego_point[1] - 0 < 1),
            #                              tf.square(ego_point[1] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))
            # elif self.task == 'straight':
            #     for ego_point in [ego_front_points, ego_rear_points]:
            #         veh2road4training += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE/2, ego_point[0] - Para.LANE_WIDTH < 1),
            #                              tf.square(ego_point[0] - Para.LANE_WIDTH -1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE/2, 2*Para.LANE_WIDTH-ego_point[0] < 1),
            #                              tf.square(2*Para.LANE_WIDTH-ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[1] > Para.CROSSROAD_SIZE/2, Para.LANE_WIDTH*Para.LANE_NUMBER - ego_point[0] < 1),
            #                              tf.square(Para.LANE_WIDTH*Para.LANE_NUMBER - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[1] > Para.CROSSROAD_SIZE/2, ego_point[0] - 0 < 1),
            #                              tf.square(ego_point[0] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))
            #
            #         veh2road4real += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2, ego_point[0]-Para.LANE_WIDTH < 1),
            #                                       tf.square(ego_point[0]-Para.LANE_WIDTH - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(
            #             logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2, 2 * Para.LANE_WIDTH - ego_point[0] < 1),
            #             tf.square(2 * Para.LANE_WIDTH - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(
            #             logical_and(ego_point[1] > Para.CROSSROAD_SIZE / 2, Para.LANE_WIDTH * Para.LANE_NUMBER - ego_point[0] < 1),
            #             tf.square(Para.LANE_WIDTH * Para.LANE_NUMBER - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(logical_and(ego_point[1] > Para.CROSSROAD_SIZE / 2, ego_point[0] - 0 < 1),
            #                                       tf.square(ego_point[0] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))
            # else:
            #     assert self.task == 'right'
            #     for ego_point in [ego_front_points, ego_rear_points]:
            #         veh2road4training += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE/2, ego_point[0] - 2*Para.LANE_WIDTH < 1),
            #                              tf.square(ego_point[0] - 2*Para.LANE_WIDTH-1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[1] < -Para.CROSSROAD_SIZE/2, Para.LANE_NUMBER*Para.LANE_WIDTH-ego_point[0] < 1),
            #                              tf.square(Para.LANE_NUMBER*Para.LANE_WIDTH-ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[0] > Para.CROSSROAD_SIZE/2, 0 - ego_point[1] < 1),
            #                              tf.square(0 - ego_point[1] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4training += tf.where(logical_and(ego_point[0] > Para.CROSSROAD_SIZE/2, ego_point[1] - (-Para.LANE_WIDTH*Para.LANE_NUMBER) < 1),
            #                              tf.square(ego_point[1] - (-Para.LANE_WIDTH*Para.LANE_NUMBER) - 1), tf.zeros_like(veh_infos[:, 0]))
            #
            #         veh2road4real += tf.where(
            #             logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2, ego_point[0] - 2 * Para.LANE_WIDTH < 1),
            #             tf.square(ego_point[0] - 2 * Para.LANE_WIDTH - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(
            #             logical_and(ego_point[1] < -Para.CROSSROAD_SIZE / 2, Para.LANE_NUMBER * Para.LANE_WIDTH - ego_point[0] < 1),
            #             tf.square(Para.LANE_NUMBER * Para.LANE_WIDTH - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(logical_and(ego_point[0] > Para.CROSSROAD_SIZE / 2, 0 - ego_point[1] < 1),
            #                                       tf.square(0 - ego_point[1] - 1), tf.zeros_like(veh_infos[:, 0]))
            #         veh2road4real += tf.where(
            #             logical_and(ego_point[0] > Para.CROSSROAD_SIZE / 2, ego_point[1] - (-Para.LANE_WIDTH * Para.LANE_NUMBER) < 1),
            #             tf.square(ego_point[1] - (-Para.LANE_WIDTH * Para.LANE_NUMBER) - 1), tf.zeros_like(veh_infos[:, 0]))

            veh2line4real = tf.zeros_like(veh_infos[:, 0])
            veh2line4training = tf.zeros_like(veh_infos[:, 0])
            # for task, task_idx in TASK_DICT.items():  # choose task
            #     if task == 'left':
            #         stop_point = tf.constant([0.5 * Para.LANE_WIDTH], dtype=tf.float32), tf.constant([-Para.CROSSROAD_SIZE / 2], dtype=tf.float32)
            #     elif task == 'straight':
            #         stop_point = tf.constant([1.5 * Para.LANE_WIDTH], dtype=tf.float32), tf.constant([-Para.CROSSROAD_SIZE / 2], dtype=tf.float32)
            #     else:
            #         stop_point = tf.constant([2.5 * Para.LANE_WIDTH], dtype=tf.float32), tf.constant([-Para.CROSSROAD_SIZE / 2], dtype=tf.float32)
            #
            #     if not self.light_flag:
            #         self.light_cond = logical_and(light_infos[:, 0] != 0., obses_ego[:, 4] < -Para.CROSSROAD_SIZE / 2 - 2)
            #
            #     if task != 'right':  # right turn always has green light
            #         for ego_point in [ego_front_points, ego_rear_points]:
            #             veh2line_dist = tf.sqrt(tf.square(ego_point[0] - stop_point[0]) + tf.square(ego_point[1] - stop_point[1]))
            #             veh2line4real_temp = tf.where(veh2line_dist - 1.0 < 0, tf.square(veh2line_dist - 1.0), tf.zeros_like(veh_infos[:, 0]))
            #             veh2line4real_pick = tf.where(self.light_cond, veh2line4real_temp, tf.zeros_like(veh_infos[:, 0]))
            #             veh2line4real += tf.where(task_infos == task_idx, veh2line4real_pick, tf.zeros_like(veh_infos[:, 0]))

            rewards = 0.01 * devi_v + 0.8 * devi_lon + 0.8 * devi_lat + 30 * devi_phi + 0.02 * punish_yaw_rate + \
                      5 * punish_steer + 0.05 * punish_a_x
            punish_term_for_training = veh2veh4training + veh2road4training + veh2line4real
            real_punish_term = veh2veh4real + veh2road4real + veh2line4real

            reward_dict = dict(punish_steer=punish_steer,
                               punish_a_x=punish_a_x,
                               punish_yaw_rate=punish_yaw_rate,
                               devi_v=devi_v,
                               devi_lon=devi_lon,
                               devi_lat=devi_lat,
                               devi_phi=devi_phi,
                               scaled_punish_steer=5 * punish_steer,
                               scaled_punish_a_x=0.05 * punish_a_x,
                               scaled_punish_yaw_rate=0.02 * punish_yaw_rate,
                               scaled_devi_v=0.01 * devi_v,
                               scaled_devi_lon=0.8 * devi_lon,
                               scaled_devi_lat=0.8 * devi_lat,
                               scaled_devi_phi=30 * devi_phi,
                               veh2veh4training=veh2veh4training,
                               veh2road4training=veh2road4training,
                               veh2veh4real=veh2veh4real,
                               veh2road4real=veh2road4real,
                               veh2line4real=veh2line4real
                               )

            return rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real,veh2line4real, reward_dict

    def compute_next_obses(self, obses, actions):
        # obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_future_point, obses_light, obses_task, obses_ref, obses_other = self._split_all(obses)
        obses_other = tf.stop_gradient(obses_other)
        next_obses_ego = self._ego_predict(obses_ego, actions)
        next_obses_track = self._compute_next_track_info_vectorized(next_obses_ego)
        next_obses_other = self._other_predict(obses_other)
        next_obses = tf.concat([next_obses_ego, next_obses_track, obses_light, obses_task, obses_ref, next_obses_other],
                               axis=-1)
        # next_obses = self._convert_to_rela(next_obses)
        return next_obses

    def _compute_next_track_info(self, next_ego_infos, ref_points):
        ego_vxs, ego_vys, ego_rs, ego_xs, ego_ys, ego_phis = [next_ego_infos[:, i] for i in range(self.ego_info_dim)]
        ref_xs, ref_ys, ref_phis, ref_vs = [ref_points[:, i] for i in range(4)]
        ref_phis_rad = ref_phis * np.pi / 180
        a, b, c = (ref_xs, ref_ys), (ref_xs + tf.cos(ref_phis_rad), ref_ys + tf.sin(ref_phis_rad)), (ego_xs, ego_ys)
        dist_a2c = tf.sqrt(tf.square(ego_xs - ref_xs) + tf.square(ego_ys - ref_ys))
        dist_c2line = tf.abs(tf.sin(ref_phis_rad) * ego_xs - tf.cos(ref_phis_rad) * ego_ys
                             - tf.sin(ref_phis_rad) * ref_xs + tf.cos(ref_phis_rad) * ref_ys)
        dist_longdi = tf.sqrt(tf.abs(tf.square(dist_a2c) - tf.square(dist_c2line)))
        signed_dist_lateral = tf.where(self._judge_is_left(a, b, c), dist_c2line, -dist_c2line)
        signed_dist_longi = tf.where(self._judge_is_ahead(a, b, c), dist_longdi, -dist_longdi)
        delta_phi = deal_with_phi_diff(ego_phis - ref_phis)
        delta_vs = ego_vxs - ref_vs
        return tf.stack([signed_dist_longi, signed_dist_lateral, delta_phi, delta_vs], axis=-1)

    def _compute_next_track_info_vectorized(self, next_ego_infos):
        ego_vxs, ego_vys, ego_rs, ego_xs, ego_ys, ego_phis = [next_ego_infos[:, i] for i in range(self.ego_info_dim)]

        # find close point
        indexes, ref_points = self._find_closest_point_batch(ego_xs, ego_ys, self.batch_path)
        # find future point
        future_data = self._future_n_data(indexes, self.future_point_num)

        ref_xs, ref_ys, ref_phis, ref_vs = [ref_points[:, i] for i in range(4)]
        ref_phis_rad = ref_phis * np.pi / 180

        vector_ref_phi = tf.stack([tf.cos(ref_phis_rad), tf.sin(ref_phis_rad)], axis=-1)
        vector_ref_phi_ccw_90 = tf.stack([-tf.sin(ref_phis_rad), tf.cos(ref_phis_rad)], axis=-1) # ccw for counterclockwise
        vector_ego2ref = tf.stack([ref_xs - ego_xs, ref_ys - ego_ys], axis=-1)

        signed_dist_longi = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi, axis=-1))
        signed_dist_lateral = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi_ccw_90, axis=-1))

        delta_phi = deal_with_phi_diff(ego_phis - ref_phis)
        delta_vs = ego_vxs - ref_vs
        track_error = tf.stack([signed_dist_longi, signed_dist_lateral, delta_phi, delta_vs], axis=-1)
        return tf.concat([track_error, future_data], axis=1)

    def _find_closest_point_batch(self, xs, ys, paths):
        xs_tile = tf.tile(tf.reshape(xs, (-1, 1)), [1, self.path_len])
        ys_tile = tf.tile(tf.reshape(ys, (-1, 1)), [1, self.path_len])
        pathx_tile = paths[:, 0, :]
        pathy_tile = paths[:, 1, :]
        dist_array = tf.square(xs_tile - pathx_tile) + tf.square(ys_tile - pathy_tile)
        indexs = tf.argmin(dist_array, 1)
        ref_points = tf.gather(paths, indices=indexs, axis=-1, batch_dims=1)
        return indexs, ref_points

    def _future_n_data(self, current_indexs, n):
        future_data_list = []
        current_indexs = tf.cast(current_indexs, tf.int32)
        for _ in range(n):
            current_indexs += 1
            current_indexs = tf.where(current_indexs >= self.path_len - 1, self.path_len - 1, current_indexs)
            ref_points = tf.gather(self.batch_path, indices=current_indexs, axis=-1, batch_dims=1)
            future_data_list.append(ref_points)
        return tf.concat(future_data_list, axis=1)

    def _judge_is_left(self, a, b, c):
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        featured = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
        return featured > 0.

    def _judge_is_ahead(self, a, b, c):
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        featured = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1)
        return featured >= 0.

    def _convert_to_rela(self, obses):
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_other = self._split_all(obses)
        obses_other_reshape = self._reshape_other(obses_other)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat([tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(len(ego_x), self.per_other_info_dim - 2))],
                        axis=-1)
        ego = tf.expand_dims(ego, 1)
        rela = obses_other_reshape - ego
        rela_obses_other = self._reshape_other(rela, reverse=True)
        return tf.concat([obses_ego, obses_track, obses_light, obses_task, obses_ref, rela_obses_other], axis=-1)

    def _convert_to_abso(self, rela_obses):
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_other = self._split_all(rela_obses)
        obses_other_reshape = self._reshape_other(obses_other)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat([tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(len(ego_x), self.per_other_info_dim - 2))],
                        axis=-1)
        ego = tf.expand_dims(ego, 1)
        abso = obses_other_reshape + ego
        abso_obses_other = self._reshape_other(abso, reverse=True)

        return tf.concat([obses_ego, obses_track, obses_light, obses_task, obses_ref, abso_obses_other], axis=-1)

    def _ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions, self.base_frequency)
        v_xs, v_ys, rs, xs, ys, phis = ego_next_infos[:, 0], ego_next_infos[:, 1], ego_next_infos[:, 2], \
                                       ego_next_infos[:, 3], ego_next_infos[:, 4], ego_next_infos[:, 5]
        v_xs = tf.clip_by_value(v_xs, 0., 35.)
        ego_next_infos = tf.stack([v_xs, v_ys, rs, xs, ys, phis], axis=-1)
        return ego_next_infos

    def _other_predict(self, obses_other):
        obses_other_reshape = self._reshape_other(obses_other)

        xs, ys, vs, phis, turn_rad = obses_other_reshape[:, :, 0], obses_other_reshape[:, :, 1], \
                                     obses_other_reshape[:, :, 2], obses_other_reshape[:, :, 3], \
                                     obses_other_reshape[:, :, -1]
        phis_rad = phis * np.pi / 180.

        middle_cond = logical_and(logical_and(xs > -Para.CROSSROAD_SIZE/2, xs < Para.CROSSROAD_SIZE/2),
                                  logical_and(ys > -Para.CROSSROAD_SIZE/2, ys < Para.CROSSROAD_SIZE/2))
        zeros = tf.zeros_like(xs)

        xs_delta = vs / self.base_frequency * tf.cos(phis_rad)
        ys_delta = vs / self.base_frequency * tf.sin(phis_rad)
        phis_rad_delta = tf.where(middle_cond, vs / self.base_frequency * turn_rad, zeros)

        next_xs, next_ys, next_vs, next_phis_rad = xs + xs_delta, ys + ys_delta, vs, phis_rad + phis_rad_delta
        next_phis_rad = tf.where(next_phis_rad > np.pi, next_phis_rad - 2 * np.pi, next_phis_rad)
        next_phis_rad = tf.where(next_phis_rad <= -np.pi, next_phis_rad + 2 * np.pi, next_phis_rad)
        next_phis = next_phis_rad * 180 / np.pi
        next_info = tf.concat([tf.stack([next_xs, next_ys, next_vs, next_phis], -1), obses_other_reshape[:, :, 4:]],
                              axis=-1)
        next_obses_other = self._reshape_other(next_info, reverse=True)
        return next_obses_other

    def _split_all(self, obses):
        obses_ego = obses[:, :self.ego_info_dim]
        obses_track = obses[:, self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
        obses_future_point = obses[:, self.ego_info_dim + self.track_info_dim:
                                      self.ego_info_dim + self.track_info_dim + self.future_point_num * self.per_path_info_dim]
        obses_light = obses[:, self.ego_info_dim + self.track_info_dim + self.future_point_num * self.per_path_info_dim:
                               self.ego_info_dim + self.track_info_dim + self.future_point_num * self.per_path_info_dim + self.light_info_dim]
        obses_task = obses[:, self.ego_info_dim + self.track_info_dim + self.future_point_num * self.per_path_info_dim + self.light_info_dim:
                              self.ego_info_dim + self.track_info_dim + self.future_point_num * self.per_path_info_dim + self.light_info_dim + self.task_info_dim]
        obses_ref = obses[:, self.ego_info_dim + self.track_info_dim + self.future_point_num * self.per_path_info_dim + self.light_info_dim + self.task_info_dim:
                             self.other_start_dim]
        obses_other = obses[:, self.other_start_dim:]
        return obses_ego, obses_track, obses_future_point, obses_light, obses_task, obses_ref, obses_other

    def _split_other(self, obses_other):
        obses_bike = obses_other[:, :self.bike_num * self.per_other_info_dim]
        obses_person = obses_other[:, self.bike_num * self.per_other_info_dim:
                                      (self.bike_num + self.person_num) * self.per_other_info_dim]
        obses_veh = obses_other[:, (self.bike_num + self.person_num) * self.per_other_info_dim:]
        return obses_bike, obses_person, obses_veh

    def _reshape_other(self, obses_other, reverse=False):
        if reverse:
            return tf.reshape(obses_other, (-1, self.other_number * self.per_other_info_dim))
        else:
            return tf.reshape(obses_other, (-1, self.other_number, self.per_other_info_dim))

    def _action_transformation_for_end2end(self, actions):  # [-1, 1]
        actions = tf.clip_by_value(actions, -1.05, 1.05)
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        steer_scale, a_xs_scale = 0.4 * steer_norm, 2.25 * a_xs_norm - 0.75
        return tf.stack([steer_scale, a_xs_scale], 1)

    # def render(self, mode='human'):
    #     if mode == 'human':
    #         # plot basic map
    #         square_length = Para.CROSSROAD_SIZE
    #         extension = 40
    #         lane_width = Para.LANE_WIDTH
    #         dotted_line_style = '--'
    #         solid_line_style = '-'
    #
    #         plt.cla()
    #         plt.title("Crossroad")
    #         ax = plt.axes(xlim=(-square_length / 2 - extension, square_length / 2 + extension),
    #                       ylim=(-square_length / 2 - extension, square_length / 2 + extension))
    #         plt.axis("equal")
    #         plt.axis('off')
    #
    #         # ax.add_patch(plt.Rectangle((-square_length / 2, -square_length / 2),
    #         #                            square_length, square_length, edgecolor='black', facecolor='none'))
    #         ax.add_patch(plt.Rectangle((-square_length / 2 - extension, -square_length / 2 - extension),
    #                                    square_length + 2 * extension, square_length + 2 * extension, edgecolor='black',
    #                                    facecolor='none'))
    #
    #         # ----------horizon--------------
    #         plt.plot([-square_length / 2 - extension, -square_length / 2], [0, 0], color='black')
    #         plt.plot([square_length / 2 + extension, square_length / 2], [0, 0], color='black')
    #
    #         #
    #         for i in range(1, Para.LANE_NUMBER+1):
    #             linestyle = dotted_line_style if i < Para.LANE_NUMBER else solid_line_style
    #             plt.plot([-square_length / 2 - extension, -square_length / 2], [i*lane_width, i*lane_width],
    #                      linestyle=linestyle, color='black')
    #             plt.plot([square_length / 2 + extension, square_length / 2], [i*lane_width, i*lane_width],
    #                      linestyle=linestyle, color='black')
    #             plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
    #                      linestyle=linestyle, color='black')
    #             plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
    #                      linestyle=linestyle, color='black')
    #
    #         # ----------vertical----------------
    #         plt.plot([0, 0], [-square_length / 2 - extension, -square_length / 2], color='black')
    #         plt.plot([0, 0], [square_length / 2 + extension, square_length / 2], color='black')
    #
    #         #
    #         for i in range(1, Para.LANE_NUMBER+1):
    #             linestyle = dotted_line_style if i < Para.LANE_NUMBER else solid_line_style
    #             plt.plot([i*lane_width, i*lane_width], [-square_length / 2 - extension, -square_length / 2],
    #                      linestyle=linestyle, color='black')
    #             plt.plot([i*lane_width, i*lane_width], [square_length / 2 + extension, square_length / 2],
    #                      linestyle=linestyle, color='black')
    #             plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
    #                      linestyle=linestyle, color='black')
    #             plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
    #                      linestyle=linestyle, color='black')
    #
    #         # ----------stop line--------------
    #         plt.plot([0, Para.LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2], color='black')
    #         plt.plot([-Para.LANE_NUMBER * lane_width, 0], [square_length / 2, square_length / 2], color='black')
    #         plt.plot([-square_length / 2, -square_length / 2], [0, -Para.LANE_NUMBER * lane_width], color='black')
    #         plt.plot([square_length / 2, square_length / 2], [Para.LANE_NUMBER * lane_width, 0], color='black')
    #
    #         # ----------Oblique--------------
    #         plt.plot([Para.LANE_NUMBER * lane_width, square_length / 2], [-square_length / 2, -Para.LANE_NUMBER * lane_width],
    #                  color='black')
    #         plt.plot([Para.LANE_NUMBER * lane_width, square_length / 2], [square_length / 2, Para.LANE_NUMBER * lane_width],
    #                  color='black')
    #         plt.plot([-Para.LANE_NUMBER * lane_width, -square_length / 2], [-square_length / 2, -Para.LANE_NUMBER * lane_width],
    #                  color='black')
    #         plt.plot([-Para.LANE_NUMBER * lane_width, -square_length / 2], [square_length / 2, Para.LANE_NUMBER * lane_width],
    #                  color='black')
    #
    #         def is_in_plot_area(x, y, tolerance=5):
    #             if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
    #                     -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
    #                 return True
    #             else:
    #                 return False
    #
    #         def draw_rotate_rec(x, y, a, l, w, color):
    #             RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
    #             RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
    #             LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
    #             LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
    #             ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
    #             ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
    #             ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
    #             ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)
    #
    #         def plot_phi_line(x, y, phi, color):
    #             line_length = 3
    #             x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
    #                              y + line_length * sin(phi * pi / 180.)
    #             plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)
    #
    #         # abso_obs = self.convert_vehs_to_abso(self.obses)
    #         obses = self.obses.numpy()
    #         ego_info, tracing_info, vehs_info = obses[0, :self.ego_info_dim], \
    #                                             obses[0, self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                                                       self.num_future_data + 1)], \
    #                                             obses[0, self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                         self.num_future_data + 1):]
    #         # plot cars
    #         for veh_index in range(int(len(vehs_info) / self.per_item_info_dim)):
    #             veh = vehs_info[self.per_item_info_dim * veh_index:self.per_item_info_dim * (veh_index + 1)]
    #             veh_x, veh_y, veh_v, veh_phi = veh
    #
    #             if is_in_plot_area(veh_x, veh_y):
    #                 plot_phi_line(veh_x, veh_y, veh_phi, 'black')
    #                 draw_rotate_rec(veh_x, veh_y, veh_phi, L, W, 'black')
    #
    #         # plot own car
    #         delta_y, delta_phi = tracing_info[0], tracing_info[1]
    #         ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = ego_info
    #
    #         plot_phi_line(ego_x, ego_y, ego_phi, 'red')
    #         draw_rotate_rec(ego_x, ego_y, ego_phi, L, W, 'red')
    #
    #         # plot text
    #         text_x, text_y_start = -110, 60
    #         ge = iter(range(0, 1000, 4))
    #         plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
    #         plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
    #         plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
    #         plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
    #         plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))
    #
    #         plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
    #         plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
    #         plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
    #         plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
    #
    #         if self.actions is not None:
    #             steer, a_x = self.actions[0, 0], self.actions[0, 1]
    #             plt.text(text_x, text_y_start - next(ge),
    #                      r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
    #             plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))
    #
    #         text_x, text_y_start = 70, 60
    #         ge = iter(range(0, 1000, 4))
    #
    #         # reward info
    #         if self.reward_info is not None:
    #             for key, val in self.reward_info.items():
    #                 plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))
    #
    #         plt.show()
    #         plt.pause(0.1)


def deal_with_phi_diff(phi_diff):
    phi_diff = tf.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = tf.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


class ReferencePath(object):
    def __init__(self, task, green_or_red='green'):
        self.task = task
        self.path_list = {}
        self.path_all = []
        self.path_len_list = []
        self.max_path_len = 180
        self.control_points = []
        self.green_or_red = green_or_red
        self._construct_ref_path(self.task)
        self._construct_all_path()
        self.path = None
        self.ref_encoding = None
        self.path_index = None
        self.set_path(task, green_or_red)

    def set_path(self, task, green_or_red='green', path_index=None):
        if path_index is None:
            if task == 'left' and green_or_red == 'green':
                path_index = np.random.choice([0, 2, 4])
            elif task == 'left' and green_or_red == 'red':
                path_index = np.random.choice([1, 3, 5])
            elif task == 'straight' and green_or_red == 'green':
                path_index = np.random.choice([6, 8, 10])
            elif task == 'straight' and green_or_red == 'red':
                path_index = np.random.choice([7, 9, 11])
            elif task == 'right':
                path_index = np.random.choice([12, 13, 14])
        self.ref_encoding = [0., 0., 1.]                                  # todo: delete
        self.path = self.path_all[path_index]
        self.path_index = path_index

    def get_future_n_point(self, ego_x, ego_y, n):  # not include the current closest point
        idx, _ = self._find_closest_point(ego_x, ego_y)
        future_n_point = []
        for _ in range(n):
            if idx + 1 >= len(self.path[0]):
                print('Caution!!! idx proceeds the maximum in {} step when getting future point'.format(n))
                idx = idx
            else:
                idx = idx + 1
            x, y, phi, v = self.idx2point(idx)
            future_n_point.extend([x, y, phi, v])
        future_n_point = np.array(future_n_point)
        return future_n_point

    def tracking_error_vector(self, ego_x, ego_y, ego_phi, ego_v):
        _, (x0, y0, phi0, v0) = self._find_closest_point(ego_x, ego_y)
        phi0_rad = phi0 * np.pi / 180
        # np.sin(phi0_rad) * x - np.cos(phi0_rad) * y - np.sin(phi0_rad) * x0 + np.cos(phi0_rad) * y0 = 0
        a, b, c = (x0, y0), (x0 + cos(phi0_rad), y0 + sin(phi0_rad)), (ego_x, ego_y)
        dist_a2c = np.sqrt(np.square(ego_x - x0) + np.square(ego_y - y0))
        dist_c2line = abs(sin(phi0_rad) * ego_x - cos(phi0_rad) * ego_y - sin(phi0_rad) * x0 + cos(phi0_rad) * y0)
        signed_dist_lateral = self._judge_sign_left_or_right(a, b, c) * dist_c2line
        signed_dist_longi = self._judge_sign_ahead_or_behind(a, b, c) * np.sqrt(np.abs(dist_a2c ** 2 - dist_c2line ** 2))
        return np.array([signed_dist_longi, signed_dist_lateral, deal_with_phi_diff(ego_phi - phi0), ego_v - v0])

    def tracking_error_vector_vectorized(self, ego_x, ego_y, ego_phi, ego_v):
        _, (x0, y0, phi0, v0) = self._find_closest_point(ego_x, ego_y)
        phi0_rad = phi0 * np.pi / 180
        # np.sin(phi0_rad) * x - np.cos(phi0_rad) * y - np.sin(phi0_rad) * x0 + np.cos(phi0_rad) * y0 = 0

        vector_ref_phi = np.array([np.cos(phi0_rad), np.sin(phi0_rad)])
        vector_ref_phi_ccw_90 = np.array([-np.sin(phi0_rad), np.cos(phi0_rad)]) # ccw for counterclockwise
        vector_ego2ref = np.array([x0 - ego_x, y0 - ego_y])

        signed_dist_longi = np.negative(np.dot(vector_ego2ref, vector_ref_phi))
        signed_dist_lateral = np.negative(np.dot(vector_ego2ref, vector_ref_phi_ccw_90))

        return np.array([signed_dist_longi, signed_dist_lateral, deal_with_phi_diff(ego_phi - phi0), ego_v - v0])

    def idx2point(self, idx):
        return self.path[0][idx], self.path[1][idx], self.path[2][idx], self.path[3][idx]

    def _judge_sign_left_or_right(self, a, b, c):
        # see https://www.cnblogs.com/manyou/archive/2012/02/23/2365538.html for reference
        # return +1 for left and -1 for right in our case
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        featured = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
        if abs(featured) < 1e-8:
            return 0.
        else:
            return featured / abs(featured)

    def _judge_sign_ahead_or_behind(self, a, b, c):
        # return +1 if head else -1
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        vector1 = np.array([x2 - x1, y2 - y1])
        vector2 = np.array([x3 - x1, y3 - y1])
        mul = np.sum(vector1 * vector2)
        if abs(mul) < 1e-8:
            return 0.
        else:
            return mul / np.abs(mul)

    def _construct_ref_path(self, task):
        sl = 40  # straight length
        dece_dist = 20
        meter_pointnum_ratio = 30
        control_ext = Para.CROSSROAD_SIZE/3.
        planed_trj_g = []
        planed_trj_r = []
        if task == 'left':
            end_offsets = [Para.LANE_WIDTH*(i+0.5) for i in range(Para.LANE_NUMBER)]
            start_offsets = [Para.LANE_WIDTH*0.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE/2
                    control_point2 = start_offset, -Para.CROSSROAD_SIZE/2 + control_ext
                    control_point3 = -Para.CROSSROAD_SIZE/2 + control_ext, end_offset
                    control_point4 = -Para.CROSSROAD_SIZE/2, end_offset
                    self.control_points.append([control_point1,control_point2,control_point3,control_point4])

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                             dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length*meter_pointnum_ratio))
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = Para.LANE_WIDTH/2 * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE/2 - sl, -Para.CROSSROAD_SIZE/2, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(-Para.CROSSROAD_SIZE/2, -Para.CROSSROAD_SIZE/2 - sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)

                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                    vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                        len(end_straight_line_x), dtype=np.float32)
                    vs_red_0 = np.array([8.33] * (len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist + int(Para.L))), dtype=np.float32)
                    vs_red_1 = np.linspace(8.33, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                    vs_red_2 = np.linspace(0.0, 0.0, meter_pointnum_ratio * (dece_dist // 2), endpoint=True, dtype=np.float32)
                    vs_red_3 = np.array([7.0] * (meter_pointnum_ratio * (int(Para.L) - dece_dist // 2) + len(trj_data[0]) - 1) + [8.33] * len(
                            end_straight_line_x), dtype=np.float32)
                    vs_red = np.append(np.append(np.append(vs_red_0, vs_red_1), vs_red_2), vs_red_3)

                    # planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    # planed_trj_red = xs_1, ys_1, phis_1, vs_red
                    # planed_trj_g.append(planed_trj_green)
                    # planed_trj_r.append(planed_trj_red)

                    # filter points by expected velocity
                    filtered_trj_g = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_green)
                    filtered_tri_r = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_red)

                    planed_trj_g.append(filtered_trj_g)
                    planed_trj_r.append(filtered_tri_r)

                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

        elif task == 'straight':
            end_offsets = [Para.LANE_WIDTH*(i+0.5) for i in range(Para.LANE_NUMBER)]
            start_offsets = [Para.LANE_WIDTH*1.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE/2
                    control_point2 = start_offset, -Para.CROSSROAD_SIZE/2 + control_ext
                    control_point3 = end_offset, Para.CROSSROAD_SIZE/2 - control_ext
                    control_point4 = end_offset, Para.CROSSROAD_SIZE/2
                    self.control_points.append([control_point1,control_point2,control_point3,control_point4])

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]]
                                             , dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length*meter_pointnum_ratio))
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE/2 - sl, -Para.CROSSROAD_SIZE/2, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    end_straight_line_y = np.linspace(Para.CROSSROAD_SIZE/2, Para.CROSSROAD_SIZE/2 + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                    vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                        len(end_straight_line_x), dtype=np.float32)
                    vs_red_0 = np.array([8.33] * (len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist + int(Para.L))), dtype=np.float32)
                    vs_red_1 = np.linspace(8.33, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                    vs_red_2 = np.linspace(0.0, 0.0, meter_pointnum_ratio * (dece_dist // 2), endpoint=True,
                                           dtype=np.float32)
                    vs_red_3 = np.array([7.0] * (meter_pointnum_ratio * (int(Para.L) - dece_dist // 2) + len(trj_data[0]) - 1) +
                                        [8.33] * len(end_straight_line_x), dtype=np.float32)
                    vs_red = np.append(np.append(np.append(vs_red_0, vs_red_1), vs_red_2), vs_red_3)

                    # planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    # planed_trj_red = xs_1, ys_1, phis_1, vs_red
                    # planed_trj_g.append(planed_trj_green)
                    # planed_trj_r.append(planed_trj_red)

                    # filter points by expected velocity
                    filtered_trj_g = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_green)
                    filtered_tri_r = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_red)

                    planed_trj_g.append(filtered_trj_g)
                    planed_trj_r.append(filtered_tri_r)

                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

        else:
            assert task == 'right'
            control_ext = Para.CROSSROAD_SIZE/5.
            end_offsets = [-Para.LANE_WIDTH * 2.5, -Para.LANE_WIDTH * 1.5, -Para.LANE_WIDTH * 0.5]
            start_offsets = [Para.LANE_WIDTH*(Para.LANE_NUMBER-0.5)]

            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -Para.CROSSROAD_SIZE/2
                    control_point2 = start_offset, -Para.CROSSROAD_SIZE/2 + control_ext
                    control_point3 = Para.CROSSROAD_SIZE/2 - control_ext, end_offset
                    control_point4 = Para.CROSSROAD_SIZE/2, end_offset
                    self.control_points.append([control_point1,control_point2,control_point3,control_point4])

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                             dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length*meter_pointnum_ratio))
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE/2 - sl, -Para.CROSSROAD_SIZE/2, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(Para.CROSSROAD_SIZE/2, Para.CROSSROAD_SIZE/2 + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                    vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                        len(end_straight_line_x), dtype=np.float32)

                    # same velocity design for turning right
                    # planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    # planed_trj_red = xs_1, ys_1, phis_1, vs_green

                    # filter points by expected velocity
                    filtered_trj_g = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_green)
                    filtered_tri_r = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_green)

                    planed_trj_g.append(filtered_trj_g)
                    planed_trj_r.append(filtered_tri_r)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

    def _construct_all_path(self):
        sl = 40  # straight length
        dece_dist = 20
        meter_pointnum_ratio = 30
        control_ext = Para.CROSSROAD_SIZE/3.
        max_path_len = 180
        # left
        end_offsets = [Para.LANE_WIDTH*(i+0.5) for i in range(Para.LANE_NUMBER)]
        start_offsets = [Para.LANE_WIDTH*0.5]
        for start_offset in start_offsets:
            for end_offset in end_offsets:
                control_point1 = start_offset, -Para.CROSSROAD_SIZE/2
                control_point2 = start_offset, -Para.CROSSROAD_SIZE/2 + control_ext
                control_point3 = -Para.CROSSROAD_SIZE/2 + control_ext, end_offset
                control_point4 = -Para.CROSSROAD_SIZE/2, end_offset
                self.control_points.append([control_point1,control_point2,control_point3,control_point4])

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                          [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                         dtype=np.float32)
                curve = bezier.Curve(node, degree=3)
                s_vals = np.linspace(0, 1.0, int(curve.length*meter_pointnum_ratio))
                trj_data = curve.evaluate_multi(s_vals)
                trj_data = trj_data.astype(np.float32)
                start_straight_line_x = Para.LANE_WIDTH/2 * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE/2 - sl, -Para.CROSSROAD_SIZE/2, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                end_straight_line_x = np.linspace(-Para.CROSSROAD_SIZE/2, -Para.CROSSROAD_SIZE/2 - sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                             np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)

                xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                    len(end_straight_line_x), dtype=np.float32)
                vs_red_0 = np.array([8.33] * (len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist + int(Para.L))), dtype=np.float32)
                vs_red_1 = np.linspace(8.33, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                vs_red_2 = np.linspace(0.0, 0.0, meter_pointnum_ratio * (dece_dist // 2), endpoint=True, dtype=np.float32)
                vs_red_3 = np.array([7.0] * (meter_pointnum_ratio * (int(Para.L) - dece_dist // 2) + len(trj_data[0]) - 1) + [8.33] * len(
                        end_straight_line_x), dtype=np.float32)
                vs_red = np.append(np.append(np.append(vs_red_0, vs_red_1), vs_red_2), vs_red_3)

                # planed_trj_green = xs_1, ys_1, phis_1, vs_green
                # planed_trj_red = xs_1, ys_1, phis_1, vs_red
                # planed_trj_g.append(planed_trj_green)
                # planed_trj_r.append(planed_trj_red)

                # filter points by expected velocity
                filtered_trj_g = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_green, equal_len=True)
                filtered_tri_r = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_red, equal_len=True)

                self.path_all.append(filtered_trj_g)
                self.path_all.append(filtered_tri_r)

        # straight
        end_offsets = [Para.LANE_WIDTH*(i+0.5) for i in range(Para.LANE_NUMBER)]
        start_offsets = [Para.LANE_WIDTH*1.5]
        for start_offset in start_offsets:
            for end_offset in end_offsets:
                control_point1 = start_offset, -Para.CROSSROAD_SIZE/2
                control_point2 = start_offset, -Para.CROSSROAD_SIZE/2 + control_ext
                control_point3 = end_offset, Para.CROSSROAD_SIZE/2 - control_ext
                control_point4 = end_offset, Para.CROSSROAD_SIZE/2
                self.control_points.append([control_point1,control_point2,control_point3,control_point4])

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                          [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]]
                                         , dtype=np.float32)
                curve = bezier.Curve(node, degree=3)
                s_vals = np.linspace(0, 1.0, int(curve.length*meter_pointnum_ratio))
                trj_data = curve.evaluate_multi(s_vals)
                trj_data = trj_data.astype(np.float32)
                start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE/2 - sl, -Para.CROSSROAD_SIZE/2, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                end_straight_line_x = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                end_straight_line_y = np.linspace(Para.CROSSROAD_SIZE/2, Para.CROSSROAD_SIZE/2 + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                             np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                    len(end_straight_line_x), dtype=np.float32)
                vs_red_0 = np.array([8.33] * (len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist + int(Para.L))), dtype=np.float32)
                vs_red_1 = np.linspace(8.33, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                vs_red_2 = np.linspace(0.0, 0.0, meter_pointnum_ratio * (dece_dist // 2), endpoint=True,
                                       dtype=np.float32)
                vs_red_3 = np.array([7.0] * (meter_pointnum_ratio * (int(Para.L) - dece_dist // 2) + len(trj_data[0]) - 1) +
                                    [8.33] * len(end_straight_line_x), dtype=np.float32)
                vs_red = np.append(np.append(np.append(vs_red_0, vs_red_1), vs_red_2), vs_red_3)

                # planed_trj_green = xs_1, ys_1, phis_1, vs_green
                # planed_trj_red = xs_1, ys_1, phis_1, vs_red
                # planed_trj_g.append(planed_trj_green)
                # planed_trj_r.append(planed_trj_red)

                # filter points by expected velocity
                filtered_trj_g = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_green, equal_len=True)
                filtered_tri_r = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_red, equal_len=True)

                self.path_all.append(filtered_trj_g)
                self.path_all.append(filtered_tri_r)

        # right
        control_ext = Para.CROSSROAD_SIZE/5.
        end_offsets = [-Para.LANE_WIDTH * 2.5, -Para.LANE_WIDTH * 1.5, -Para.LANE_WIDTH * 0.5]
        start_offsets = [Para.LANE_WIDTH*(Para.LANE_NUMBER-0.5)]

        for start_offset in start_offsets:
            for end_offset in end_offsets:
                control_point1 = start_offset, -Para.CROSSROAD_SIZE/2
                control_point2 = start_offset, -Para.CROSSROAD_SIZE/2 + control_ext
                control_point3 = Para.CROSSROAD_SIZE/2 - control_ext, end_offset
                control_point4 = Para.CROSSROAD_SIZE/2, end_offset
                self.control_points.append([control_point1,control_point2,control_point3,control_point4])

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                          [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                         dtype=np.float32)
                curve = bezier.Curve(node, degree=3)
                s_vals = np.linspace(0, 1.0, int(curve.length*meter_pointnum_ratio))
                trj_data = curve.evaluate_multi(s_vals)
                trj_data = trj_data.astype(np.float32)
                start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE/2 - sl, -Para.CROSSROAD_SIZE/2, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                end_straight_line_x = np.linspace(Para.CROSSROAD_SIZE/2, Para.CROSSROAD_SIZE/2 + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                             np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                    len(end_straight_line_x), dtype=np.float32)

                # filter points by expected velocity
                filtered_trj_g = self._get_point_by_speed(xs_1, ys_1, phis_1, vs_green, equal_len=True)

                self.path_all.append(filtered_trj_g)

    def _get_point_by_speed(self, xs, ys, phis, vs, dt=0.1, equal_len=False):
        assert len(xs) == len(ys) == len(phis) == len(vs), 'len of path variable is not equal'
        idx = 0
        future_n_x, future_n_y, future_n_phi, future_n_v = [], [], [], []
        while idx + 1 < len(xs):
            x, y, phi, v = xs[idx], ys[idx], phis[idx], vs[idx]
            ds = v * dt
            if ds <= 0.:
                break
            s = 0
            while s < ds:
                if idx + 1 >= len(xs):
                    break
                next_x, next_y, _, _ = xs[idx+1], ys[idx+1], phis[idx+1], vs[idx+1]
                s += np.sqrt(np.square(next_x - x) + np.square(next_y - y))
                x, y = next_x, next_y
                idx += 1
            x, y, phi, v = xs[idx], ys[idx], phis[idx], vs[idx]
            future_n_x.append(x)
            future_n_y.append(y)
            future_n_phi.append(phi)
            future_n_v.append(v)
        if equal_len:
            while len(future_n_x) < self.max_path_len:
                future_n_x.append(future_n_x[-1])
                future_n_y.append(future_n_y[-1])
                future_n_phi.append(future_n_phi[-1])
                future_n_v.append(future_n_v[-1])
        filtered_trj = np.array(future_n_x), np.array(future_n_y), np.array(future_n_phi), np.array(future_n_v)
        return filtered_trj

    def _find_closest_point(self, x, y):
        path_len = len(self.path[0])
        reduced_idx = np.arange(0, path_len)
        reduced_path_x, reduced_path_y = self.path[0][reduced_idx], self.path[1][reduced_idx]
        dists = np.square(x - reduced_path_x) + np.square(y - reduced_path_y)
        idx = np.argmin(dists)
        return idx, self.idx2point(idx)

    def plot_path(self, x, y):
        plt.axis('equal')
        plt.plot(self.path_list[self.traffic_light][0][0], self.path_list[self.traffic_light][0][1], 'b')
        plt.plot(self.path_list[self.traffic_light][1][0], self.path_list[self.traffic_light][1][1], 'r')
        plt.plot(self.path_list[self.traffic_light][2][0], self.path_list[self.traffic_light][2][1], 'g')
        print(self.path_len_list)

        index, closest_point = self._find_closest_point(np.array([x], np.float32),
                                                       np.array([y], np.float32))
        plt.plot(x, y, 'b*')
        plt.plot(closest_point[0], closest_point[1], 'ro')
        plt.show()


def test_ref_path():
    path = ReferencePath('right', '0')
    path.plot_path(1.875, 0)


def test_future_n_data():
    path = ReferencePath('straight', '0')
    plt.axis('equal')
    current_i = 600
    plt.plot(path.path[0], path.path[1])
    future_data_list = path.future_n_data(current_i, 5)
    plt.plot(path.indexs2points(current_i)[0], path.indexs2points(current_i)[1], 'go')
    for point in future_data_list:
        plt.plot(point[0], point[1], 'r*')
    plt.show()


def test_tracking_error_vector():
    path = ReferencePath('straight', "0")
    xs = np.array([1.875, 1.875, -10, -20], np.float32)
    ys = np.array([-20, 0, -10, -1], np.float32)
    phis = np.array([90, 135, 135, 180], np.float32)
    vs = np.array([10, 12, 10, 10], np.float32)

    tracking_error_vector = path.tracking_error_vector(xs, ys, phis, vs, 10)
    print(tracking_error_vector)


def test_model():
    from endtoend import CrossroadEnd2endPiIntegrate
    env = CrossroadEnd2endPiIntegrate('left', 0)
    model = EnvironmentModel('left', '0', 0)
    obs_list = []
    obs = env.reset()
    done = 0
    # while not done:
    for i in range(10):
        obs_list.append(obs)
        action = np.array([0, -1], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        env.render()
    obses = np.stack(obs_list, 0)
    model.reset(obses, 'left')
    print(obses.shape)
    for rollout_step in range(100):
        actions = tf.tile(tf.constant([[0.5, 0]], dtype=tf.float32), tf.constant([len(obses), 1]))
        obses, rewards, punish1, punish2, _, _ = model.rollout_out(actions)
        print(rewards.numpy()[0], punish1.numpy()[0])
        model.render()


def test_tf_function():
    class Test2():
        def __init__(self):
            self.c = 2

        def step1(self, a):
            print('trace')
            self.c = a

        def step2(self):
            return self.c

    test2 = Test2()

    @tf.function#(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
    def f(a):
        test2.step1(a)
        return test2.step2()

    print(f(2), type(test2.c))
    print(f(2), test2.c)

    print(f(tf.constant(2)), type(test2.c))
    print(f(tf.constant(3)), test2.c)

    # print(f(2), test2.c)
    # print(f(3), test2.c)
    # print(f(2), test2.c)
    # print(f())
    # print(f())
    #
    # test2.c.assign_add(12)
    # print(test2.c)
    # print(f())





    # b= test2.create_test1(1)
    # print(test2.b,b, test2.b.a)
    # b=test2.create_test1(2)
    # print(test2.b,b,test2.b.a)
    # b=test2.create_test1(1)
    # print(test2.b,b,test2.b.a)
    # test2.create_test1(1)
    # test2.pc()
    # test2.create_test1(1)
    # test2.pc()
@tf.function
def test_tffunc(inttt):
    print(22)
    if inttt=='1':
        a = 2
    elif inttt == '2':
        a = 233
    else:
        a=22
    return a

def test_ref():
    import numpy as np
    import matplotlib.pyplot as plt
    # ref = ReferencePath('left')
    # path1, path2, path3 = ref.path_list
    # path1, path2, path3 = [ite[1200:-1200] for ite in path1],\
    #                       [ite[1200:-1200] for ite in path2], \
    #                       [ite[1200:-1200] for ite in path3]
    # x1, y1, phi1 = path1
    # x2, y2, phi2 = path2
    # x3, y3, phi3 = path3
    # p1, p2, p3 = np.arctan2(y1-(-Para.CROSSROAD_SIZE/2), x1 - (-Para.CROSSROAD_SIZE/2)), \
    #              np.arctan2(y2 - (-Para.CROSSROAD_SIZE / 2), x2 - (-Para.CROSSROAD_SIZE / 2)), \
    #              np.arctan2(y3 - (-Para.CROSSROAD_SIZE / 2), x3 - (-Para.CROSSROAD_SIZE / 2))
    # d1, d2, d3 = np.sqrt(np.square(x1-(-Para.CROSSROAD_SIZE/2))+np.square(y1-(-Para.CROSSROAD_SIZE/2))),\
    #              np.sqrt(np.square(x2-(-Para.CROSSROAD_SIZE/2))+np.square(y2-(-Para.CROSSROAD_SIZE/2))),\
    #              np.sqrt(np.square(x3-(-Para.CROSSROAD_SIZE/2))+np.square(y3-(-Para.CROSSROAD_SIZE/2)))
    #
    # plt.plot(p1, d1, 'r')
    # plt.plot(p2, d2, 'g')
    # plt.plot(p3, d3, 'b')
    # z1 = np.polyfit(p1, d1, 3, rcond=None, full=False, w=None, cov=False)
    # p1_fit = np.poly1d(z1)
    # plt.plot(p1, p1_fit(p1), 'r*')
    #
    # z2 = np.polyfit(p2, d2, 3, rcond=None, full=False, w=None, cov=False)
    # p2_fit = np.poly1d(z2)
    # plt.plot(p2, p2_fit(p2), 'g*')
    #
    # z3 = np.polyfit(p3, d3, 3, rcond=None, full=False, w=None, cov=False)
    # p3_fit = np.poly1d(z3)
    # plt.plot(p3, p3_fit(p3), 'b*')

    ref = ReferencePath('left', '0')
    # print(ref.path_list[ref.judge_traffic_light('0')])
    path1, path2, path3 = ref.path_list[LIGHT_PHASE_TO_GREEN_OR_RED[0]]
    path1, path2, path3 = [ite[1200:-1200] for ite in path1], \
                          [ite[1200:-1200] for ite in path2], \
                          [ite[1200:-1200] for ite in path3]
    x1, y1, phi1, v1 = path1
    x2, y2, phi2, v1 = path2
    x3, y3, phi3, v1 = path3

    plt.plot(y1, x1, 'r')
    plt.plot(y2, x2, 'g')
    plt.plot(y3, x3, 'b')
    z1 = np.polyfit(y1, x1, 3, rcond=None, full=False, w=None, cov=False)
    print(type(list(z1)))
    p1_fit = np.poly1d(z1)
    print(z1, p1_fit)
    plt.plot(y1, p1_fit(y1), 'r*')
    plt.show()


if __name__ == '__main__':
    test_ref()


