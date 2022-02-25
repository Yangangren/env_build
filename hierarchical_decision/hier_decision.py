#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================

import datetime
import shutil
import time
import json
import matplotlib.patches as mpatch
import os
import heapq
from math import cos, sin, pi

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PatchCollection
import numpy as np
import tensorflow as tf

from dynamics_and_models import EnvironmentModel, ReferencePath
from endtoend import CrossroadEnd2endMixPI
from endtoend_env_utils import *
from multi_path_generator import MultiPathGenerator
from utils.load_policy import LoadPolicy
from utils.misc import TimerStat
from utils.recorder import Recorder


class HierarchicalDecision(object):
    def __init__(self, train_exp_dir, ite, logdir=None):
        self.policy = LoadPolicy('../utils/models/{}'.format(train_exp_dir), ite)
        self.args = self.policy.args
        self.env = CrossroadEnd2endMixPI()
        self.model = EnvironmentModel()
        self.recorder = Recorder()
        self.episode_counter = -1
        self.step_counter = -1
        self.obs = None
        self.stg = MultiPathGenerator()
        self.step_timer = TimerStat()
        self.ss_timer = TimerStat()
        self.logdir = logdir
        if self.logdir is not None:
            config = dict(train_exp_dir=train_exp_dir, ite=ite)
            with open(self.logdir + '/config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.fig_plot = 0
        self.hist_posi = []
        self.old_index = 0
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        # ------------------build graph for tf.function in advance-----------------------
        # todo
        if self.env.training_task == 'straight':
            path_number = 2
        else:
            path_number = 3
        for i in range(path_number):
            obs, _ = self.env.reset()
            obs = obs[np.newaxis, :]
            # self.is_safe(obs)
        obs, _ = self.env.reset()
        obs = obs[np.newaxis, :]
        self.policy.run_batch(obs)
        self.policy.obj_value_batch(obs)
        # ------------------build graph for tf.function in advance-----------------------
        self.reset()

    def reset(self,):
        self.obs, _ = self.env.reset()
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        self.recorder.reset()
        self.old_index = 0
        self.hist_posi = []
        if self.logdir is not None:
            self.episode_counter += 1
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))
            self.step_counter = -1
            self.recorder.save(self.logdir)
            if self.episode_counter >= 1:
                select_and_rename_snapshots_of_an_episode(self.logdir, self.episode_counter-1, 12)
                self.recorder.plot_and_save_ith_episode_curves(self.episode_counter-1,
                                                               self.logdir + '/episode{}/figs'.format(self.episode_counter-1),
                                                               isshow=False)
        return self.obs

    @tf.function
    def is_safe(self, obs):
        # todo
        # self.model.add_traj(obs_ego, obs_other, [self.env.veh_num], path_index)
        punish = 0.
        for step in range(5):
            action = self.policy.run_batch(obs)
            obses, _, _, _, _, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, veh2speed4real\
                = self.model.rollout_out(action)
            punish += veh2veh4real[0] + veh2road4real[0] + veh2bike4real[0] + veh2person4real[0] + veh2speed4real[0]
        return False if punish > 0 else True

    def safe_shield(self, real_obs):
        action_safe_set = [[[0., -1.]]]
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :], dtype=tf.float32)
        # todo
        # if not self.is_safe(real_obs):
        #     print('SAFETY SHIELD STARTED!')
        #     return np.array(action_safe_set[0], dtype=np.float32).squeeze(0), True
        # else:
        #     return self.policy.run_batch(real_obs).numpy()[0], False
        return self.policy.run_batch(real_obs).numpy()[0], False

    def step(self):
        self.step_counter += 1
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        with self.step_timer:
            obs_list, mask_list, future_n_point_list = [], [], []
            # select optimal path
            for path in self.path_list:
                self.env.set_traj(path)
                vector, mask_vector, future_n_point = self.env._get_obs()
                obs_list.append(vector)
                mask_list.append(mask_vector)
                future_n_point_list.append(future_n_point)
            all_obs = tf.stack(obs_list, axis=0).numpy()

            path_values = self.policy.obj_value_batch(all_obs).numpy()
            # path_values = [1.0, 1.0, 3.] * path_values
            old_value = path_values[self.old_index]
            # value is to approximate (- sum of reward)
            new_index, new_value = int(np.argmin(path_values)), min(path_values)
            # rule for equal traj value
            path_index_error = []
            if self.step_counter % 3 == 0:
                if heapq.nsmallest(2, path_values)[0] == heapq.nsmallest(2, path_values)[1]:
                    for i in range(len(path_values)):
                        if path_values[i] == min(path_values):
                            index_error = abs(self.old_index - i)
                            path_index_error.append(index_error)
                    # new_index_new = min(path_index_error) + self.old_index if min(path_index_error) + self.old_index < 4 else self.old_index - min(path_index_error)
                    new_index_new = self.old_index - min(path_index_error) if self.old_index - min(path_index_error) > -1 else self.old_index + min(path_index_error)
                    new_value_new = path_values[new_index_new]
                    path_index = self.old_index if old_value - new_value_new < 0.1 else new_index_new
                else:
                    path_index = self.old_index if old_value - new_value < 0.1 else new_index
                self.old_index = path_index
            else:
                path_index = self.old_index
            self.env.set_traj(self.path_list[path_index])
            obs_real, mask_real, future_n_point_real = obs_list[path_index], mask_list[path_index], future_n_point_list[path_index]

            # obtain safe action
            with self.ss_timer:
                safe_action, is_ss = self.safe_shield(obs_real)
            # print('ALL TIME:', self.step_timer.mean, 'ss', self.ss_timer.mean)
        self.render(self.path_list, path_values, path_index)
        self.recorder.record(obs_real, safe_action, self.step_timer.mean, path_index, path_values, self.ss_timer.mean, is_ss)
        self.obs, r, done, info = self.env.step(safe_action)
        return done

    def render(self, traj_list, path_values, path_index):
        extension = 40
        dotted_line_style = '--'
        solid_line_style = '-'

        plt.clf()
        ax = plt.axes([-0.00, -0.00, 1.0, 1.0], facecolor='white')
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

        def get_partici_type_str(partici_type):
            if partici_type[0] == 1.:
                return 'bike'
            elif partici_type[1] == 1.:
                return 'person'
            elif partici_type[2] == 1.:
                return 'veh'

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
        for i in range(len(self.env.interested_other)):
            item = self.env.interested_other[i]
            item_mask = item['exist']
            item_x = item['x']
            item_y = item['y']
            item_phi = item['phi']
            item_l = item['l']
            item_w = item['w']
            item_type = item['type']
            # todo
            if is_in_plot_area(item_x, item_y):
                plot_phi_line(item_type, item_x, item_y, item_phi, 'black')
                draw_rotate_rec(item_type, item_x, item_y, item_phi, item_l, item_w, color='g', linestyle=':', patch=True)
            #   plt.text(item_x, item_y, str(item_mask)[0])

        # plot own car
        abso_obs = self.env._convert_to_abso(self.obs)
        obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other = self.env._split_all(abso_obs)
        ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = obs_ego
        devi_longi, devi_lateral, devi_phi, devi_v = obs_track
        plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'fuchsia')
        draw_rotate_rec('self_car', ego_x, ego_y, ego_phi, self.env.ego_l, self.env.ego_w, 'fuchsia')

        # plot history
        self.hist_posi.append((ego_x, ego_y))
        xs = [pos[0] for pos in self.hist_posi]
        ys = [pos[1] for pos in self.hist_posi]
        plt.scatter(np.array(xs), np.array(ys), color='fuchsia', alpha=0.1)

        # ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
        # _, point = self.ref_path._find_closest_point(ego_x, ego_y)
        # path_x, path_y, path_phi, path_v = point[0], point[1], point[2], point[3]
        # plt.plot(path_x, path_y, 'g.')
        # plt.plot(self.future_n_point[0], self.future_n_point[1], 'g.')

        # todo
        # plot real time traj
        color = ['blue', 'coral', 'darkcyan', 'pink']
        for i, item in enumerate(self.path_list):
            if i == path_index:
                plt.plot(item.path[0], item.path[1], color=color[i], alpha=1.0)
            else:
                plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
        _, point = self.env.ref_path._find_closest_point(ego_x, ego_y)
        # plt.plot(path_x, path_y, 'g.')
        path_x, path_y, path_phi, path_v = point[0], point[1], point[2], point[3]

        # text
        text_x, text_y_start = -90, 60
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
        plt.text(text_x, text_y_start - next(ge), 'light: {}'.format(self.env.light_phase))
        if self.env.action is not None:
            steer, a_x = self.env.action[0], self.env.action[1]
            plt.text(text_x, text_y_start - next(ge),
                     r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
            plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

        text_x, text_y_start = 65, 60
        ge = iter(range(0, 1000, 4))

        # done info
        plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.env.done_type))

        # reward info
        if self.env.reward_info is not None:
            for key, val in self.env.reward_info.items():
                plt.text(text_x, text_y_start - next(ge), 'rew_{}: {:.4f}'.format(key, val))

        # indicator for trajectory selection
        # text_x, text_y_start = 25, -30
        # ge = iter(range(0, 1000, 6))
        # if path_values is not None:
        #     for i, value in enumerate(path_values):
        #         if i == path_index:
        #             plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=14,
        #                      color=color[i], fontstyle='italic')
        #         else:
        #             plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=12,
        #                      color=color[i], fontstyle='italic')
        ax.add_collection(PatchCollection(patches, match_original=True, zorder=4))
        plt.show()
        plt.pause(0.001)
        if self.logdir is not None:
            plt.savefig(self.logdir + '/episode{}'.format(self.episode_counter) + '/step{}.pdf'.format(self.step_counter))


def plot_and_save_ith_episode_data(logdir, i):
    recorder = Recorder()
    recorder.load(logdir)
    save_dir = logdir + '/episode{}/figs'.format(i)
    os.makedirs(save_dir, exist_ok=True)
    recorder.plot_and_save_ith_episode_curves(i, save_dir, True)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    os.makedirs(logdir)
    hier_decision = HierarchicalDecision('experiment-2022-02-21-21-39-22', 200000, logdir)

    for i in range(300):
        for _ in range(200):
            done = hier_decision.step()
            if done: break
        hier_decision.reset()


def plot_static_path():
    extension = 20
    light_line_width = 3
    dotted_line_style = '--'
    solid_line_style = '-'
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes([0, 0, 1, 1])
    for ax in fig.get_axes():
        ax.axis('off')
    ax.axis("equal")

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
        [Para.OFFSET_L + 0.2 - extension * sin(Para.ANGLE_L / 180 * pi), Para.OFFSET_L + 0.2], color='orange', zorder=1)
    plt.plot(
        [-Para.CROSSROAD_SIZE_LAT / 2 - extension * cos(Para.ANGLE_L / 180 * pi), -Para.CROSSROAD_SIZE_LAT / 2],
        [Para.OFFSET_L - 0.2 - extension * sin(Para.ANGLE_L / 180 * pi), Para.OFFSET_L - 0.2], color='orange', zorder=1)
    plt.plot(
        [Para.CROSSROAD_SIZE_LAT / 2 + extension * cos(Para.ANGLE_R / 180 * pi), Para.CROSSROAD_SIZE_LAT / 2],
        [Para.OFFSET_R + 0.2 + extension * sin(Para.ANGLE_R / 180 * pi), Para.OFFSET_R + 0.2], color='orange', zorder=1)
    plt.plot(
        [Para.CROSSROAD_SIZE_LAT / 2 + extension * cos(Para.ANGLE_R / 180 * pi), Para.CROSSROAD_SIZE_LAT / 2],
        [Para.OFFSET_R - 0.2 + extension * sin(Para.ANGLE_R / 180 * pi), Para.OFFSET_R - 0.2], color='orange', zorder=1)

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

    for task in ['left', 'straight', 'right']:
        path = ReferencePath(task)
        path_list = path.path_list['green']
        control_points = path.control_points
        color = ['royalblue', 'coral', 'darkcyan', 'firebrick']

        for i, (path_x, path_y, _, _) in enumerate(path_list):
            plt.plot(path_x[600:-600], path_y[600:-600], color=color[i])
        for i, four_points in enumerate(control_points):
            for point in four_points:
                plt.scatter(point[0], point[1], color=color[i], s=20, alpha=0.7)
            plt.plot([four_points[0][0], four_points[1][0]], [four_points[0][1], four_points[1][1]], linestyle='--', color=color[i], alpha=0.5)
            plt.plot([four_points[1][0], four_points[2][0]], [four_points[1][1], four_points[2][1]], linestyle='--', color=color[i], alpha=0.5)
            plt.plot([four_points[2][0], four_points[3][0]], [four_points[2][1], four_points[3][1]], linestyle='--', color=color[i], alpha=0.5)

    plt.savefig('./multipath_planning.png')
    plt.show()


def select_and_rename_snapshots_of_an_episode(logdir, epinum, num):
    file_list = os.listdir(logdir + '/episode{}'.format(epinum))
    file_num = len(file_list) - 1
    interval = file_num // (num-1)
    start = file_num % (num-1)
    print(start, file_num, interval)
    selected = [start//2] + [start//2+interval*i for i in range(1, num-1)]
    print(selected)
    if file_num > 0:
        for i, j in enumerate(selected):
            shutil.copyfile(logdir + '/episode{}/step{}.pdf'.format(epinum, j),
                            logdir + '/episode{}/figs/{}.pdf'.format(epinum, i))


if __name__ == '__main__':
    main()
    # plot_static_path()
    # plot_and_save_ith_episode_data('./results/good/2021-03-15-23-56-21', 0)
    # select_and_rename_snapshots_of_an_episode('./results/good/2021-03-15-23-56-21', 0, 12)


