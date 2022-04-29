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

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dynamics_and_models import EnvironmentModel, ReferencePath
from endtoend import CrossroadEnd2end
from endtoend_env_utils import *
from multi_path_generator import MultiPathGenerator
from utils.load_policy import LoadPolicy
from utils.misc import TimerStat
from utils.recorder import Recorder


class HierarchicalDecision(object):
    def __init__(self, train_exp_dir, ite, logdir=None):
        self.policy = LoadPolicy('../utils/models/{}'.format(train_exp_dir), ite)
        self.args = self.policy.args
        self.env = CrossroadEnd2end(mode='testing', future_point_num=self.args.num_rollout_list_for_policy_update[0])
        self.model = EnvironmentModel(mode='testing')
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
        obs, all_info = self.env.reset()
        mask, path_index = all_info['mask'], all_info['path_index']
        obs = tf.convert_to_tensor(obs[np.newaxis, :], dtype=tf.float32)
        mask = tf.convert_to_tensor(mask[np.newaxis, :], dtype=tf.float32)
        path_index = tf.expand_dims(path_index, axis=0)
        self.is_safe(obs, mask, path_index)
        self.policy.run_batch(obs, mask)
        self.policy.obj_value_batch(obs, mask)
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
    def is_safe(self, obses, masks, path_index):
        self.model.reset(obses, path_index)
        punish = 0.
        for step in range(5):
            action, _ = self.policy.run_batch(obses, masks)
            obses, _, _, _, veh2veh4real, veh2road4real, veh2line4real = self.model.rollout_out(action)
            punish += veh2veh4real[0]
        return False if punish > 0 else True

    def safe_shield(self, real_obs, real_mask, path_index_real):
        action_safe_set = [[[0., -1.]]]
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :], dtype=tf.float32)
        real_mask = tf.convert_to_tensor(real_mask[np.newaxis, :], dtype=tf.float32)
        path_index_real = tf.expand_dims(path_index_real, axis=0)
        if not self.is_safe(real_obs, real_mask, path_index_real):
            print('SAFETY SHIELD STARTED!')
            return np.array(action_safe_set[0], dtype=np.float32).squeeze(0), np.zeros(shape=(self.env.other_number)), True
        else:
            action, weight = self.policy.run_batch(real_obs, real_mask)
            return action.numpy()[0], weight.numpy()[0], False

    def step(self):
        self.step_counter += 1
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        with self.step_timer:
            obs_list, mask_list, path_index_list = [], [], []
            # select optimal path
            for path in self.path_list:
                self.env.set_traj(path)
                vector, mask_vector, _ = self.env._get_obs()
                obs_list.append(vector)
                mask_list.append(mask_vector)
                path_index_list.append(path.path_index)
            all_obs = tf.stack(obs_list, axis=0).numpy()
            all_mask = tf.stack(mask_list, axis=0).numpy()

            path_values = self.policy.obj_value_batch(all_obs, all_mask).numpy()
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
            obs_real, mask_real, path_index_real = obs_list[path_index], mask_list[path_index], path_index_list[path_index]

            # obtain safe action
            with self.ss_timer:
                safe_action, weights, is_ss = self.safe_shield(obs_real, mask_real, path_index_real)
            # print('ALL TIME:', self.step_timer.mean, 'ss', self.ss_timer.mean)
        self.render(path_values, path_index, weights)
        self.recorder.record(obs_real, safe_action, self.step_timer.mean, path_index, path_values, self.ss_timer.mean, is_ss)
        self.obs, r, done, info = self.env.step(safe_action)
        if done:
            print(self.env.done_type)
        return done

    def render(self, path_values, path_index, weights):
        square_length = Para.CROSSROAD_SIZE
        extension = 30
        lane_width = Para.LANE_WIDTH
        light_line_width = 3
        dotted_line_style = '--'
        solid_line_style = '-'

        if not self.fig_plot:
            self.fig = plt.figure(figsize=(8, 8))
            self.fig_plot = 1
        plt.ion()
        plt.cla()
        ax = plt.axes([-0.00, -0.00, 1.0, 1.0])
        for ax in self.fig.get_axes():
            ax.axis('off')
        ax.axis("equal")

        # ----------arrow--------------
        plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 5, color='b')
        plt.arrow(lane_width / 2, -square_length / 2 - 10 + 5, -0.5, 0, color='b', head_width=1)
        plt.arrow(lane_width * 0.5, -square_length / 2 - 10, 0, 4, color='b', head_width=1)
        plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 5, color='b')
        plt.arrow(lane_width * 1.5, -square_length / 2 - 10 + 5, 0.5, 0, color='b', head_width=1)

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

        v_light = self.env.light_phase
        if v_light == 0:
            v_color, h_color = 'green', 'red'
        elif v_light == 1:
            v_color, h_color = 'orange', 'red'
        elif v_light == 2:
            v_color, h_color = 'red', 'green'
        else:
            v_color, h_color = 'red', 'orange'

        plt.plot([0, (Para.LANE_NUMBER - 1) * lane_width], [-square_length / 2, -square_length / 2],
                 color=v_color, linewidth=light_line_width)
        plt.plot([(Para.LANE_NUMBER - 1) * lane_width, Para.LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
                 color='green', linewidth=light_line_width)

        plt.plot([-Para.LANE_NUMBER * lane_width, -(Para.LANE_NUMBER - 1) * lane_width], [square_length / 2, square_length / 2],
                 color='green', linewidth=light_line_width)
        plt.plot([-(Para.LANE_NUMBER - 1) * lane_width, 0], [square_length / 2, square_length / 2],
                 color=v_color, linewidth=light_line_width)

        plt.plot([-square_length / 2, -square_length / 2], [0, -(Para.LANE_NUMBER - 1) * lane_width],
                 color=h_color, linewidth=light_line_width)
        plt.plot([-square_length / 2, -square_length / 2], [-(Para.LANE_NUMBER - 1) * lane_width, -Para.LANE_NUMBER * lane_width],
                 color='green', linewidth=light_line_width)

        plt.plot([square_length / 2, square_length / 2], [(Para.LANE_NUMBER - 1) * lane_width, 0],
                 color=h_color, linewidth=light_line_width)
        plt.plot([square_length / 2, square_length / 2], [Para.LANE_NUMBER * lane_width, (Para.LANE_NUMBER - 1) * lane_width],
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

        def draw_rotate_rec(x, y, a, l, w, c, facecolor='white', alpha=None):
            bottom_left_x, bottom_left_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            ax.add_patch(plt.Rectangle((x + bottom_left_x, y + bottom_left_y), w, l, edgecolor=c,
                                       facecolor=facecolor, angle=-(90 - a), alpha=alpha, zorder=50))

        def plot_phi_line(x, y, phi, color):
            line_length = 3
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                             y + line_length * sin(phi * pi / 180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        def draw_sensor_range(x_ego, y_ego, a_ego, l_bias, w_bias, angle_bias, angle_range, dist_range, color):
            x_sensor = x_ego + l_bias * cos(a_ego) - w_bias * sin(a_ego)
            y_sensor = y_ego + l_bias * sin(a_ego) + w_bias * cos(a_ego)
            theta1 = a_ego + angle_bias - angle_range / 2
            theta2 = a_ego + angle_bias + angle_range / 2
            sensor = mpatch.Wedge([x_sensor, y_sensor], dist_range, theta1=theta1 * 180 / pi,
                                   theta2=theta2 * 180 / pi, fc=color, alpha=0.2)
            ax.add_patch(sensor)

        # plot cars
        for veh in self.env.all_other:
            veh_x = veh['x']
            veh_y = veh['y']
            veh_phi = veh['phi']
            veh_l = veh['l']
            veh_w = veh['w']
            if is_in_plot_area(veh_x, veh_y):
                plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

        # plot vehicles from sensors
        # for veh in self.env.detected_vehicles:
        #     veh_x = veh['x']
        #     veh_y = veh['y']
        #     veh_phi = veh['phi']
        #     veh_l = veh['l']
        #     veh_w = veh['w']
        #     plot_phi_line(veh_x, veh_y, veh_phi, 'lime')
        #     draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'lime')

        # plot interested others
        if weights is not None:
            assert weights.shape == (self.args.other_number,), print(weights.shape)
            index_top_k_in_weights = weights.argsort()[-4:][::-1]
        for i in range(len(self.env.interested_other)):
            item = self.env.interested_other[i]
            item_mask = item['exist']
            item_x = item['x']
            item_y = item['y']
            item_phi = item['phi']
            item_l = item['l']
            item_w = item['w']
            # if is_in_plot_area(item_x, item_y):
            #     plot_phi_line(item_x, item_y, item_phi, 'black')
            #     draw_rotate_rec(item_x, item_y, item_phi, item_l, item_w, c='m')
            # if (weights is not None) and (item_mask == 1.0):
            #     draw_rotate_rec(item_x, item_y, item_phi, item_l, item_w, c='lime', facecolor='lime', alpha=weights[i])
            #     plt.text(item_x + 0.05, item_y + 0.15, "{:.2f}".format(weights[i]), color='purple', fontsize=12)

        # plot_interested vehs
        # for mode, num in self.env.veh_mode_dict.items():
        #     for i in range(num):
        #         veh = self.env.interested_vehs[mode][i]
        #         veh_x = veh['x']
        #         veh_y = veh['y']
        #         veh_phi = veh['phi']
        #         veh_l = veh['l']
        #         veh_w = veh['w']
        #         task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}
        #
        #         if is_in_plot_area(veh_x, veh_y):
        #             plot_phi_line(veh_x, veh_y, veh_phi, 'black')
        #             task = MODE2TASK[mode]
        #             color = task2color[task]
        #             draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, facecolor=color)

        # plot own car
        abso_obs = self.env._convert_to_abso(self.obs)
        obs_ego, obs_track, obs_future_point, obs_light, obs_task, obs_ref, obs_other = self.env._split_all(abso_obs)
        ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = obs_ego
        devi_longi, devi_lateral, devi_phi, devi_v = obs_track

        plot_phi_line(ego_x, ego_y, ego_phi, 'fuchsia')
        draw_rotate_rec(ego_x, ego_y, ego_phi, self.env.ego_l, self.env.ego_w, 'fuchsia', facecolor='white')
        self.hist_posi.append((ego_x, ego_y))

        # plot sensors
        draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.env.ego_l / 2, w_bias=0, angle_bias=0,
                          angle_range=2 * pi, dist_range=35, color='thistle')
        draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.env.ego_l / 2, w_bias=0, angle_bias=0,
                          angle_range=70 * pi / 180, dist_range=40, color="slategray")
        draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.env.ego_l / 2, w_bias=0, angle_bias=0,
                          angle_range=90 * pi / 180, dist_range=30, color="slategray")

        # plot history
        xs = [pos[0] for pos in self.hist_posi]
        ys = [pos[1] for pos in self.hist_posi]
        plt.scatter(np.array(xs), np.array(ys), color='fuchsia', alpha=0.1)

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
        text_x, text_y_start = -120, 60
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
        plt.text(text_x, text_y_start - next(ge), ' ')
        plt.text(text_x, text_y_start - next(ge), 'light: {}'.format(self.env.light_phase))
        plt.text(text_x, text_y_start - next(ge), ' ')
        if self.env.action is not None:
            steer, a_x = self.env.action[0], self.env.action[1]
            plt.text(text_x, text_y_start - next(ge),
                     r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
            plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

        text_x, text_y_start = 76, 60
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
        plt.xlim(-(square_length / 2 + extension), square_length / 2 + extension)
        plt.ylim(-(square_length / 2 + extension), square_length / 2 + extension)
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
    hier_decision = HierarchicalDecision('experiment-2021-11-13-19-55-15', 300000, logdir)

    for i in range(300):
        for _ in range(200):
            done = hier_decision.step()
            if done: break
        hier_decision.reset()


def plot_static_path():
    square_length = Para.CROSSROAD_SIZE
    extension = 20
    lane_width = Para.LANE_WIDTH
    light_line_width = 3
    dotted_line_style = '--'
    solid_line_style = '-'
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes([0, 0, 1, 1])
    for ax in fig.get_axes():
        ax.axis('off')
    ax.axis("equal")

    # ----------arrow--------------
    plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 5, color='b')
    plt.arrow(lane_width / 2, -square_length / 2 - 10 + 5, -0.5, 0, color='b', head_width=1)
    plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='b', head_width=1)
    plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 5, color='b')
    plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 5, 0.5, 0, color='b', head_width=1)

    # ----------horizon--------------

    plt.plot([-square_length / 2 - extension, -square_length / 2], [0.3, 0.3], color='orange')
    plt.plot([-square_length / 2 - extension, -square_length / 2], [-0.3, -0.3], color='orange')
    plt.plot([square_length / 2 + extension, square_length / 2], [0.3, 0.3], color='orange')
    plt.plot([square_length / 2 + extension, square_length / 2], [-0.3, -0.3], color='orange')

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
    plt.plot([0, Para.LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([Para.LANE_NUMBER * lane_width, 0], [square_length / 2, square_length / 2],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([-square_length / 2, -square_length / 2], [0, Para.LANE_NUMBER * lane_width],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([square_length / 2, square_length / 2], [-Para.LANE_NUMBER * lane_width, 0],
             color='black', linewidth=light_line_width, alpha=0.3)

    # ----------Oblique--------------
    plt.plot([Para.LANE_NUMBER * lane_width, square_length / 2], [-square_length / 2, -Para.LANE_NUMBER * lane_width],
             color='black', linewidth=2)
    plt.plot([Para.LANE_NUMBER * lane_width, square_length / 2], [square_length / 2, Para.LANE_NUMBER * lane_width],
             color='black', linewidth=2)
    plt.plot([-Para.LANE_NUMBER * lane_width, -square_length / 2], [-square_length / 2, -Para.LANE_NUMBER * lane_width],
             color='black', linewidth=2)
    plt.plot([-Para.LANE_NUMBER * lane_width, -square_length / 2], [square_length / 2, Para.LANE_NUMBER * lane_width],
             color='black', linewidth=2)

    for task in ['left', 'straight', 'right']:
        path = ReferencePath(task)
        path_list = path.path_list['green']
        control_points = path.control_points
        color = ['blue', 'coral', 'darkcyan']
        for i, (path_x, path_y, _, _) in enumerate(path_list):
            plt.plot(path_x[600:-600], path_y[600:-600], color=color[i])
        for i, four_points in enumerate(control_points):
            for point in four_points:
                plt.scatter(point[0], point[1], color=color[i])
            plt.plot([four_points[0][0], four_points[1][0]], [four_points[0][1], four_points[1][1]], linestyle='--', color=color[i])
            plt.plot([four_points[1][0], four_points[2][0]], [four_points[1][1], four_points[2][1]], linestyle='--', color=color[i])
            plt.plot([four_points[2][0], four_points[3][0]], [four_points[2][1], four_points[3][1]], linestyle='--', color=color[i])
    plt.savefig('./multipath_planning.pdf')
    plt.show()


def select_and_rename_snapshots_of_an_episode(logdir, epinum, num):
    file_list = os.listdir(logdir + '/episode{}'.format(epinum))
    file_num = len(file_list) - 1
    interval = file_num // (num-1)
    start = file_num % (num-1)
    # print(start, file_num, interval)
    selected = [start//2] + [start//2+interval*i for i in range(1, num-1)]
    # print(selected)
    if file_num > 0:
        for i, j in enumerate(selected):
            shutil.copyfile(logdir + '/episode{}/step{}.pdf'.format(epinum, j),
                            logdir + '/episode{}/figs/{}.pdf'.format(epinum, i))


if __name__ == '__main__':
    main()
    # plot_static_path()
    # plot_and_save_ith_episode_data('./results/good/2021-03-15-23-56-21', 0)
    # select_and_rename_snapshots_of_an_episode('./results/good/2021-03-15-23-56-21', 0, 12)


