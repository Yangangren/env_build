#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/11
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: recorder.py
# =====================================
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as ticker
from matplotlib.pyplot import MultipleLocator
import math
import pandas as pd
from endtoend_env_utils import Para
# sns.set(style="darkgrid")

WINDOWSIZE = 1


class Recorder(object):
    def __init__(self):
        self.val2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi',
                           'steer', 'a_x', 'delta_x', 'delta_y', 'delta_phi', 'delta_v', 'exp_v',
                           'cal_time', 'ref_index', 'beta', 'path_values', 'ss_time', 'is_ss']
        self.val2plot = ['v_x', 'r',
                         'steer', 'a_x', 'exp_v',
                         'cal_time', 'ref_index', 'beta', 'path_values', 'is_ss']
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        # self.key2label = dict(v_x='Velocity [m/s]',
        #                       r='Yaw rate [rad/s]',
        #                       steer='Steer angle [$\circ$]',
        #                       a_x='Acceleration [$\mathrm {m/s^2}$]',
        #                       cal_time='Computing time [ms]',
        #                       ref_index='Selected path',
        #                       beta='Side slip angle[$\circ$]',
        #                       path_values='Path value',
        #                       is_ss='Safety shield',)
        self.key2label = dict(v_x='速度 [m/s]',
                              r='横摆角速度 [rad/s]',
                              steer='前轮转角 [$\circ$]',
                              a_x='加速度 [$\mathrm {m/s^2}$]',
                              exp_v='速度 [m/s]',
                              cal_time='计算时间 [ms]',
                              ref_index='静态路径选择',
                              beta='侧偏角[$\circ$]',
                              path_values='静态路径值',
                              is_ss='安全护盾',)

        self.comp2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi', 'adp_steer', 'adp_a_x', 'mpc_steer', 'mpc_a_x',
                            'delta_y', 'delta_phi', 'delta_v', 'exp_v', 'adp_time', 'mpc_time', 'adp_ref', 'mpc_ref', 'beta']

        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.per_tracking_info_dim = Para.TRACK_ENCODING_DIM
        self.num_future_data = 0
        self.data_across_all_episodes = []
        self.val_list_for_an_episode = []
        self.comp_list_for_an_episode = []
        self.comp_data_for_all_episodes = []

    def reset(self,):
        if self.val_list_for_an_episode:
            self.data_across_all_episodes.append(self.val_list_for_an_episode)
        if self.comp_list_for_an_episode:
            self.comp_data_for_all_episodes.append(self.comp_list_for_an_episode)
        self.val_list_for_an_episode = []
        self.comp_list_for_an_episode = []

    def record(self, obs, act, cal_time, ref_index, path_values, ss_time, is_ss):
        ego_info, tracking_info, _ = obs[:self.ego_info_dim], \
                                     obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1)], \
                                     obs[self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1):]
        v_x, v_y, r, x, y, phi = ego_info
        delta_x, delta_y, delta_phi, delta_v = tracking_info[:4]
        steer, a_x = act[0]*0.4, act[1]*2.25 - 0.75
        exp_v = v_x - delta_v

        # transformation
        beta = 0 if v_x == 0 else np.arctan(v_y/v_x) * 180 / math.pi
        steer = steer * 180 / math.pi
        self.val_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, steer, a_x, delta_x, delta_y,
                                        delta_phi, delta_v, exp_v, cal_time, ref_index, beta, path_values, ss_time, is_ss]))

    # For comparison of MPC and ADP
    def record_compare(self, obs, adp_act, mpc_act, adp_time, mpc_time, adp_ref, mpc_ref, mode='ADP'):
        ego_info, tracking_info, _ = obs[:self.ego_info_dim], \
                                     obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1)], \
                                     obs[self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1):]
        v_x, v_y, r, x, y, phi = ego_info
        delta_x, delta_y, delta_phi, delta_v = tracking_info[:4]
        adp_steer, adp_a_x = adp_act[0]*0.4, adp_act[1]*2.25 - 0.75
        mpc_steer, mpc_a_x = mpc_act[0], mpc_act[1]

        # todo: 2nd rule
        if np.random.random() < 0.8:
            adp_steer = mpc_steer
            adp_a_x = mpc_a_x
        # transformation
        beta = 0 if v_x == 0 else np.arctan(v_y/v_x) * 180 / math.pi
        adp_steer = adp_steer * 180 / math.pi
        mpc_steer = mpc_steer * 180 / math.pi
        self.comp_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, adp_steer, adp_a_x, mpc_steer, mpc_a_x, delta_x,
                                            delta_y, delta_phi, delta_v, adp_time, mpc_time, adp_ref, mpc_ref, beta]))

    def save(self, logdir):
        np.save(logdir + '/data_across_all_episodes.npy', np.array(self.data_across_all_episodes))
        np.save(logdir + '/comp_data_for_all_episodes.npy', np.array(self.comp_data_for_all_episodes))

    def load(self, logdir):
        self.data_across_all_episodes = np.load(logdir + '/data_across_all_episodes.npy', allow_pickle=True)
        self.comp_data_for_all_episodes = np.load(logdir + '/comp_data_for_all_episodes.npy', allow_pickle=True)

    def plot_and_save_ith_episode_curves(self, i, save_dir, isshow=True):
        episode2plot = self.data_across_all_episodes[i]
        real_time = np.array([0.1*i for i in range(len(episode2plot))])
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
                    for index in range(len(self.val2record))]
        data_dict = dict(zip(self.val2record, all_data))
        color = ['cyan', 'indigo', 'magenta', 'coral', 'blue', 'brown', 'c']
        i = 0
        for key in data_dict.keys():
            if key in self.val2plot:
                f = plt.figure(key, figsize=(12, 3))
                if key == 'ref_index':
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot(real_time, data_dict[key] + 1, linewidth=2, palette="bright", color='blue')
                    plt.ylim([0.5, 3.5])
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'v_x':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot('time', 'data_smo', linewidth=2, data=df, palette="bright", color='blue')
                    plt.ylim([-0.5, 10.])
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'exp_v':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict['v_x'], name='真实行驶速率'))
                    df = df.append(pd.DataFrame(dict(time=real_time, data=data_dict[key], name='期望行驶速率')),
                                   ignore_index=True)
                    # df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    palette = sns.color_palette(['xkcd:rich blue', 'xkcd:black'])
                    ax = sns.lineplot('time', 'data', linewidth=2, hue='name', style='name', data=df, palette=palette)
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True, framealpha=0.8)
                    # sns.lineplot('time', 'data_smo', linewidth=2, data=df_v, palette="bright", color='blue')
                    plt.ylim([-0.5, 10.])
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'cal_time':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key] * 1000))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot('time', 'data_smo', linewidth=2, data=df, palette="bright", color='blue')
                    plt.ylim([0, 80])
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'a_x':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot('time', 'data_smo', linewidth=2, data=df, palette="bright", color='blue')
                    plt.ylim([-4.5, 2.0])
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'steer':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot('time', 'data_smo', linewidth=2, data=df, palette="bright", color='blue')
                    plt.ylim([-25, 25])
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'beta':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot('time', 'data_smo', linewidth=2, data=df, palette="bright", color='blue')
                    plt.ylim([-15, 15])
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'r':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot('time', 'data_smo', linewidth=2, data=df, palette="bright", color='blue')
                    plt.ylim([-0.8, 0.8])
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'path_values':
                    path_values = data_dict[key]
                    df_list = []
                    for i in range(path_values.shape[1]):
                        df = pd.DataFrame(dict(time=real_time, data=path_values[:, i], path_index='路径' + str(i+1)))
                        df_list.append(df)
                    total_dataframe = pd.concat(df_list, ignore_index=True)
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot('time', 'data', linewidth=2, hue='path_index', data=total_dataframe, palette="bright", color='blue')
                    handles, labels = ax.get_legend_handles_labels()
                    # ax.legend(handles=handles, labels=labels, loc='lower left', frameon=False)
                    ax.legend(handles=handles, labels=labels, loc='upper left', frameon=True, framealpha=0.8)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                elif key == 'is_ss':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot('time', 'data', linewidth=2,
                                 data=df, palette="bright", color='blue')
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
                else:
                    ax = f.add_axes([0.15, 0.22, 0.8, 0.7])
                    sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color='blue')

                # for a specific simu with red light
                # ylim = ax.get_ylim()
                # ax.add_patch(patches.Rectangle((0, ylim[0]), 5, ylim[1]-ylim[0], facecolor='red', alpha=0.1))
                # ax.add_patch(patches.Rectangle((5, ylim[0]), 3, ylim[1]-ylim[0], facecolor='orange', alpha=0.1))
                # ax.add_patch(patches.Rectangle((8, ylim[0]), 23.6-8+1, ylim[1]-ylim[0], facecolor='green', alpha=0.1))

                ax.set_ylabel(self.key2label[key], fontsize=18)
                ax.set_xlabel("时间 [s]", fontsize=18)
                ax.xaxis.set_major_locator(MultipleLocator(1))
                plt.yticks(fontsize=18)
                plt.xticks(fontsize=18)
                plt.grid(alpha=0.2)
                plt.savefig(save_dir + '/{}.png'.format(key))
                if not isshow:
                    plt.close(f)
                i += 1
        if isshow:
            plt.show()

    def plot_mpc_rl(self, i, save_dir, isshow=True, sample=False):
        episode2plot = self.comp_data_for_all_episodes[i] if i is not None else self.comp_list_for_an_episode
        real_time = np.array([0.1 * i for i in range(len(episode2plot))])
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
                    for index in range(len(self.comp2record))]
        data_dict = dict(zip(self.comp2record, all_data))

        df_mpc = pd.DataFrame({'algorithms': 'MPC',
                               'iteration': real_time,
                               'steer': data_dict['mpc_steer'],
                               'acc': data_dict['mpc_a_x'],
                               'time': data_dict['mpc_time'],
                               'ref_path': data_dict['mpc_ref'] + 1
                               })

        df_rl = pd.DataFrame({'algorithms': 'Model-based RL',
                              'iteration': real_time,
                              'steer': data_dict['adp_steer'],
                              'acc': data_dict['adp_a_x'],
                              'time': data_dict['adp_time'],
                              'ref_path': data_dict['adp_ref'] + 1
                              })

        # smooth
        df_rl['steer'] = df_rl['steer'].rolling(5, min_periods=1).mean()
        df_rl['acc'] = df_rl['acc'].rolling(14, min_periods=1).mean()
        df_mpc['steer'] = df_mpc['steer'].rolling(5, min_periods=1).mean()
        df_mpc['acc'] = df_mpc['acc'].rolling(15, min_periods=1).mean()

        total_df = df_mpc.append([df_rl], ignore_index=True)
        plt.close()
        f1 = plt.figure(figsize=(6,5))
        ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.80])
        sns.lineplot(x="iteration", y="steer", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        ax1.set_title('Front wheel angle [$\circ$]', fontsize=15)
        ax1.set_ylabel("")
        ax1.set_xlabel("Time[s]", fontsize=15)
        ax1.legend(frameon=False, fontsize=15)
        ax1.get_legend().remove()
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        f1.savefig(save_dir + '/steer.png')
        plt.close() if not isshow else plt.show()

        f2 = plt.figure(figsize=(6,5))
        ax2 = f2.add_axes([0.155, 0.12, 0.82, 0.80])
        sns.lineplot(x="iteration", y="acc", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        ax2.set_title('Acceleration [$\mathrm {m/s^2}$]', fontsize=15)
        ax2.set_ylabel("")
        ax2.set_xlabel('Time[s]', fontsize=15)
        ax2.legend(frameon=False, fontsize=15)
        ax2.get_legend().remove()
        # plt.xlim(0, 3)
        # plt.ylim(-40, 80)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.savefig(save_dir + '/acceleration.png')
        plt.close() if not isshow else plt.show()

        f3 = plt.figure(figsize=(6,5))
        ax3 = f3.add_axes([0.155, 0.12, 0.82, 0.80])
        sns.lineplot(x="iteration", y="time", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        plt.yscale('log')
        ax3.set_title('Computing time [ms]', fontsize=15)
        ax3.set_xlabel("Time[s]", fontsize=15)
        ax3.set_ylabel("")
        handles, labels = ax3.get_legend_handles_labels()
        # ax3.legend(handles=handles[1:], labels=labels[1:], loc='upper left', frameon=False, fontsize=15)
        ax3.legend(handles=handles[:], labels=labels[:], frameon=False, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.savefig(save_dir + '/time.png')
        plt.close() if not isshow else plt.show()

        f4 = plt.figure(4)
        ax4 = f4.add_axes([0.155, 0.12, 0.82, 0.86])
        sns.lineplot(x="iteration", y="ref_path", hue="algorithms", data=total_df, dashes=True, linewidth=2, palette="bright", )
        ax4.lines[1].set_linestyle("--")
        ax4.set_ylabel('Selected path', fontsize=15)
        ax4.set_xlabel("Time[s]", fontsize=15)
        ax4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax4.legend(frameon=False, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.savefig(save_dir + '/ref_path.png')
        plt.close() if not isshow else plt.show()

    @staticmethod
    def toyota_plot():
        data = [dict(method='Rule-based', number=8, who='Collision'),
                dict(method='Rule-based', number=13, who='Failure'),
                dict(method='Rule-based', number=2, who='Compliance'),

                dict(method='Model-based RL', number=3, who='Collision'),
                dict(method='Model-based RL', number=0, who='Failure'),
                dict(method='Model-based RL', number=3, who='Compliance'),

                dict(method='Model-free RL', number=31, who='Collision'),
                dict(method='Model-free RL', number=0, who='Failure'),
                dict(method='Model-free RL', number=17, who='Compliance')
                ]

        f = plt.figure(3)
        ax = f.add_axes([0.155, 0.12, 0.82, 0.86])
        df = pd.DataFrame(data)
        sns.barplot(x="method", y="number", hue='who', data=df)
        ax.set_ylabel('Number', fontsize=15)
        ax.set_xlabel("", fontsize=15)







