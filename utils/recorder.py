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
# sns.set(style="darkgrid")

WINDOWSIZE = 15


class Recorder(object):
    def __init__(self):
        self.val2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi',
                           'steer', 'a_x', 'delta_y', 'delta_v', 'delta_phi',
                           'cal_time', 'ref_index', 'beta', 'path_values', 'ss_time', 'is_ss']
        self.val2plot = ['v_x', 'phi', 'r',
                         'steer', 'a_x',
                         'cal_time', 'ref_index', 'beta', 'path_values', 'is_ss']
        self.key2label = dict(v_x='Speed [m/s]',
                              phi='Heading angle [$\circ$]',
                              r='Yaw rate [rad/s]',
                              steer='Steer angle [$\circ$]',
                              a_x='Acceleration [$\mathrm {m/s^2}$]',
                              cal_time='Computing time [ms]',
                              ref_index='Selected path',
                              beta='Side slip angle[$\circ$]',
                              path_values='Path value',
                              is_ss='Safety shield',)

        self.comp2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi', 'adp_steer', 'adp_a_x', 'mpc_steer', 'mpc_a_x',
                            'delta_y', 'delta_v', 'delta_phi', 'adp_time', 'mpc_time', 'adp_ref', 'mpc_ref', 'beta']

        self.ego_info_dim = 6
        self.per_tracking_info_dim = 3
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
        delta_y, delta_phi, delta_v = tracking_info[:3]
        steer, a_x = act[0]*0.4, act[1]*2.25 - 0.75

        # transformation
        beta = 0 if v_x == 0 else np.arctan(v_y/v_x) * 180 / math.pi
        steer = steer * 180 / math.pi
        self.val_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, steer, a_x, delta_y,
                                        delta_phi, delta_v, cal_time, ref_index, beta, path_values, ss_time, is_ss]))

    # For comparison of MPC and ADP
    def record_compare(self, obs, adp_act, mpc_act, adp_time, mpc_time, adp_ref, mpc_ref, mode='ADP'):
        ego_info, tracking_info, _ = obs[:self.ego_info_dim], \
                                     obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1)], \
                                     obs[self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1):]
        v_x, v_y, r, x, y, phi = ego_info
        delta_y, delta_phi, delta_v = tracking_info[:3]
        adp_steer, adp_a_x = adp_act[0]*0.4, adp_act[1]*2.25 - 0.75
        mpc_steer, mpc_a_x = mpc_act[0], mpc_act[1]

        # transformation
        beta = 0 if v_x == 0 else np.arctan(v_y/v_x) * 180 / math.pi
        adp_steer = adp_steer * 180 / math.pi
        mpc_steer = mpc_steer * 180 / math.pi
        self.comp_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, adp_steer, adp_a_x, mpc_steer, mpc_a_x,
                                            delta_y, delta_phi, delta_v, adp_time, mpc_time, adp_ref, mpc_ref, beta]))

    def save(self, logdir):
        np.save(logdir + '/data_across_all_episodes.npy', np.array(self.data_across_all_episodes))
        np.save(logdir + '/comp_data_for_all_episodes.npy', np.array(self.comp_data_for_all_episodes))

    def load(self, logdir):
        self.data_across_all_episodes = np.load(logdir + '/data_across_all_episodes.npy', allow_pickle=True)
        self.comp_data_for_all_episodes = np.load(logdir + '/comp_data_for_all_episodes.npy', allow_pickle=True)

    def plot_and_save_ith_episode_curves(self, i, save_dir, isshow=False):
        episode2plot = self.data_across_all_episodes[i]
        real_time = np.array([0.1*i for i in range(len(episode2plot))])
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
                    for index in range(len(self.val2record))]
        data_dict = dict(zip(self.val2record, all_data))
        color = ['cyan', 'indigo', 'magenta', 'coral', 'b', 'brown', 'c']
        i = 0

        f = plt.figure(0, figsize=(8, 1.6))
        ax1 = f.add_axes([0.12, 0.35, 0.8, 0.6])
        df = pd.DataFrame(dict(time=real_time, steer=data_dict['steer'], a_x=data_dict['a_x'],))
        df['steer'] = df['steer'].rolling(WINDOWSIZE, min_periods=1).mean()
        df['a_x'] = df['a_x'].rolling(1, min_periods=1).mean()

        l1 = plt.plot(real_time, df['steer'], linewidth='1.5', color='r')
        l2 = plt.plot(real_time, df['a_x'], linewidth='1.5', color='k', ls='--')
        plt.legend(labels=['steer', 'acceleration'], bbox_to_anchor=(0.10, 0.98),
                   loc='upper left', ncol=2, frameon=False)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel('$\delta$[$\circ$]',  fontsize=12)
        ax1.yaxis.set_label_coords(-0.11, 0.4)
        ax2 = ax1.twinx()
        ax2.set_yticks([-10, 0, 10])
        ax2.set_yticklabels(('-4', '0', '4'), fontsize=12)
        plt.ylabel('a[$\mathrm {m/s^2}$]', fontsize=12)
        ax2.yaxis.set_label_coords(1.05, 0.35)
        # ax2.set_ylim([-3, 3])
        plt.savefig(save_dir + '/{}.pdf'.format('demo-steer-acc'))

        f = plt.figure(1, figsize=(8, 1.6))
        ax1 = f.add_axes([0.12, 0.35, 0.8, 0.6])
        df = pd.DataFrame(dict(time=real_time, v_x=data_dict['v_x'], phi=data_dict['phi'],))
        df['v_x'] = df['v_x'].rolling(1, min_periods=1).mean()
        df['phi'] = df['phi'].rolling(1, min_periods=1).mean()
        df['phi'] = [i if i > 0 else i + 360 for i in df['phi']]
        df['phi'] = df['phi'] * math.pi / 180
        l1 = plt.plot(real_time, df['v_x'], linewidth='1.5', color='r')
        l2 = plt.plot(real_time, df['phi'], linewidth='1.5', color='k', ls='--')
        plt.legend(labels=['speed', 'heading angle'], loc='upper center', ncol=2, frameon=False)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel(r"$v_x$ $\mathrm{[m/s]}$",  fontsize=12)
        ax1.yaxis.set_label_coords(-0.11, 0.4)
        ax1.set_ylim([0, 8])
        ax2 = ax1.twinx()
        ax2.set_yticks([0, 1.57, 3.14])
        ax2.set_yticklabels(('0', '90', '180'), fontsize=12)
        plt.ylabel('r[$\circ$]', fontsize=12)
        ax2.yaxis.set_label_coords(1.065, 0.45)
        ax2.set_ylim(ax1.get_ylim())
        plt.savefig(save_dir + '/{}.pdf'.format('demo-speed-heading'))
        plt.show()

        for key in data_dict.keys():
            if key in self.val2plot:
                f = plt.figure(key, figsize=(6, 5))
                if key == 'ref_index':
                    ax = f.add_axes([0.12, 0.15, 0.88, 0.85])
                    sns.lineplot(real_time, data_dict[key] + 1, linewidth=2, palette="bright", color='indigo')
                    plt.ylim([0.5, 3.5])
                    x_major_locator = MultipleLocator(10)
                    # ax.xaxis.set_major_locator(x_major_locator)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                elif key == 'v_x':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.15, 0.85, 0.85])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-0.5, 10.])
                elif key == 'cal_time':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key] * 1000))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.15, 0.85, 0.85])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([0, 10])
                elif key == 'a_x':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.16, 0.15, 0.84, 0.85])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-4.5, 2.0])
                elif key == 'steer':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.18, 0.15, 0.82, 0.85])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-25, 25])
                elif key == 'beta':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.12, 0.85, 0.86])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-15, 15])
                elif key == 'r':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.12, 0.85, 0.86])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-0.8, 0.8])
                elif key == 'path_values':
                    path_values = data_dict[key]
                    df1 = pd.DataFrame(dict(time=real_time, data=-path_values[:, 0], path_index='Path 1'))
                    df2 = pd.DataFrame(dict(time=real_time, data=-path_values[:, 1], path_index='Path 2'))
                    df3 = pd.DataFrame(dict(time=real_time, data=-path_values[:, 2], path_index='Path 3'))
                    total_dataframe = df1.append([df2, df3], ignore_index=True)
                    ax = f.add_axes([0.18, 0.15, 0.82, 0.85])
                    sns.lineplot('time', 'data', linewidth=2, hue='path_index',
                                 data=total_dataframe, palette="bright", color='indigo')
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles=handles, labels=labels, loc='lower left', frameon=False, fontsize=20)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                elif key == 'is_ss':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    ax = f.add_axes([0.12, 0.15, 0.88, 0.85])
                    sns.lineplot('time', 'data', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                else:
                    ax = f.add_axes([0.11, 0.12, 0.88, 0.86])
                    sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color='indigo')

                # for a specific simu with red light (2021-03-15-23-56-21)
                # ylim = ax.get_ylim()
                # ax.add_patch(patches.Rectangle((0, ylim[0]), 5, ylim[1]-ylim[0], facecolor='red', alpha=0.1))
                # ax.add_patch(patches.Rectangle((5, ylim[0]), 3, ylim[1]-ylim[0], facecolor='orange', alpha=0.1))
                # ax.add_patch(patches.Rectangle((8, ylim[0]), 23.6-8+1, ylim[1]-ylim[0], facecolor='green', alpha=0.1))

                # ax.add_patch(patches.Rectangle((0., ylim[0]), 16, ylim[1] - ylim[0], facecolor='r', alpha=0.1))
                # ax.add_patch(patches.Rectangle((16., ylim[0]), 5, ylim[1] - ylim[0], facecolor='orange', alpha=0.1))
                # ax.add_patch(patches.Rectangle((21., ylim[0]), 32, ylim[1] - ylim[0], facecolor='g', alpha=0.1))

                ax.set_ylabel(self.key2label[key], fontsize=20)
                ax.set_xlabel("Time [s]", fontsize=20)
                plt.yticks(fontsize=20)
                plt.xticks(fontsize=20)
                plt.savefig(save_dir + '/{}.pdf'.format(key))
                if not isshow:
                    plt.close(f)
                i += 1
        if isshow:
            plt.show()

        # # for paper plot
        # self.data_across_all_episodes_fp = np.load('./results/2021-09-21-15-09-43/data_across_all_episodes.npy', allow_pickle=True)
        # self.data_across_all_episodes_dp = np.load('./results/2021-09-21-16-43-51/data_across_all_episodes.npy', allow_pickle=True)
        #
        # acc_dp, steer_dp = [], []
        # for episode2plot in self.data_across_all_episodes_dp[:]:
        #     real_time = np.array([0.1 * i for i in range(len(episode2plot))])
        #     all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
        #                 for index in range(len(self.val2record))]
        #     data_dict = dict(zip(self.val2record, all_data))
        #     acc_dp.extend([data_dict['a_x'][i+1] - data_dict['a_x'][i] for i in range(len(data_dict['a_x'])-1)])
        #     steer_dp.extend([data_dict['steer'][i+1] - data_dict['steer'][i] for i in range(len(data_dict['steer'])-1)])
        #
        # df_dp = pd.DataFrame({'algorithms': 'RL w. DP',
        #                    'steer': steer_dp,
        #                    'acc': acc_dp,
        #                        })
        #
        # acc_fp, steer_fp = [], []
        # for episode2plot in self.data_across_all_episodes_fp[:]:
        #     real_time = np.array([0.1 * i for i in range(len(episode2plot))])
        #     all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
        #                 for index in range(len(self.val2record))]
        #     data_dict = dict(zip(self.val2record, all_data))
        #     acc_fp.extend([data_dict['a_x'][i+1] - data_dict['a_x'][i] for i in range(len(data_dict['a_x'])-1)])
        #     steer_fp.extend([data_dict['steer'][i+1] - data_dict['steer'][i] for i in range(len(data_dict['steer'])-1)])
        #
        # df_fp = pd.DataFrame({'algorithms': 'RL w. FP',
        #                    'steer': steer_fp,
        #                    'acc': acc_fp,
        #                        })
        # df = df_dp.append(df_fp, ignore_index=True)
        # f = plt.figure(22, figsize=(6, 5))
        # ax = f.add_axes([0.12, 0.15, 0.88, 0.85])
        # sns.jointplot(x="acc", y="steer", data=df, hue="algorithms", kind="hist")
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=handles[1:], labels=labels[1:], loc='upper right', frameon=False, fontsize=25)
        # plt.ylim([-5, 5])
        # plt.xlim([-2, 2])
        # plt.show()

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

        df_rl = pd.DataFrame({'algorithms': 'GEP',
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
        f1 = plt.figure(figsize=(6.2,5.2))
        ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.80])
        sns.lineplot(x="iteration", y="steer", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        # ax1.set_title('Front wheel angle [$\circ$]', fontsize=20)
        ax1.set_ylabel('Front wheel angle [$\circ$]', fontsize=20)
        ax1.set_xlabel("Time [s]", fontsize=20)
        ax1.legend(frameon=False, fontsize=20)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles[:], labels=labels[:], frameon=False, fontsize=20)
        # ax1.get_legend().remove()
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        f1.savefig(save_dir + '/steer.pdf')
        plt.close() if not isshow else plt.show()

        f2 = plt.figure(figsize=(6.2, 5.2))
        ax2 = f2.add_axes([0.155, 0.12, 0.82, 0.80])
        sns.lineplot(x="iteration", y="acc", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        # ax2.set_title('Acceleration [$\mathrm {m/s^2}$]', fontsize=20)
        ax2.set_ylabel('Acceleration [$\mathrm {m/s^2}$]', fontsize=20)
        ax2.set_xlabel('Time [s]', fontsize=20)
        ax2.legend(frameon=False, fontsize=20)
        ax2.get_legend().remove()
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.savefig(save_dir + '/acceleration.pdf')
        plt.close() if not isshow else plt.show()

        f3 = plt.figure(figsize=(6.2,5.2))
        ax3 = f3.add_axes([0.155, 0.12, 0.82, 0.86])
        sns.lineplot(x="iteration", y="ref_path", hue="algorithms", data=total_df, dashes=True, linewidth=2, palette="bright")
        ax3.lines[1].set_linestyle("--")
        ax3.set_ylabel('Selected path', fontsize=20)
        ax3.set_xlabel("Time [s]", fontsize=20)
        ax3.legend(frameon=False, fontsize=20)
        ax3.get_legend().remove()
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.savefig(save_dir + '/ref_path.pdf')
        plt.close() if not isshow else plt.show()

        f4 = plt.figure(figsize=(6.2, 5.2))
        ax4 = f4.add_axes([0.155, 0.12, 0.82, 0.80])
        sns.lineplot(x="iteration", y="time", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        plt.yscale('log')
        # ax3.set_title('Computing time [ms]', fontsize=20)
        ax4.set_xlabel("Time [s]", fontsize=20)
        ax4.set_ylabel('Computing time [ms]', fontsize=20)
        handles, labels = ax4.get_legend_handles_labels()
        ax4.legend(handles=handles[:], labels=labels[:], frameon=False, fontsize=20)
        ax4.get_legend().remove()
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.savefig(save_dir + '/time.pdf')
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
        plt.xticks(fontsize=15)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=labels[:], frameon=False, fontsize=15)
        plt.show()






