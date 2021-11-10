#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: preprocessor.py
# =====================================

import numpy as np
import tensorflow as tf
import math


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    '''
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    '''
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def np_rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
    transformed_x = orig_x * np.cos(coordi_rotate_d_in_rad) + orig_y * np.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * np.sin(coordi_rotate_d_in_rad) + orig_y * np.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    while np.any(transformed_d>180):
        transformed_d = np.where(transformed_d>180, transformed_d - 360, transformed_d)
    while np.any(transformed_d <= -180):
        transformed_d = np.where(transformed_d <= -180, transformed_d + 360, transformed_d)
    return transformed_x, transformed_y, transformed_d


def tf_rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
    transformed_x = orig_x * tf.cos(coordi_rotate_d_in_rad) + orig_y * tf.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * tf.sin(coordi_rotate_d_in_rad) + orig_y * tf.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    while tf.reduce_any(transformed_d > 180):
        transformed_d = tf.where(transformed_d > 180, transformed_d - 360, transformed_d)
    while tf.reduce_any(transformed_d <= -180):
        transformed_d = tf.where(transformed_d <= -180, transformed_d + 360, transformed_d)
    return transformed_x, transformed_y, transformed_d


def np_shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = np_rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def tf_shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = tf_rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.tf_mean = tf.Variable(tf.zeros(shape), dtype=tf.float32, trainable=False)
        self.tf_var = tf.Variable(tf.ones(shape), dtype=tf.float32, trainable=False)

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        self.tf_mean.assign(tf.constant(self.mean))
        self.tf_var.assign(tf.constant(self.var))

    def set_params(self, mean, var, count):
        self.mean = mean
        self.var = var
        self.count = count
        self.tf_mean.assign(tf.constant(self.mean))
        self.tf_var.assign(tf.constant(self.var))

    def get_params(self, ):
        return self.mean, self.var, self.count


class Preprocessor(object):
    def __init__(self, obs_scale=None, rew_scale=None, rew_shift=None, args=None, **kwargs):
        self.obs_scale = obs_scale
        self.rew_scale = rew_scale
        self.rew_shift = rew_shift
        self.args = args

    def convert_ego_coordinate(self, obs):
        obs_ego = obs[:, :self.args.other_start_dim]
        obses_ego_all = np.reshape(np.tile(obs_ego, (1, self.args.other_number)), (-1, self.args.other_start_dim))
        obs_other = np.reshape(obs[:, self.args.other_start_dim:], (-1, self.args.per_other_dim))

        transformed_x, transformed_y, transformed_d = np_shift_and_rotate_coordination(obs_other[:, 0], obs_other[:, 1], obs_other[:, 3],
                                                                                    obses_ego_all[:, 3], obses_ego_all[:, 4], obses_ego_all[:, 5])
        obs_other_transformed = np.stack([transformed_x, transformed_y, obs_other[:, 2], transformed_d], axis=-1)
        obs_other_transformed = np.concatenate([obs_other_transformed, obs_other[:, 4:]], axis=1)
        obs_other_reshaped = np.reshape(obs_other_transformed, (-1, self.args.per_other_dim * self.args.other_number))
        obs_transformed = np.concatenate([obs_ego, obs_other_reshaped], axis=1)
        return np.squeeze(obs_transformed)

    def tf_convert_ego_coordinate(self, obs):
        obs_ego = obs[:, :self.args.other_start_dim]
        obses_ego_all = tf.reshape(tf.tile(obs_ego, (1, self.args.other_number)), (-1, self.args.other_start_dim))
        obs_other = tf.reshape(obs[:, self.args.other_start_dim:], (-1, self.args.per_other_dim))

        transformed_x, transformed_y, transformed_d = tf_shift_and_rotate_coordination(obs_other[:, 0], obs_other[:, 1], obs_other[:, 3],
                                                                                    obses_ego_all[:, 3], obses_ego_all[:, 4], obses_ego_all[:, 5])

        obs_other_transformed = tf.stack([transformed_x, transformed_y, obs_other[:, 2], transformed_d], axis=-1)
        obs_other_transformed = tf.concat([obs_other_transformed, obs_other[:, 4:]], axis=1)
        obs_other_reshaped = tf.reshape(obs_other_transformed, (-1, self.args.per_other_dim * self.args.other_number))
        obs_transformed = tf.concat([obs_ego, obs_other_reshaped], axis=1)
        return obs_transformed

    def process_rew(self, rew):
        if self.rew_scale:
            return (rew + self.rew_shift) * self.rew_scale
        else:
            return rew

    def process_obs(self, obs):
        if self.obs_scale:
            return obs * self.obs_scale
        else:
            return obs

    def np_process_obses(self, obses):
        if self.obs_scale:
            return obses * self.obs_scale
        else:
            return obses

    def tf_process_obses(self, obses):
        if self.obs_scale:
            return obses * tf.convert_to_tensor(self.obs_scale, dtype=tf.float32)
        else:
            return tf.convert_to_tensor(obses, dtype=tf.float32)

    def np_process_rewards(self, rewards):
        if self.rew_scale:
            return (rewards + self.rew_shift) * self.rew_scale
        else:
            return rewards

    def tf_process_rewards(self, rewards):
        if self.rew_scale:
            return (rewards+tf.convert_to_tensor(self.rew_shift, dtype=tf.float32)) \
                   * tf.convert_to_tensor(self.rew_scale, dtype=tf.float32)
        else:
            return tf.convert_to_tensor(rewards, dtype=tf.float32)
