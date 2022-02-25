#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: load_policy.py
# =====================================
import argparse
import json

import tensorflow as tf
import numpy as np

from endtoend import CrossroadEnd2endMixPI
from utils.policy import Policy4Toyota
from utils.preprocessor import Preprocessor


class LoadPolicy(object):
    def __init__(self, exp_dir, iter):
        model_dir = exp_dir + '/models'
        parser = argparse.ArgumentParser()
        params = json.loads(open(exp_dir + '/config.json').read())
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        self.args = parser.parse_args()
        env = CrossroadEnd2endMixPI()
        self.policy = Policy4Toyota(self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args,
                                         gamma=self.args.gamma)
        # self.preprocessor.load_params(load_dir)
        init_obs, _ = env.reset()
        init_obs = init_obs[np.newaxis, :]
        self.run_batch(init_obs)
        self.obj_value_batch(init_obs)

    @tf.function
    def run_batch(self, mb_obs):
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        mb_state = self.get_states(processed_mb_obs)
        actions, _ = self.policy.compute_action(mb_state)
        return actions

    @tf.function
    def obj_value_batch(self, mb_obs):
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        mb_state = self.get_states(processed_mb_obs)
        values = self.policy.compute_obj_v(tf.stop_gradient(mb_state))
        return values

    def get_states(self, mb_obs, mb_mask=None):
        mb_obs_other = tf.reshape(mb_obs[:, self.args.state_other_start_dim:],
                                       (-1, self.args.max_other_num, self.args.per_other_dim))
        mb_obs_other_encode = self.policy.compute_pi_encode(mb_obs_other, mb_mask)
        mb_state = tf.concat((mb_obs[:, :self.args.state_other_start_dim], mb_obs_other_encode), axis=1)
        return mb_state
