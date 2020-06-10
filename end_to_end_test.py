#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: end_to_end_test.py
# =====================================

import numpy as np

from endtoend import CrossroadEnd2end


def action_fn(obs):
    # grid_list, supplement_vector = obs
    return np.array([0, 0])


# class Testt:
#     def __init__(self):
#         pass
#
#     def testtt(self, b):
#         self.a = 1
#         b = b
#         def fn():
#             return b
#         return fn
#
#     def change_a(self, x):
#         fn = self.testtt(x)
#         self.a = fn()
#         return self.a


if __name__ == '__main__':
    env = CrossroadEnd2end(frameskip=1, training_task='right')
    done = 0
    episode_num = 10
    for i in range(episode_num):  # run episode_num episodes
        done = 0
        obs = env.reset()
        ret = 0
        ite = 0
        while not done:
            ite += 1
            obs, rew, done, info = env.step([np.random.random()*2-1, 1])
            env.render()
            ret += rew
            print('reward: ', rew)

        print('return: ', ret)


