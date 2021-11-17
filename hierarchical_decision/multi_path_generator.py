#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================
from dynamics_and_models import ReferencePath
from endtoend_env_utils import REF_NUM
import copy


class MultiPathGenerator(object):
    def __init__(self):
        self.path_list = []
        self.ref = ReferencePath(task='left')

    def generate_path(self, task, light_phase):
        task_path_num = REF_NUM[task]
        path_list = []
        for path_index in range(task_path_num):
            self.ref.set_path(task, light_phase, path_index)
            # print(ReferencePath.path_index)
            path_list.append(copy.deepcopy(self.ref))
        return path_list

