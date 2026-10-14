#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
# pylint: disable=unspecified-encoding

import os
import stat
import json
import argparse
import yaml


def dump_objective(objective_key, objective_value):
    json_path = './{}.json'.format(objective_key)
    if os.path.exists(json_path):
        os.remove(json_path)
    with os.fdopen(os.open(json_path, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fout:
        json.dump({objective_key: objective_value}, fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--config_file", type=str)
    opt = parser.parse_args()
    with open(opt.config_file) as f:
        config_yml = yaml.safe_load(f)
    batch_size = config_yml.get("batch_size")
    dump_objective('lr', opt.lr)
    dump_objective('batch_size', batch_size)
    dump_objective('accuracy', 0.8)
    dump_objective('latency', 10)


if __name__ == '__main__':
    main()
