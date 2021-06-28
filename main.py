#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：EDataSaver
@File ：edatasaver.py
@Author ：septemberhx
@Date ：2021/4/28
@Description:
"""

import csv
import datetime
import os
import json
from greedy.AxAlgo import AxAlgo
from commom.logger import get_logger

logger = get_logger('Main')


def read_from_files(dir_path):
    if not os.path.isdir(dir_path):
        print(f'{dir_path} is not a directory')
        return

    def jsonKeys2int(x):
        if isinstance(x, dict):
            return {int(k): v for k, v in x.items()}
        return x

    with open(os.path.join(dir_path, 'svc_objs.json'), 'r') as f:
        svc_objs = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
    with open(os.path.join(dir_path, 'node_objs.json'), 'r') as f:
        node_objs = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
    with open(os.path.join(dir_path, 'func_objs.json'), 'r') as f:
        func_objs = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
    with open(os.path.join(dir_path, 'users.json'), 'r') as f:
        users = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
    with open(os.path.join(dir_path, 'chains.json'), 'r') as f:
        chains = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
    with open(os.path.join(dir_path, 'chain_list.json'), 'r') as f:
        chain_list = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
    with open(os.path.join(dir_path, 'connections.json'), 'r') as f:
        connections = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
    with open(os.path.join(dir_path, 'prices.json'), 'r') as f:
        prices = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})

    return svc_objs, node_objs, func_objs, users, chains, chain_list, connections, prices


def run_single(exp_name, data_dir_path, method):
    items = os.listdir(data_dir_path)
    items.sort(key=lambda x: int(x))
    headers = ['X', method]
    rows = []
    for item in items:
        path = os.path.join(data_dir_path, item)
        if not os.path.isdir(path):
            continue

        logger.info(f'======> Run Experiments with {item} r.w.t {exp_name} <======')
        svc_objs, node_objs, func_objs, users, chains, chain_list, connections, prices = read_from_files(path)

        result = 0
        if method == 'DFS':
            logger.info(f'oooooo> Run with AxAlgo(1):')
            ax_algo = AxAlgo(svc_objs, node_objs, func_objs, users, chains, chain_list, connections, prices)
            result = ax_algo.solve(1)
        elif method == 'BFS':
            logger.info(f'oooooo> Run with AxAlgo(2):')
            ax_algo = AxAlgo(svc_objs, node_objs, func_objs, users, chains, chain_list, connections, prices)
            result = ax_algo.solve(2)

        rows.append([item, result])
        logger.info(f'======>              {item} finished                <======')
    with open(f'./{exp_name}_{method}_{datetime.datetime.now()}.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
        logger.info(f'Result locates at {f.name}')


if __name__ == '__main__':
    # prepare your data according to the readme.md
    # check read_from_files() to rename your data file
    run_single('experiment name', 'data dir path', 'DFS')
