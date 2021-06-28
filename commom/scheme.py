#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：AlgoGroupDeployment 
@File ：scheme.py
@Author ：septemberhx
@Date ：2021/3/30
@Description:
"""

import math
import networkx as nx

from typing import Dict, List

from commom.logger import get_logger

logger = get_logger('scheme')


def cal_min_svc_count(graph: nx.DiGraph, svc_objs) -> Dict[int, int]:
    """
    calculate minimum service instance count of each service
    :return: None
    """
    min_svc_count = {}
    for svc in graph.nodes:
        if svc < 0:
            continue
        total_freq = 0
        for pred_svc in graph.predecessors(svc):
            for func, freq in graph[pred_svc][svc]['attr'].items():
                total_freq += freq
        # logger.debug(f'Service {svc} has {total_freq} requests in total')
        min_svc_count[svc] = int(math.ceil(total_freq / svc_objs[svc]['ability']))
    return min_svc_count


def check_problem_is_feasible(svc_objs, node_objs, min_svc_count: Dict[int, int],
                              svc_prices: Dict[int, float], max_price: float):
    total_cpu_demand = 0.0
    total_ram_demand = 0.0
    total_price = 0.0
    for svc, count in min_svc_count.items():
        total_cpu_demand += svc_objs[svc]['res']['cpu'] * count
        total_ram_demand += svc_objs[svc]['res']['ram'] * count
        total_price += svc_prices[svc] * count

    total_cpu = 0.0
    total_ram = 0.0
    for node_obj in node_objs:
        total_cpu += node_obj['res']['cpu']
        total_ram += node_obj['res']['ram']

    if total_ram < total_ram_demand or total_cpu < total_cpu_demand:
        logger.info('Insufficient resources')
        return False
    if total_price > max_price:
        logger.info('Insufficient price')
        return False
    return True


def check_scheme_is_feasible(scheme: Dict[int, Dict[int, int]], svc_objs, node_objs, min_svc_count: Dict[int, int],
                             svc_prices: Dict[int, float], max_price: float):
    total_price = 0.0
    for svc, count in min_svc_count.items():
        svc_count = 0
        for node in scheme:
            svc_count += scheme[node][svc]
        if svc_count < count:
            logger.info(f'Service {svc} has {svc_count} but {count} is desired')
            return False
        total_price += svc_prices[svc] * svc_count
    if total_price > max_price:
        logger.info('Price > max price !')
        return False

    for node in scheme:
        used_cpu = 0.0
        used_ram = 0.0
        for svc, count in scheme[node].items():
            used_cpu += svc_objs[svc]['res']['cpu'] * count
            used_ram += svc_objs[svc]['res']['ram'] * count
            if used_cpu > node_objs[node]['res']['cpu'] or used_ram > node_objs[node]['res']['ram']:
                logger.info(f'Node {node} holds too many instances')
                return False
    return True


def evaluate(scheme: Dict[int, Dict[int, int]], node_objs: List[Dict], func_objs: List[Dict],
             users: Dict, chains: Dict[int, Dict], connections) -> float:
    """
    evaluate the average response time of the given deployment scheme
    :return:
    """
    # prepare data for calculating probability
    total_freq = 0
    func_freq = {}
    for node in users:
        for func_t, freq_t in users[node].items():
            total_freq += freq_t
            if func_t not in func_freq:
                func_freq[func_t] = 0
            func_freq[func_t] += freq_t

    svc_inst_count = {}
    for node in scheme:
        for svc in scheme[node]:
            if svc not in svc_inst_count:
                svc_inst_count[svc] = 0
            svc_inst_count[svc] += scheme[node][svc]

    result = 0.0
    # calculating average response time for each user desired function
    for func in func_freq:
        prob = func_freq[func] * 1.0 / total_freq
        call_pairs = [(-1, func, 1)]
        index = 0
        total_t = 0
        while index < len(call_pairs):
            t = 0
            pred_func = call_pairs[index][0]
            pred_svc = func_objs[pred_func]['svcIndex']
            succ_func = call_pairs[index][1]
            succ_func_obj = func_objs[succ_func]
            succ_svc = func_objs[succ_func]['svcIndex']
            for pred_node in range(0, len(node_objs)):
                if pred_func == -1:
                    if succ_func in users[pred_node]:
                        t_prob = users[pred_node][succ_func] * 1.0 / func_freq[succ_func]
                    else:
                        continue
                else:
                    node_inst = scheme[pred_node][pred_svc]
                    t_prob = node_inst * 1.0 / svc_inst_count[pred_svc]
                for succ_node in range(0, len(node_objs)):
                    node_inst = scheme[succ_node][succ_svc]
                    curr_prob = t_prob * node_inst * 1.0 / svc_inst_count[succ_svc]
                    # calculating response time
                    if pred_node != succ_node:
                        t_delay = connections[pred_node][succ_node]['delay']
                        t_transfer = (succ_func_obj['input'] + succ_func_obj['output']) / connections[pred_node][succ_node]['bandwidth']
                        t += curr_prob * (t_delay + t_transfer)
            # call coefficient should be included
            t *= call_pairs[index][2]
            # process next call along the chains
            if succ_func in chains:
                for next_func, coe in chains[succ_func].items():
                    call_pairs.append((succ_func, next_func, coe * call_pairs[index][2]))
            index += 1
            total_t += t
        result += prob * total_t
    return result
