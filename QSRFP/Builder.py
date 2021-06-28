#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：AlgoGroupDeployment 
@File ：Builder.py
@Author ：septemberhx
@Date ：2021/4/4
@Description:
"""
import itertools

import sympy
from sympy import fraction
from typing import Dict, List

from QSRFP.QSRFP import QSRFP
from commom import scheme
from commom.SvcCallGraph import SvcCallGraph


class MathBuilder:

    syms: List[sympy.core.symbol.Symbol]

    def __init__(self, node_objs: List[Dict], svc_objs: List[Dict], func_objs: List[Dict],
                 users: Dict, chains: Dict[int, Dict], chain_List, connections, price: Dict[str, float], max_price):
        self.syms = []
        self.node_size = len(node_objs)
        self.svc_size = len(svc_objs)
        self.svc_call_graph = SvcCallGraph()
        self.min_svc_freq = {}

        # cache
        self.node_objs = node_objs
        self.svc_objs = svc_objs
        self.func_objs = func_objs
        self.users = users
        self.chains = chains
        self.chain_List = chain_List
        self.connections = connections
        self.price = price
        self.max_price = max_price

        # create symbols
        for node in range(0, self.node_size):
            for svc in range(0, self.svc_size):
                self.syms.append(sympy.symbols('x_{0}_{1}'.format(node, svc)))
        self.svc_call_graph.create_graph(svc_objs=self.svc_objs, func_objs=self.func_objs, chains=self.chains, chain_list=self.chain_List, users=self.users)

    def to_QSRFP(self):
        self.cal_min_svc_count()
        min_inst_count = scheme.cal_min_svc_count(self.svc_call_graph.graph_bak, self.svc_objs)

        constraints = [self.build_price_constraint()]
        constraints.extend(self.build_res_constraints())
        constraints.extend(self.build_ability_constraints())
        function_set = set()
        for node in self.users:
            for func in self.users[node]:
                function_set.add(func)
        f_s = []
        g_s = []
        for func in function_set:
            for r in self.build_target_function_for_func(func):
                f_s.append(r[0].expand())
                g_s.append(r[1].expand())
        # prepare Y
        Y = []
        for node in range(0, len(self.node_objs)):
            for svc in range(0, len(self.svc_objs)):
                if svc not in min_inst_count:
                    Y.append([0, 0])
                else:
                    Y.append([0, min_inst_count[svc]])

        return QSRFP(f_s, g_s, constraints, Y, self.syms)

    def cal_min_svc_count(self) -> None:
        """
        calculate minimum service instance count of each service
        :return: None
        """
        for svc in self.svc_call_graph.graph_bak.nodes:
            if svc < 0:
                continue
            total_freq = 0
            for pred_svc in self.svc_call_graph.graph_bak.predecessors(svc):
                for func, freq in self.svc_call_graph.graph_bak[pred_svc][svc]['attr'].items():
                    total_freq += freq
            self.min_svc_freq[svc] = total_freq

    def sym(self, node, svc):
        return self.syms[node * self.svc_size + svc]

    def svc_inst_count(self, svc):
        r = sympy.simplify(0)
        for node in range(0, self.node_size):
            r += self.sym(node, svc)
        return r

    def build_price_constraint(self):
        total = sympy.simplify(0)
        for svc in range(0, self.svc_size):
            p = self.price['cpu'] * self.svc_objs[svc]['res']['cpu'] + self.price['ram'] * self.svc_objs[svc]['res']['ram']
            total += p * self.svc_inst_count(svc)
        total -= self.max_price  # make sure it <= 0
        return total.expand()

    def build_res_constraints(self):
        results = []
        for node in range(0, self.node_size):
            total_res = [0, 0]  # cpu, ram
            for svc in range(0, self.svc_size):
                total_res[0] += self.sym(node, svc) * self.svc_objs[svc]['res']['cpu']
                total_res[1] += self.sym(node, svc) * self.svc_objs[svc]['res']['ram']
            # make sure it <= 0
            results.append((total_res[0] - self.node_objs[node]['res']['cpu']).expand())
            results.append((total_res[1] - self.node_objs[node]['res']['ram']).expand())
        return results

    def build_ability_constraints(self):
        results = []
        for svc, minimum_freq in self.min_svc_freq.items():
            ability = self.svc_objs[svc]['ability'] * self.svc_inst_count(svc)
            # make sure it <= 0
            results.append((minimum_freq - ability).expand())
        return results

    def build_target_function_for_func(self, func):
        # prepare data for calculating probability
        total_freq = 0
        func_freq = {}
        for node in self.users:
            for func_t, freq_t in self.users[node].items():
                total_freq += freq_t
                if func_t not in func_freq:
                    func_freq[func_t] = 0
                func_freq[func_t] += freq_t

        svc_inst_count = {}
        for svc in range(0, self.svc_size):
            svc_inst_count[svc] = self.svc_inst_count(svc)

        results = []
        # calculating average response time for the user desired function
        prob = func_freq[func] * 1.0 / total_freq
        call_pairs = [(-1, func, 1)]
        index = 0
        while index < len(call_pairs):
            t = sympy.simplify(0)
            pred_func = call_pairs[index][0]
            pred_svc = self.func_objs[pred_func]['svcIndex']
            succ_func = call_pairs[index][1]
            succ_func_obj = self.func_objs[succ_func]
            succ_svc = self.func_objs[succ_func]['svcIndex']
            for pred_node in range(0, self.node_size):
                if pred_func == -1:
                    if func not in self.users[pred_node]:
                        continue
                    t_prob = self.users[pred_node][succ_func] * 1.0 / func_freq[succ_func]
                else:
                    node_inst = self.sym(pred_node, pred_svc)
                    t_prob = node_inst * 1.0 / svc_inst_count[pred_svc]
                for succ_node in range(0, self.node_size):
                    node_inst = self.sym(succ_node, succ_svc)
                    curr_prob = t_prob * node_inst * 1.0 / svc_inst_count[succ_svc]
                    # calculating response time
                    if pred_node != succ_node:
                        t_delay = self.connections[pred_node][succ_node]['delay']
                        t_transfer = (succ_func_obj['input'] + succ_func_obj['output']) / self.connections[pred_node][succ_node]['bandwidth']
                        t += curr_prob * (t_delay + t_transfer)
            # call coefficient should be included
            t *= call_pairs[index][2]
            # process next call along the chains
            if succ_func in self.chains:
                for next_func, coe in self.chains[succ_func].items():
                    call_pairs.append((succ_func, next_func, coe * call_pairs[index][2]))
            index += 1
            # make sure that the function is quadratic sum-of-ratios fractional programs
            # in each round while loop, the denominator is the same in 2-depth for loop
            #   but different with other while loops, thus we need to collect the equation at the end of the while loop
            # extend should be called otherwise sympy does not remove brackets
            results.append(fraction(sympy.simplify(prob * t)))
        return results

    def symbol_index(self, sym):
        return self.syms.index(sym)

    def to_sym_value_Dict(self, values: List[float]) -> Dict[sympy.core.symbol.Symbol, float]:
        sym_value_Dict = {}
        for i in range(0, len(self.syms)):
            sym_value_Dict[self.syms[i]] = values[i]
        return sym_value_Dict

    # below are codes that are used to calculate the target function with the path way:
    #   it will loop all the possible response path [#node x #chain_len] with their probability
    def build_by_path(self):
        result = 0.0
        for chain in self.chain_List:
            result += self.build_target_function_for_func_by_path(chain[0], chain)
        return result

    def build_target_function_for_func_by_path(self, func, chain):
        # prepare data for calculating probability
        total_freq = 0
        func_freq = {}
        for node in self.users:
            for func_t, freq_t in self.users[node].items():
                total_freq += freq_t
                if func_t not in func_freq:
                    func_freq[func_t] = 0
                func_freq[func_t] += freq_t

        svc_inst_count = {}
        for svc in range(0, self.svc_size):
            svc_inst_count[svc] = self.svc_inst_count(svc)

        node_indexs = [x for x in range(0, self.node_size)]
        total_t = 0.0
        for path in self.all_paths(len(chain) + 1):
            if path[0] in self.users and func in self.users[path[0]]:
                prob = self.users[path[0]][func] * 1.0 / func_freq[func]
            else:
                continue
            t = 0.0
            for i in range(0, len(path)):
                if i == 0:
                    continue
                else:
                    prob *= self.sym(path[i], self.func_objs[chain[i - 1]]['svcIndex']) / svc_inst_count[self.func_objs[chain[i - 1]]['svcIndex']]
                    if path[i - 1] != path[i]:
                        t_delay = self.connections[path[i - 1]][path[i]]['delay']
                        t_transfer = (self.func_objs[chain[i - 1]]['input'] + self.func_objs[chain[i - 1]]['output']) / self.connections[path[i - 1]][path[i]]['bandwidth']
                        t += t_delay + t_transfer
            total_t += t * prob
        return total_t * (func_freq[func] * 1.0 / total_freq)

    def all_paths(self, size):
        result = []
        self.all_paths_(size, [], result)
        return result

    def all_paths_(self, size, curr, result):
        if len(curr) >= size:
            result.append(curr)
        else:
            for node in range(0, len(self.node_objs)):
                tmp_curr = list(curr)
                tmp_curr.append(node)
                self.all_paths_(size, tmp_curr, result)