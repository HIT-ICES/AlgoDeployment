#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：AlgoGroupDeployment 
@File ：AxAlgo.py
@Author ：septemberhx
@Date ：2021/3/28
@Description: Approximation algorithm based on greedy strategy

Please remember:
    node_objs holds details of each node
    node is the id
"""
import datetime
import math
from typing import Dict, List

from QSRFP.Builder import MathBuilder
from commom import scheme
from commom.SvcCallGraph import SvcCallGraph
from commom.logger import get_logger


class AxAlgo:
    min_svc_count: Dict[int, int]
    deployment: Dict[int, Dict[int, int]]  # Dict[node, Dict[service, instance count]]
    svc_call_graph: SvcCallGraph

    def __init__(self, svc_objs: List[Dict], node_objs: List[Dict], func_objs: List[Dict], users: Dict,
                 chains: Dict[int, Dict], chain_list: List[List[int]], connections, price: Dict[str, float]):
        self.svc_call_graph = SvcCallGraph()
        self.min_svc_count = {}
        self.remain_res = {}
        self.svc_inst_count = {}
        self.logger = get_logger('AxAlgo')

        # prepare necessary data
        self.svc_objs = svc_objs
        self.node_objs = node_objs
        self.func_objs = func_objs
        self.chains = chains
        self.chain_list = chain_list
        self.users = users
        self.connections = connections
        self.prices = price
        for node_obj in self.node_objs:
            self.remain_res[node_obj['index']] = {
                'cpu': node_obj['res']['cpu'],
                'ram': node_obj['res']['ram']
            }
        self.svc_prices = {}
        for svc in range(0, len(self.svc_objs)):
            self.svc_prices[svc] = self.prices['cpu'] * self.svc_objs[svc]['res']['cpu'] + self.prices['ram'] * \
                                   self.svc_objs[svc]['res']['ram']

        self.svc_req_on_node = {}
        self.svc_reqs = {}
        # service request frequency on each node
        for node in self.users:
            self.svc_req_on_node[node] = {}
            for func, freq in self.users[node].items():
                svc = self.func_objs[func]['svcIndex']
                if svc not in self.svc_req_on_node[node]:
                    self.svc_req_on_node[node][svc] = 0
                self.svc_req_on_node[node][svc] += freq
        # service request frequency in total
        for node in self.svc_req_on_node:
            for svc, freq in self.svc_req_on_node[node].items():
                if svc not in self.svc_reqs:
                    self.svc_reqs[svc] = 0
                self.svc_reqs[svc] += freq

        # init
        self.deployment = {}
        for node in range(0, len(self.node_objs)):
            self.deployment[node] = {}
            for svc in range(0, len(self.svc_objs)):
                self.deployment[node][svc] = 0
        for svc in range(0, len(self.svc_objs)):
            self.svc_inst_count[svc] = 0

    def solve(self, strategy=1, push_needed=True) -> (float, float):
        self.svc_call_graph.create_graph(svc_objs=self.svc_objs, func_objs=self.func_objs, chains=self.chains,
                                         chain_list=self.chain_list, users=self.users)
        self.min_svc_count = scheme.cal_min_svc_count(self.svc_call_graph.graph_bak, self.svc_objs)
        try:
            t1 = datetime.datetime.now().timestamp()
            total_price = 0.0
            if not scheme.check_problem_is_feasible(self.svc_objs, self.node_objs, self.min_svc_count,
                                                    self.svc_prices, self.prices['max']):
                raise Exception('The problem is infeasible due to insufficient resources')

            first_func_freq = {}
            for node in self.users:
                for func, freq in self.users[node].items():
                    if func not in first_func_freq:
                        first_func_freq[func] = 0
                    first_func_freq[func] += freq

            # strategy 1: DFS
            if strategy == 1:
                solved_svc_freq = {}
                results = []
                if strategy == 1:
                    results = self.svc_call_graph.calc_paths(self.min_svc_count)
                elif strategy == 3:
                    results = self.svc_call_graph.calc_paths_by_unit_datasize(self.min_svc_count)
                # 这里的 svcs 顺序很重要，如果只按照数据量进行排序，那么容易出现一些短链服务被忽略
                for svcs, _, chain in results:
                    call_coe = 1.0
                    for i in range(0, len(svcs)):
                        curr_svc = svcs[i]
                        if curr_svc not in solved_svc_freq:
                            solved_svc_freq[curr_svc] = 0.0
                        if i == 0:
                            prev_svc = -1
                        else:
                            prev_svc = self.func_objs[chain[i - 1]]['svcIndex']
                            call_coe *= self.chains[chain[i - 1]][chain[i]]
                        # needs_solve_freq = self.svc_call_graph.graph_bak[prev_svc][curr_svc]['attr'][chain[i]]
                        needs_solve_freq = call_coe * first_func_freq[chain[0]]
                        existed_ability = self.svc_inst_count[curr_svc] * self.svc_objs[curr_svc]['ability']
                        if existed_ability < solved_svc_freq[curr_svc] + needs_solve_freq:
                            need_to_deploy_count = int(math.ceil(
                                (solved_svc_freq[curr_svc] + needs_solve_freq - existed_ability) /
                                self.svc_objs[curr_svc][
                                    'ability']))
                            tmp_price = total_price + self.svc_prices[curr_svc] * need_to_deploy_count
                            if tmp_price > self.prices['max']:
                                raise Exception('Price too low!!')
                            self.deploy_svc_with_best_node_considering_spread(curr_svc, need_to_deploy_count)
                            solved_svc_freq[curr_svc] += needs_solve_freq
                            total_price = tmp_price
                        else:
                            solved_svc_freq[curr_svc] += needs_solve_freq
                    self.svc_call_graph.remove_path(svcs)
            elif strategy == 2:
                # strategy 2: BFS
                while not self.svc_call_graph.is_solved():
                    svc_candidate = self.svc_call_graph.find_next_svcs()
                    best_svc = None
                    best_value = 0.0
                    for svc in svc_candidate:
                        if best_svc is None or self.svc_objs[svc]['ability'] / self.svc_objs[svc]['res'][
                            'cpu'] > best_value:
                            best_svc = svc
                            best_value = self.svc_objs[svc]['ability'] / self.svc_objs[svc]['res']['cpu']
                    if best_svc is not None:
                        while self.svc_inst_count[best_svc] < self.min_svc_count[best_svc]:
                            # deploy minimum count of instances
                            desired_count = self.min_svc_count[best_svc] - self.svc_inst_count[best_svc]
                            tmp_price = total_price + self.svc_prices[best_svc] * desired_count
                            if tmp_price > self.prices['max']:
                                raise Exception('Price too low!!')
                            self.deploy_svc_with_best_node_considering_spread(best_svc, desired_count)
                            total_price = tmp_price
                        self.svc_call_graph.remove_path([best_svc])

            if not self.svc_call_graph.is_solved():
                raise Exception('Failed to find a solution')
            elif push_needed:
                # try to improve the solution
                self.push_bound(total_price)

            if not scheme.check_scheme_is_feasible(self.deployment, self.svc_objs, self.node_objs, self.min_svc_count,
                                                   self.svc_prices, self.prices['max']):
                raise Exception('Failed to verify solution!!!')

            f = scheme.evaluate(self.deployment, self.node_objs, self.func_objs, self.users, self.chains,
                                self.connections)
            used_time = datetime.datetime.now().timestamp() - t1
            self.logger.info(f'Result of AxAlgo is {f} with time = {used_time}')
            self.logger.info(self.deployment)
            self.log_res_usage()

            #  ==> Use this to check whether the QSRFP equals to FPP
            # self.test_path_math_builder(f)
            #  ==> try a small size data

            return f, used_time

        except Exception as e:
            self.logger.error(e)
            self.logger.info('Failed to find a solution!')
        return -1, -1

    def deploy_svc_with_best_node_considering_spread(self, curr_svc, count):
        while count > 0:
            target_node, _ = self.find_best_node_strong(curr_svc)
            if target_node is None:
                print(self.remain_res)
                raise Exception('No best node!!')
            max_count = min(self.node_res_for_svc(target_node, curr_svc), count)
            self.deploy_svc_on_node(curr_svc, target_node, max_count)
            count -= max_count

            # 波动思想，部署了一个服务，对所有相连的服务均照成波动并传播至边界，多波动几次
            affected_svcs = []
            for pred_svc in self.svc_call_graph.graph_bak.predecessors(curr_svc):
                if pred_svc >= 0:
                    affected_svcs.append(pred_svc)
            for succ_svc in self.svc_call_graph.graph_bak.successors(curr_svc):
                affected_svcs.append(succ_svc)

            affected_index = 0
            processed_svcs = {curr_svc}
            while affected_index < len(affected_svcs):
                affected_svc = affected_svcs[affected_index]

                if affected_svc >= 0 and self.svc_inst_count[affected_svc] > 0:
                    # record existing instances
                    deployed_before = {}
                    inst_count_before = self.svc_inst_count[affected_svc]
                    # release all the affected service instances
                    for node in range(0, len(self.node_objs)):
                        if self.deployment[node][affected_svc] > 0:
                            deployed_before[node] = self.deployment[node][affected_svc]
                            self.consume_resource(node, self.svc_objs[affected_svc]['res'], True,
                                                  self.deployment[node][affected_svc])
                            self.deployment[node][affected_svc] = 0
                    self.svc_inst_count[affected_svc] = 0
                    # re-deploy it. total price is not changed.
                    deployed_after = {}
                    while self.svc_inst_count[affected_svc] < inst_count_before:
                        target_node, _ = self.find_best_node_strong(affected_svc)
                        if target_node is None:
                            raise Exception('No best node!!')
                        tmp_count = min(self.node_res_for_svc(target_node, affected_svc),
                                        inst_count_before - self.svc_inst_count[affected_svc])
                        self.deploy_svc_on_node(affected_svc, target_node, tmp_count)
                        if target_node not in deployed_after:
                            deployed_after[target_node] = 0
                        deployed_after[target_node] += tmp_count
                    # compare deployed_before and deployed_after to check
                    #   whether pred_svcs of affected_svc are needed to re-deployed
                    redeploy_needed = False
                    if len(deployed_after) != len(deployed_before):
                        redeploy_needed = True
                    else:
                        for before_node, before_value in deployed_before.items():
                            if before_node not in deployed_after or deployed_after[before_node] != before_value:
                                redeploy_needed = True
                                break
                    if redeploy_needed:
                        # print(f'Service {affected_svc} is improved due to spread')
                        for pred_pred in self.svc_call_graph.graph_bak.predecessors(affected_svc):
                            if pred_pred >= 0 and pred_pred not in processed_svcs:
                                affected_svcs.append(pred_pred)
                        for succ_succ in self.svc_call_graph.graph_bak.successors(affected_svc):
                            if succ_succ not in processed_svcs:
                                affected_svcs.append(succ_succ)
                    processed_svcs.add(affected_svc)
                affected_index += 1

    def node_res_for_svc(self, node, svc) -> int:
        """
        Check the remain resource of node can deploy how many new instances for service svc
        """
        count_cpu = math.floor(self.remain_res[node]['cpu'] / self.svc_objs[svc]['res']['cpu'])
        count_ram = math.floor(self.remain_res[node]['ram'] / self.svc_objs[svc]['res']['ram'])
        return min(int(count_cpu), int(count_ram))

    def deploy_svc_on_node(self, svc, node, count):
        # create an instance on target_node
        self.deployment[node][svc] += count
        # print(f'Deploy {svc} on node {node}')
        # consume necessary resources on target_node
        self.consume_resource(node, self.svc_objs[svc]['res'], count=count)
        self.svc_inst_count[svc] += count
        if self.svc_inst_count[svc] > self.min_svc_count[svc]:
            raise Exception('Something wrong')

    def find_best_node_weak(self, svc) -> int or None:
        """
        Find the best node for given service to create an instance weakly, i.e, without considering successors
          or predecessors when not deployed
        :param svc:
        :return:
        """
        best_node = None
        best_value = float('inf')
        # loop all nodes to find the best node
        for node in range(0, len(self.node_objs)):
            if not self.res_enough(node, self.svc_objs[svc]['res']):
                continue
            value = 0
            # the algorithm cannot promise that all the predecessor/successors service are deployed before svc
            #    since one service can have multiple predecessors/successors
            # but value should be 0 if so
            for pred_svc in self.svc_call_graph.graph_bak.predecessors(svc):
                value += self.calc_contri_between_svcs_for_svc_on_node(pred_svc, svc, node, svc_count_diff=1)

            if value < best_value:
                best_value = value
                best_node = node
        return best_node

    def find_best_node_strong(self, svc) -> (int, float) or None:
        best_node = None
        best_value = float('inf')
        for node in range(0, len(self.node_objs)):
            if not self.res_enough(node, self.svc_objs[svc]['res']):
                continue
            value = 0

            has_pred_or_succ_deployed = False
            for pred_svc in self.svc_call_graph.graph_bak.predecessors(svc):
                if pred_svc < 0 or self.svc_inst_count[pred_svc] > 0:
                    has_pred_or_succ_deployed = True
                    value += self.calc_contri_between_svcs_with_diff(pred_svc, svc, diff_svc_node=node,
                                                                     diff_svc_count=1)
            for succ_svc in self.svc_call_graph.graph_bak.successors(svc):
                if self.svc_inst_count[succ_svc] > 0:
                    has_pred_or_succ_deployed = True
                    value += self.calc_contri_between_svcs_with_diff(svc, succ_svc, diff_prev_node=node,
                                                                     diff_prev_count=1)

            if has_pred_or_succ_deployed and value < best_value:
                best_value = value
                best_node = node
        return best_node, best_value

    def calc_contribution(self, svc) -> float:
        """
        Calculate the contribution of given services, i.e, the average response time
        :param svc: target services
        :return: contributed average response time
        """
        if svc not in self.svc_call_graph.graph_bak.nodes:
            # which means it is not used by users
            return 0

        value = 0
        for pred_svc in self.svc_call_graph.graph_bak.predecessors(svc):
            value += self.calc_contri_between_svcs(pred_svc, svc)
        for succ_svc in self.svc_call_graph.graph_bak.successors(svc):
            value += self.calc_contri_between_svcs(svc, succ_svc)
        return value

    def calc_contri_between_svcs(self, pred_svc, svc):
        value = 0
        for svc_node in range(0, len(self.node_objs)):
            value += self.calc_contri_between_svcs_for_svc_on_node(pred_svc, svc, svc_node)
        return value

    def calc_contri_between_svcs_with_diff(self, pred_svc, svc, diff_prev_node=None, diff_prev_count=0,
                                           diff_svc_node=None, diff_svc_count=0):
        if diff_prev_node is not None:
            self.deployment[diff_prev_node][pred_svc] += diff_prev_count
            self.svc_inst_count[pred_svc] += diff_prev_count
        if diff_svc_node is not None:
            self.deployment[diff_svc_node][svc] += diff_svc_count
            self.svc_inst_count[svc] += diff_svc_count
        result = self.calc_contri_between_svcs(pred_svc, svc)
        if diff_prev_node is not None:
            self.deployment[diff_prev_node][pred_svc] -= diff_prev_count
            self.svc_inst_count[pred_svc] -= diff_prev_count
        if diff_svc_node is not None:
            self.deployment[diff_svc_node][svc] -= diff_svc_count
            self.svc_inst_count[svc] -= diff_svc_count
        return result

    def calc_contri_between_svcs_for_svc_on_node(self, pred_svc, svc, svc_node, svc_count_diff=0) -> float:
        """
        Calculate average time contribution between service #pred_svc and #svc
        It allows templating add #svc_count_diff instances on #svc_node
        :param pred_svc: predecessor service
        :param svc: service wanted to calculate its contribution
        :param svc_node: instances on which node you want ot calculate their contribution
        :param svc_count_diff: temp add instances on svc_node for svc while not affect global scheme
        :return: the contribution
        """
        value = 0
        if self.deployment[svc_node][svc] == 0 and svc_count_diff == 0:
            return value
        svc_prob = (svc_count_diff + self.deployment[svc_node][svc]) * 1.0 / (svc_count_diff + self.svc_inst_count[svc])

        total_data_size = 0.0
        total_freq = 0.0
        for func, func_freq in self.svc_call_graph.graph_bak[pred_svc][svc]['attr'].items():
            total_freq += func_freq
            total_data_size += (self.func_objs[func]['input'] + self.func_objs[func]['output']) * func_freq

        for pred_node in range(0, len(self.node_objs)):
            if pred_node == svc_node:
                continue

            if pred_svc >= 0:
                # predecessor is a service
                if self.deployment[pred_node][pred_svc] == 0:
                    continue
                prob = (self.deployment[pred_node][pred_svc]) * 1.0 / (self.svc_inst_count[pred_svc])

            else:
                # predecessor is the user
                if pred_node not in self.svc_req_on_node:
                    continue
                if svc not in self.svc_req_on_node[pred_node]:
                    continue
                prob = self.svc_req_on_node[pred_node][svc] * 1.0 / self.svc_reqs[svc]
            prob *= svc_prob
            delay = self.connections[pred_node][svc_node]['delay'] * total_freq
            transform_time = total_data_size / self.connections[pred_node][svc_node]['bandwidth']
            value += prob * (delay + transform_time)

        # # calculate response time between pred_svc and svc without considering successor of svc for each function
        # for func, func_freq in self.svc_call_graph.graph_bak[pred_svc][svc]['attr'].items():
        #     tmp_value = 0
        #     for pred_node in range(0, len(self.node_objs)):
        #         if pred_node == svc_node:
        #             continue
        #
        #         if pred_svc >= 0:
        #             # predecessor is a service
        #             if self.deployment[pred_node][pred_svc] == 0:
        #                 continue
        #             prob = (self.deployment[pred_node][pred_svc]) * 1.0 / (self.svc_inst_count[pred_svc])
        #
        #         else:
        #             # predecessor is the user
        #             if pred_node not in self.svc_req_on_node:
        #                 continue
        #             if svc not in self.svc_req_on_node[pred_node]:
        #                 continue
        #             prob = self.svc_req_on_node[pred_node][svc] * 1.0 / self.svc_reqs[svc]
        #         prob *= svc_prob
        #         delay = self.connections[pred_node][svc_node]['delay']
        #         transform_time = (self.func_objs[func]['input'] + self.func_objs[func]['output']) / \
        #                          self.connections[pred_node][svc_node]['bandwidth']
        #         tmp_value += prob * (delay + transform_time)
        #     # current value is the average time for one request. Need to multiply it with call frequency
        #     value += tmp_value * func_freq
        return value

    def res_enough(self, node, res) -> bool:
        if node not in self.remain_res:
            raise Exception('Resource not init on node {0}'.format(node))
        else:
            return self.remain_res[node]['cpu'] >= res['cpu'] and self.remain_res[node]['ram'] >= res['ram']

    def consume_resource(self, node, res, reverse=False, count=1) -> None:
        if node not in self.remain_res:
            raise Exception('Resource not init on node {0}'.format(node))
        else:
            coe = 1.0 if not reverse else -1.0
            self.remain_res[node]['cpu'] -= coe * res['cpu'] * count
            self.remain_res[node]['ram'] -= coe * res['ram'] * count

    def push_bound(self, curr_cost) -> None:
        """
        improve the solution by deploying more instances than minimum count
        :return: None
        """
        while True:
            best_svc = None
            best_node = None
            best_diff_value = 0
            for svc in range(0, len(self.svc_objs)):
                if self.svc_prices[svc] > self.prices['max'] - curr_cost:
                    continue

                raw_value = self.calc_contribution(svc)
                if raw_value == 0:
                    # svc not used by users
                    continue
                node, value = self.find_best_node_strong(svc)
                if raw_value - value > 10e-6 and raw_value - value > best_diff_value:
                    best_diff_value = raw_value - value
                    best_node = node
                    best_svc = svc

            if best_svc is not None:
                self.deployment[best_node][best_svc] += 1
                self.svc_inst_count[best_svc] += 1
                self.consume_resource(best_node, self.svc_objs[best_svc]['res'])
                curr_cost += self.svc_prices[best_svc]
            else:
                break

    def test_path_math_builder(self, curr_result):
        """
        code to compare the result calculated by the sum of avg time of each Response path with their probability
        """
        t1 = datetime.datetime.now()
        builder = MathBuilder(self.node_objs, self.svc_objs, self.func_objs, self.users, self.chains, self.chain_list,
                              self.connections, self.prices, self.prices['max'])
        path_model = builder.build_by_path()
        v = []
        for node in range(0, len(self.node_objs)):
            for svc in range(0, len(self.svc_objs)):
                v.append(self.deployment[node][svc])
        t2 = datetime.datetime.now()
        print(f'=====> test math time is {t2 - t1} ')
        vv = path_model.evalf(subs=builder.to_sym_value_Dict(v))
        print(f'The result calculated by path is {vv}, which should equal to {curr_result}')

    def log_res_usage(self):
        total_cpu = 0.0
        total_ram = 0.0
        total_left_cpu = 0.0
        total_left_ram = 0.0
        for node_obj in self.node_objs:
            total_cpu += node_obj['res']['cpu']
            total_ram += node_obj['res']['ram']

        for node in self.remain_res:
            total_left_cpu += self.remain_res[node]['cpu']
            total_left_ram += self.remain_res[node]['ram']
        self.logger.info(self.remain_res)
        self.logger.info(f'{total_cpu}, {total_ram}, {total_left_cpu}, {total_left_ram}')
        self.logger.info(
            f'used cpu {(total_cpu - total_left_cpu) * 1.0 / total_cpu}, used ram {(total_ram - total_left_ram) * 1.0 / total_ram}')
