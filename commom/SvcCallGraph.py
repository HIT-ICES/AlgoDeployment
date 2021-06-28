#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：AlgoGroupDeployment
@File    ：SvcCallGraph.py
@Author  ：septemberhx
@Date    ：2021/3/27
@Description: Build service all graph and support other operations
"""
from typing import Dict, List, Tuple, Set

import networkx as nx
import matplotlib.pyplot as plt


class SvcCallGraph:

    # d_size_total: Dict[int, float]  # Dict[service, float], total data transfer size of each service
    # d_sizes: Dict[int, Dict[int, float]]  # Dict[service, Dict[service, float]], data transfer size between services
    graph_bak: nx.DiGraph
    user_demands: Set[int]
    critical_paths: Dict[int, Tuple[List[int], float]]  # Dict[node, (path, value)]
    graph: nx.DiGraph
    users: Dict[int, Dict[int, float]]
    func_objs: List[Dict]
    chains: Dict[int, Dict[int, float]]

    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph_bak = nx.DiGraph()
        self.svc_objs = []
        self.func_objs = []
        self.chains = {}
        self.chain_list = []
        self.users = {}
        self.critical_paths = {}
        self.user_demands = set()

        # prepared data
        # self.d_sizes = {}
        # self.d_size_total = {}
        self.freq_for_svc = {}

    def create_graph(self, svc_objs: List[Dict], func_objs: List[Dict], chains: Dict[int, Dict], chain_list: List[List[int]], users: Dict) -> None:
        self.func_objs = func_objs
        self.chains = chains
        self.chain_list = chain_list
        self.users = users
        self.svc_objs = svc_objs

        # 1. get total requests of each function on all server nodes
        total_demands = {}
        for node in users:
            for func in users[node]:
                # cache it for other usage
                self.user_demands.add(func)
                if func not in total_demands:
                    total_demands[func] = 0
                total_demands[func] += users[node][func]

        # 2. for each function request, extend the graph
        #   service as node, call relation as edge, call frequency as edge weight
        #   treat user as node -1
        self.graph.add_node(-1)

        for chain in self.chain_list:
            call_coe = 1.0
            for i in range(0, len(chain)):
                curr_func = chain[i]
                curr_svc = self.func_objs[curr_func]['svcIndex']
                if i == 0:
                    pred_svc = -1
                else:
                    pred_svc = self.func_objs[chain[i - 1]]['svcIndex']
                    call_coe *= self.chains[chain[i - 1]][curr_func]
                if curr_svc not in self.graph or curr_svc not in self.graph[pred_svc]:
                    self.graph.add_edge(pred_svc, curr_svc, attr={curr_func: total_demands[chain[0]] * call_coe})
                else:
                    if curr_func not in self.graph[pred_svc][curr_svc]['attr']:
                        self.graph[pred_svc][curr_svc]['attr'][curr_func] = 0.0
                    self.graph[pred_svc][curr_svc]['attr'][curr_func] += total_demands[chain[0]] * call_coe

        # todo: 根据 chain_list 来计算，而不是依赖 chains 23->18->3 与 24->18->3->4->20 这两种情况可以同时成立（3中做判断），那么就不能依赖这个
        # for func in total_demands:
        #     # edge from user to their demand service
        #     if func_objs[func]['svcIndex'] not in self.graph or func_objs[func]['svcIndex'] not in self.graph[-1]:
        #         self.graph.add_edge(-1, func_objs[func]['svcIndex'], attr={func: total_demands[func]})
        #     else:
        #         self.graph[-1][func_objs[func]['svcIndex']]['attr'][func] = total_demands[func]
        #
        #     # update edge weight for the edges behind func with given call coefficient in chains
        #     changed_value_Dict = {func: total_demands[func]}
        #     caller_funcs = [func]
        #     curr_index = 0
        #     while curr_index < len(caller_funcs):
        #         caller_func = caller_funcs[curr_index]
        #         caller_svc = func_objs[caller_func]['svcIndex']
        #
        #         if caller_func in chains:
        #             called_func_Dict = chains[caller_func]
        #             # one function may call many functions in parallel
        #             for called_func, call_coe in called_func_Dict.items():
        #                 added_v = call_coe * changed_value_Dict[caller_func]
        #                 called_svc = func_objs[called_func]['svcIndex']
        #                 if called_svc not in self.graph[caller_svc]:
        #                     # add new edge if not present before
        #                     self.graph.add_edge(caller_svc, called_svc, attr={called_func: added_v})
        #                 else:
        #                     if called_func not in self.graph[caller_svc][called_svc]['attr']:
        #                         # add new attr in the edge if not present before
        #                         self.graph[caller_svc][called_svc]['attr'][called_func] = added_v
        #                     else:
        #                         # update value
        #                         self.graph[caller_svc][called_svc]['attr'][called_func] += added_v
        #                 changed_value_Dict[called_func] = added_v
        #                 caller_funcs.append(called_func)
        #         curr_index += 1

        # 3. prepare useful data for others
        # for dep in self.graph.edges:
        #     data_size = 0
        #     for func, freq in self.graph[dep[0]][dep[1]]['attr'].items():
        #         data_size += (self.func_objs[func]['input'] + self.func_objs[func]['output']) * freq
        #     if dep[0] not in self.d_sizes:
        #         self.d_sizes[dep[0]] = {}
        #     self.d_sizes[dep[0]][dep[1]] = data_size
        #
        # for svc in self.d_sizes:
        #     data_size = 0
        #     for succ_svc in self.d_sizes[svc]:
        #         data_size += self.d_sizes[svc][succ_svc]
        #     self.d_size_total[svc] = data_size

        for svc in range(0, len(self.svc_objs)):
            total_freq = 0.0
            if svc not in self.graph.nodes:
                self.freq_for_svc[svc] = 0
            else:
                for pred in self.graph.predecessors(svc):
                    for _, freq in self.graph[pred][svc]['attr'].items():
                        total_freq += freq
                self.freq_for_svc[svc] = total_freq

        # 4. create a copy graph
        self.graph_bak = self.graph.copy()
        self.draw()

    def find_next_svcs(self) -> List[int]:
        result = []
        for demand in self.user_demands:
            svc = self.func_objs[demand]['svcIndex']
            if svc not in self.graph.nodes:
                continue

            # considering the services that are called by users only first
            if len(self.graph.pred[svc]) != 1:
                continue
            result.append(svc)

        # all the user-connected nodes are removed
        for svc in self.graph.nodes:
            # if there is no predecessor for this node and it is not the user node (-1)
            if len(self.graph.pred[svc]) == 0 and svc >= 0:
                result.append(svc)

        if len(result) == 0:
            for svc in self.graph.succ[-1]:
                result.append(svc)

        if len(result) == 0:
            # sometime there contains circle on service level but not in API level
            # just return all services
            for svc in self.graph.nodes:
                if svc < 0:
                    continue
                result.append(svc)

        return result

    # def find_next_svcs_2(self, min_svc_count) -> int:
    #     svcs = self.find_next_svcs()
    #     best_v = 0.0
    #     best_svc = None
    #     for svc in svcs:
    #         total = 0.0
    #         for pred in self.graph_bak.predecessors(svc):
    #             total += self.d_sizes[pred][svc]
    #         total = total / (min_svc_count[svc] * self.svc_objs[svc]['res']['cpu'])
    #         if total > best_v:
    #             best_v = total
    #             best_svc = svc
    #     return best_svc

    def get_in_data_size(self, start_n) -> float:
        data_size = 0.0
        for successor in self.graph_bak[start_n]:
            for func, freq in self.graph_bak[start_n][successor]['attr'].items():
                data_size += (self.func_objs[func]['input'] + self.func_objs[func]['output']) * freq
        return data_size

    def calc_paths(self, min_svc_count):
        results = []
        for chain in self.chain_list:
            svcs, data_size = self.calc_path_by_chains(chain)
            data_size += self.graph_bak[-1][svcs[0]]['attr'][chain[0]] * (self.func_objs[chain[0]]['input'] + self.func_objs[chain[0]]['output'])
            results.append((svcs, data_size, chain))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def calc_paths_by_unit_datasize(self, min_svc_count):
        results = []
        for chain in self.chain_list:
            svcs, data_size = self.calc_path_by_chains(chain)
            data_size += self.graph_bak[-1][svcs[0]]['attr'][chain[0]] * (self.func_objs[chain[0]]['input'] + self.func_objs[chain[0]]['output'])
            total_res = 0.0
            for svc in svcs:
                total_res += self.svc_objs[svc]['res']['cpu'] * min_svc_count[svc]
            results.append((svcs, data_size / total_res, chain))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def calc_path_by_chains(self, chain: List[int]):
        svcs = [self.func_objs[chain[0]]['svcIndex']]
        data_size = 0.0
        for i in range(0, len(chain) - 1):
            curr_svc = self.func_objs[chain[i]]['svcIndex']
            next_svc = self.func_objs[chain[i + 1]]['svcIndex']
            freq = self.graph_bak[curr_svc][next_svc]['attr'][chain[i + 1]]
            data_size += freq * (self.func_objs[chain[i + 1]]['input'] + self.func_objs[chain[i + 1]]['output'])
            svcs.append(next_svc)
        return svcs, data_size

    def calc_func_max_critical_path(self) -> (List[int], float) or None:
        """
        Find the path with max value for each user required functions.
          The path begins with the function that users require;
          The last function of the path should have no other successors;
          Max value is the total size of the input & output data along the path
        We assume that there is no circle in the graph since circular dependency is a kind of bad smells
        :return: (critical path, value) or None
        """
        result = None
        for demand in self.user_demands:
            if self.func_objs[demand]['svcIndex'] not in self.graph.nodes:
                continue

            # considering the services that are called by users only first
            if len(self.graph.pred[self.func_objs[demand]['svcIndex']]) != 1:
                continue

            path, value = self.find_max_critical_path(self.func_objs[demand]['svcIndex'])
            if result is None or result[1] < value:
                result = (path, value)

        if result is None:
            # all the user-connected nodes are removed
            for node in self.graph.nodes:
                # if there is no predecessor for this node and it is not the user node (-1)
                if len(self.graph.pred[node]) == 0 and node >= 0:
                    path, value = self.find_max_critical_path(node)
                    if result is None or result[1] < value:
                        result = (path, value)
        return result

    def find_max_critical_path(self, start_n: int, early_stop=True) -> (List[int], float):
        """
        Find critical path for any given start node
        :return: (path, value)
        """
        # caches for speeding up
        if start_n in self.critical_paths:
            return self.critical_paths[start_n]

        max_path = []
        max_value = 0
        for successor in self.graph[start_n]:
            data_size = 0
            for func, freq in self.graph[start_n][successor]['attr'].items():
                data_size += (self.func_objs[func]['input'] + self.func_objs[func]['output']) * freq

            if early_stop:
                # 判断后继节点与该节点的连接是不是 该节点与其所有前级节点连接中 数据量最大的，如果是则继续，如果不是，则后退
                check_flag = True
                for t_pred in self.graph.predecessors(successor):
                    if t_pred == start_n:
                        continue
                    t_data_size = 0
                    for t_func, t_freq in self.graph[t_pred][successor]['attr'].items():
                        t_data_size += (self.func_objs[t_func]['input'] + self.func_objs[t_func]['output']) * t_freq
                    if t_data_size > data_size:
                        check_flag = False
                        break
                if not check_flag:
                    continue

                # 判断后继节点与该节点的连接是不是 该节点与其所有**后继**节点连接中 数据量最大的，如果是则继续，如果不是，则后退
                #   这是为了判断 successor 是和 start_n 在一起，还是和他的后继节点在一起
                for t_succ in self.graph.successors(successor):
                    t_data_size = 0
                    for t_func, t_freq in self.graph[successor][t_succ]['attr'].items():
                        t_data_size += (self.func_objs[t_func]['input'] + self.func_objs[t_func]['output']) * t_freq
                    if t_data_size > data_size:
                        check_flag = False
                        break
                if not check_flag:
                    continue

            path, value = self.find_max_critical_path(successor, early_stop)
            if max_value < value + data_size:
                max_value = value + data_size
                max_path = path

        if len(self.graph.pred[start_n]) == 1 and -1 in self.graph.pred[start_n]:
            t_data_size = 0
            for t_func, t_freq in self.graph[-1][start_n]['attr'].items():
                t_data_size += (self.func_objs[t_func]['input'] + self.func_objs[t_func]['output']) * t_freq
            max_value += t_data_size
        max_path = [start_n] + max_path
        self.critical_paths[start_n] = (max_path, max_value)
        return self.critical_paths[start_n]

    def remove_path(self, path: List[int]) -> None:
        for node in path:
            if node in self.graph:
                self.graph.remove_node(node)
        # the cache must be cleared since some nodes are removed
        self.critical_paths.clear()

    def draw(self):
        plt.figure(figsize=(20, 12))
        plt.subplot(121)
        nx.draw(self.graph, pos=nx.circular_layout(self.graph), arrowsize=15, node_size=400, node_color='#bce672', with_labels=True)
        nx.write_gexf(self.graph_bak, './test.gexf')
        nx.draw_networkx_edge_labels(self.graph, pos=nx.circular_layout(self.graph), font_size=8)
        plt.show()

    def is_solved(self):
        # problem only solved when there is no other nodes except user node (-1)
        return self.graph is not None and len(self.graph.nodes) <= 1
