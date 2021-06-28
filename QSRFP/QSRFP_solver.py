#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：AlgoGroupDeployment 
@File ：QSRFP_solver.py
@Author ：septemberhx
@Date ：2021/4/6
@Description:
"""
import sympy, math

from QSRFP.QSRFP import QSRFP
import numpy as np
from typing import Dict, List, Tuple

from commom.logger import get_logger


def Gamma_m(m, omega, varpi, Y):
    """
    Gamma_m(x) at page 18
    """
    r = 0
    for j in range(0, omega.shape[1]):
        r += min(omega[m][j] * Y[j][0], omega[m][j] * Y[j][1])
    r += varpi[m]
    return r


def Gamma_ms(m, s, omega, varpi, Y):
    """
    Gamma_ms(x) at page 18
    """
    r = 0
    for j in range(0, omega.shape[1]):
        if j == s:
            continue
        r += min(omega[m][j] * Y[j][0], omega[m][j] * Y[j][1])
    r += varpi[m]
    return r


def zeta_s(s, UBk, omega, varpi, Y):
    """
    zeta_s(x) at page 18
    """
    return (UBk - Gamma_ms(0, s, omega, varpi, Y)) / omega[0][s]


def xi_ms(m, s, omega, varpi, Y):
    """
    xi_ms(x) at page 18
    """
    return -Gamma_ms(m, s, omega, varpi, Y) / omega[m][s]


def bisection(Yk: List[Tuple[float, float]]) -> (List[Tuple[float, float]], List[Tuple[float, float]]):
    """
    Branching technique at page 16
    """
    theta = 0
    diff = 0
    for i in range(0, len(Yk)):
        if Yk[i][1] - Yk[i][0] > diff:
            diff = Yk[i][1] - Yk[i][0]
            theta = i

    Yk_1 = []
    Yk_2 = []
    for i in range(0, len(Yk)):
        if i != theta:
            Yk_1.append(Yk[i])
            Yk_2.append(Yk[i])
        else:
            Yk_1.append((Yk[theta][0], (Yk[theta][0] + Yk[theta][1]) / 2))
            Yk_2.append(((Yk[theta][0] + Yk[theta][1]) / 2, Yk[theta][1]))
    return Yk_1, Yk_2


def midpoint(Yk: List[Tuple[float, float]]):
    result = []
    for yk in Yk:
        result.append((yk[0] + yk[1]) / 2)
    return result


class QSRFP_solver:

    def __init__(self, qsrfp: QSRFP):
        self.qsrfp = qsrfp
        self.logger = get_logger(QSRFP_solver.__name__)

    def solve(self, epsilon=0.01) -> (float, Dict[sympy.core.symbol.Symbol, float]):
        """
        Main algorithm at page 19
        """
        # Step 1. (Initializing)
        UB_0 = 100000000
        rho = np.zeros((self.qsrfp.n, self.qsrfp.n))
        status, LB_0, y_0 = self.qsrfp.to_PRLP(self.qsrfp.Y, rho)

        if status != 1:
            raise Exception('Failed to solve problem due to the failure at the beginning')

        y_k = None
        final_Y = None
        if self.qsrfp.is_feasible(y_0):
            UB_0 = min(self.qsrfp.evalf(y_0), UB_0)
            y_k = y_0
        if UB_0 - LB_0 <= epsilon:
            return UB_0, y_0

        Pi_0 = list_set()
        Pi_0.add(self.qsrfp.Y)
        F = list_set()

        UB_k_1 = UB_0  # UB_k-1
        Y_k_1 = self.qsrfp.Y  # Y_k-1
        Pi_k_1 = Pi_0
        prlp_result_cache = {}
        LB_k = LB_0

        k = 1
        Y_k = Y_k_1
        while True:
            self.logger.debug('-' * 20 + 'round starts' + '-' * 20)
            tmp_set = list_set()
            if not F.contains(Y_k_1):
                UB_k = UB_k_1
                # Step 2. (Bisection)
                Y_k1, Y_k2 = bisection(Y_k_1)
                F.add(Y_k_1)

                self.logger.debug(f'Y_k-1 = {Y_k_1}')
                self.logger.debug(f'Before reducing, \tY_k1 = {Y_k1}, \tY_k2 = {Y_k2}')
                # Step 3. (Reducing)
                deleted_Y = list_set()
                for Y_kt in (Y_k1, Y_k2):
                    omega, varpi = self.qsrfp.reducing_parser(Y_kt, rho)
                    for m in range(1, self.qsrfp.M + 1):
                        gamma_m = Gamma_m(m, omega, varpi, Y_kt)
                        if gamma_m > 10e-6:
                            self.logger.debug(f'Abandon {Y_kt} due to Gamma_m = {gamma_m} > 0')
                            deleted_Y.add(Y_kt)
                            break
                    if not deleted_Y.contains(Y_kt):
                        for m in range(1, self.qsrfp.M + 1):
                            for s in range(0, self.qsrfp.n):
                                if omega[m][s] == 0:
                                    continue
                                xi = xi_ms(m, s, omega, varpi, Y_kt)
                                if omega[m][s] > 0 and xi < Y_kt[s][1]:
                                    Y_kt[s] = (Y_kt[s][0], xi)
                                elif omega[m][s] < 0 and xi > Y_kt[s][0]:
                                    Y_kt[s] = (xi, Y_kt[s][1])
                self.logger.debug(f'Reducing step 1, \tY_k1 = {Y_k1}, \tY_k2 = {Y_k2}')

                for Y_kt in (Y_k1, Y_k2):
                    if deleted_Y.contains(Y_kt):
                        continue
                    omega, varpi = self.qsrfp.reducing_parser(Y_kt, rho)
                    gamma_0 = Gamma_m(0, omega, varpi, Y_kt)
                    if gamma_0 > UB_k:
                        deleted_Y.add(Y_kt)
                        self.logger.debug(f'Abandon {Y_kt} due to Gamma_m = {gamma_0} > 0')
                    else:
                        for s in range(0, self.qsrfp.n):
                            zeta = zeta_s(s, UB_k, omega, varpi, Y_kt)
                            if omega[0][s] > 0 and zeta < Y_kt[s][1]:
                                Y_kt[s] = (Y_kt[s][0], zeta)
                            elif omega[0][s] < 0 and zeta > Y_kt[s][0]:
                                Y_kt[s] = (zeta, Y_kt[s][1])
                self.logger.debug(f'Reducing step 2, \tY_k1 = {Y_k1}, \tY_k2 = {Y_k2}')

                # Step 4. (Bounding)
                Pi_k = list_set()
                for Y_k_t in (Y_k1, Y_k2):
                    if deleted_Y.contains(Y_k_t):
                        continue
                    status, LB_kt, y_kt = self.qsrfp.to_PRLP(Y_k_t, rho)
                    if status != 1:
                        self.logger.debug(f'Failed to solve PRLP with bound {Y_k_t}, skip.')
                        deleted_Y.add(Y_k_t)
                        continue

                    Y_k_tuple = tuple((x[0], x[1]) for x in Y_k_t)
                    prlp_result_cache[Y_k_tuple] = (LB_kt, y_kt)

                    y_mid = midpoint(Y_k_t)
                    y_mid_dict = self.qsrfp.to_sym_value_dict(y_mid)
                    if self.qsrfp.is_feasible(y_mid_dict):
                        t_mid = self.qsrfp.evalf(y_mid_dict)
                        if t_mid <= UB_k + 10e-6:
                            UB_k = t_mid
                            y_k = y_mid_dict
                            final_Y = Y_k_t
                    else:
                        self.logger.debug(f'mid point {y_mid} of {Y_k_t} infeasible')

                    if self.qsrfp.is_feasible(y_kt):
                        t_k = self.qsrfp.evalf(y_kt)
                        if t_k <= UB_k + 10e-6:
                            UB_k = t_k
                            y_k = y_kt
                            final_Y = Y_k_t
                    else:
                        self.logger.debug(f'y_kt {y_kt} of {Y_k_t} infeasible')

                    # Step 5. (Bounding)
                    if UB_k <= LB_kt + 10e-6:
                        F.add(Y_k_t)

                for Y_kt in (Y_k1, Y_k2):
                    if not deleted_Y.contains(Y_kt):
                        tmp_set.add(Y_kt)

            Pi_k = Pi_k_1.union(tmp_set).sub(F)

            if len(Pi_k) == 0:
                self.logger.debug('something strange happened here with empty Pi_k set')
            else:
                min_v = 1000000
                min_Y = None
                min_y = None
                for Y_k_tuple in Pi_k.values:
                    if min_v + 10e-6 >= prlp_result_cache[Y_k_tuple][0]:
                        min_v, min_y = prlp_result_cache[Y_k_tuple]
                        min_Y = [(x, y) for x, y in Y_k_tuple]
                LB_k = min_v
                Y_k = min_Y

            self.logger.info(f'k = {k}, len(Pi_k) = {len(Pi_k)}, LB_k = {LB_k}, UB_k = {UB_k}, y_k = {y_k}, Y_k = {Y_k}, LB_y_k = {min_y}, final_Y = {final_Y}')
            k += 1
            # Step 6. (Judgement)
            if UB_k - LB_k <= epsilon or len(Pi_k) == 0:
                return self.qsrfp.evalf(y_k), y_k
            else:
                Y_k_1 = Y_k
                UB_k_1 = UB_k
                Pi_k_1 = Pi_k
            self.logger.debug('-' * 20 + 'round ends' + '-' * 20)


class list_set:

    def __init__(self):
        self.values = set()

    def __len__(self):
        return len(self.values)

    def add(self, l):
        self.values.add(tuple((x[0], x[1]) for x in l))

    def contains(self, l):
        return tuple((x[0], x[1]) for x in l) in self.values

    def union(self, o):
        n = list_set()
        n.values = self.values.union(o.values)
        return n

    def sub(self, o):
        n = list_set()
        n.values = self.values - o.values
        return n

    def remove(self, l):
        self.values.remove(tuple((x[0], x[1]) for x in l))
