#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：AlgoGroupDeployment 
@File ：QSRFP.py
@Author ：septemberhx
@Date ：2021/4/2
@Description: Algorithm Quadratic Sum-of-Ratios Fractional Programs Problem based on
  Hongwei Jiao & Sanyang Liu (2017): An Efficient Algorithm for Quadratic Sum-of-Ratios Fractional Programs Problem,
  Numerical Functional Analysis and Optimization, DOI: 10.1080/01630563.2017.1327869
"""
import math

import numpy as np
import sympy

from QSRFP import PRLP
from typing import Dict, List, Tuple


class QSRFP:

    sym_list: List[sympy.core.symbol.Symbol]
    Y: List[Tuple[float, float]]
    H_s: List[sympy.core.add.Add]
    g_s: List[sympy.core.add.Add]
    f_s: List[sympy.core.add.Add]

    def __init__(self, f_s, g_s, H_s, Y, sym_list):
        self.f_s = f_s
        self.g_s = g_s
        self.H_s = H_s
        self.Y = Y
        self.sym_list = sym_list

        # prepare
        self.p = len(self.f_s)
        self.M = len(self.H_s)
        self.n = len(self.sym_list)

        # f(x) stuffs at page 3
        self.delta = np.zeros(shape=(self.p, self.n, self.n), dtype=float)
        self.c = np.zeros(shape=(self.p, self.n), dtype=float)
        self.delta_hat = np.zeros(shape=(self.p, ), dtype=float)

        # g(x) stuffs at page 3
        self.beta = np.zeros(shape=(self.p, self.n, self.n), dtype=float)
        self.d = np.zeros(shape=(self.p, self.n), dtype=float)
        self.beta_hat = np.zeros(shape=(self.p, ), dtype=float)

        # H(x) stuffs at page 3
        self.gamma = np.zeros(shape=(self.M, self.n, self.n), dtype=float)
        self.e = np.zeros(shape=(self.M, self.n), dtype=float)
        self.gamma_hat = np.zeros(shape=(self.M, ), dtype=float)

        self.init_stuffs()

    def init_stuffs(self):
        self.init_f_stuffs()
        self.init_g_stuffs()
        self.init_h_stuffs()

    def init_f_stuffs(self):
        self.parse(self.f_s, self.delta, self.c, self.delta_hat)

    def init_g_stuffs(self):
        self.parse(self.g_s, self.beta, self.d, self.beta_hat)

    def init_h_stuffs(self):
        self.parse(self.H_s, self.gamma, self.e, self.gamma_hat)

    def parse(self, funcs, quad_coe, singe_coe, constant):
        for i in range(0, len(funcs)):
            # each f(x) should be quadratic sum-of-rations fractional program
            if isinstance(funcs[i], sympy.core.add.Add):
                for unit in funcs[i].args:
                    self.parse_unit(unit, i, quad_coe, singe_coe, constant)
            else:
                self.parse_unit(funcs[i], i, quad_coe, singe_coe, constant)

    def parse_unit(self, unit, i, quad_coe, singe_coe, constant):
        sym_size = len(unit.free_symbols)
        if sym_size > 2:
            raise Exception('Only quadratic is supported rather than {0}: {1}'.format(sym_size, unit))
        elif sym_size == 2:
            if not isinstance(unit, sympy.core.mul.Mul):
                raise Exception('Wrong object type {0} instead of sympy.core.mul.Mul'.format(type(unit)))
            delta = list(unit.as_coefficients_dict().values())[0]
            syms = list(unit.free_symbols)
            j = self.sym_list.index(syms[0])
            k = self.sym_list.index(syms[1])

            quad_coe[i][j][k] = float(delta)
            quad_coe[i][k][j] = float(delta)
        elif sym_size == 1:
            if isinstance(unit, sympy.core.symbol.Symbol):
                singe_coe[i][self.sym_list.index(unit)] = 1
            elif isinstance(unit, sympy.core.power.Pow):
                if unit.as_base_exp()[1] != 2:
                    raise Exception('Unexpected expression {0}'.format(unit))
                syms = list(unit.free_symbols)
                k = self.sym_list.index(syms[0])
                quad_coe[i][k][k] = 1
            else:
                coe = 1.0
                for arg in unit.args:
                    if arg.is_number:
                        coe = float(arg)
                    elif isinstance(arg, sympy.core.symbol.Symbol):
                        k = self.sym_list.index(arg)
                        singe_coe[i][k] = coe
                    elif isinstance(arg, sympy.core.power.Pow):
                        if arg.as_base_exp()[1] != 2:
                            raise Exception('Unexpected expression {0}'.format(arg))
                        syms = list(arg.free_symbols)
                        k = self.sym_list.index(syms[0])

                        quad_coe[i][k][k] = coe
        elif sym_size == 0:
            constant[i] = float(unit)
        else:
            raise Exception('Unexpected expression {0}'.format(unit))

    def reducing_parser(self, Y_t, rho):
        """
        calculate omega and varpi matrix at page 18
        """
        omega = np.zeros((self.M + 1, self.n), dtype=float)
        varpi = np.zeros((self.M + 1, ), dtype=float)

        fL_s = self.build_fL(Y_t, rho)
        gU_s = self.build_gU(Y_t, rho)
        gU_hat_s = self.build_gU_hat(gU_s, Y_t)
        HL_s = self.build_HL(Y_t, rho)

        HL_0 = sympy.simplify('0')
        for i in range(0, len(fL_s)):
            HL_0 += fL_s[i] / gU_hat_s[i]
        HL_0.expand()

        self.reducing_parser_one(HL_0, 0, omega, varpi)
        for i in range(0, len(HL_s)):
            self.reducing_parser_one(HL_s[i], 1 + i, omega, varpi)
        return omega, varpi

    def reducing_parser_one(self, func, index, omega, varpi):
        if isinstance(func, sympy.core.add.Add):
            for arg in func.args:
                self.reducing_parser_unit(arg, index, omega, varpi)
        else:
            self.reducing_parser_unit(func, index, omega, varpi)

    def reducing_parser_unit(self, unit, index, omega, varpi):
        if unit.is_number:
            varpi[index] = float(unit)
        elif isinstance(unit, sympy.core.symbol.Symbol):
            omega[index][self.sym_list.index(unit)] = 1
        elif isinstance(unit, sympy.core.mul.Mul):
            coe = list(unit.as_coefficients_dict().values())[0]
            sym = list(unit.free_symbols)[0]
            omega[index][self.sym_list.index(sym)] = coe
        else:
            raise Exception('Unexpected unit {0}'.format(unit))

    def to_PRLP(self, Y_t, rho):
        fL_s = self.build_fL(Y_t, rho)
        gU_s = self.build_gU(Y_t, rho)
        gU_hat_s = self.build_gU_hat(gU_s, Y_t)
        HL_s = self.build_HL(Y_t, rho)

        # print('fL_s = ', fL_s)
        # print('gU_s = ', gU_s)
        # print('gU_hat_s = ', gU_hat_s)
        # print('HL_s = ', HL_s)
        # print('delta = ', self.delta)
        # print('c = ', self.c)
        # print('delta_hat = ', self.delta_hat)
        #
        # print('beta = ', self.beta)
        # print('d = ', self.d)
        # print('beta_hat = ', self.beta_hat)
        #
        # print('gamma = ', self.gamma)
        # print('e = ', self.e)
        # print('gamma_hat = ', self.gamma_hat)

        prlp = PRLP.PRLP(fL_s, gU_hat_s, HL_s, self.sym_list, Y_t)
        return prlp.solve()

    def build_fL(self, Y_t, rho):
        """
        fL(x) at page 11
        """
        return self.build_LU(self.p, PRLP.phi_l, self.delta, self.c, self.delta_hat, Y_t, rho)

    def build_gU(self, Y_t, rho):
        """
        HL(x) at page 11
        """
        return self.build_LU(self.p, PRLP.psi_h, self.beta, self.d, self.beta_hat, Y_t, rho)

    def build_gU_hat(self, gU_s, Y_t):
        """
        gU_hat(x) at page 11
        """
        result = []
        for gU in gU_s:
            r = 0.0
            if isinstance(gU, sympy.core.add.Add):
                for arg in gU.args:
                    r += self.build_gU_hat_unit(arg, Y_t)
            else:
                r = self.build_gU_hat_unit(gU, Y_t)
            result.append(r)
        return result

    def build_gU_hat_unit(self, unit, Y_t):
        if unit.is_number:
            return float(unit)
        elif isinstance(unit, sympy.core.symbol.Symbol):
            return Y_t[self.sym_list.index(unit)][1]
        elif isinstance(unit, sympy.core.mul.Mul):
            coe = list(unit.as_coefficients_dict().values())[0]
            sym = list(unit.free_symbols)[0]
            return float(coe * Y_t[self.sym_list.index(sym)][1]) if coe > 0 else float(coe * Y_t[self.sym_list.index(sym)][0])
        else:
            raise Exception('Unexpected unit {0}'.format(unit))

    def build_HL(self, Y_t, rho):
        """
        HL(x) at page 11
        """
        return self.build_LU(self.M, PRLP.lambda_l, self.gamma, self.e, self.gamma_hat, Y_t, rho)

    def build_LU(self, length, LU_func, quad_coe, single_coe, constant, Y_t, rho):
        result = []
        for i in range(0, length):
            r = sympy.simplify(0)
            for j in range(0, self.n):
                for k in range(0, self.n):
                    r += LU_func(
                        y_j=self.sym_list[j],
                        y_j_l=Y_t[j][0],
                        y_j_h=Y_t[j][1],
                        y_k=self.sym_list[k],
                        y_k_l=Y_t[k][0],
                        y_k_h=Y_t[k][1],
                        p_jk=rho[j][k],
                        v_jk=quad_coe[i][j][k]
                    )
            for k in range(0, self.n):
                r += single_coe[i][k] * self.sym_list[k]
            r += constant[i]
            result.append(r)
        return result

    def is_feasible(self, sym_value_dict: Dict[sympy.core.symbol.Symbol, float]) -> bool:
        t_dict = dict(sym_value_dict)
        for sym in self.sym_list:
            if sym not in t_dict:
                t_dict[sym] = 0

        for H in self.H_s:
            if H.evalf(subs=t_dict) > 10e-6:
                return False
        return True

    def evalf(self, sym_value_dict: Dict[sympy.core.symbol.Symbol, float]) -> float:
        t_dict = dict(sym_value_dict)
        for sym in self.sym_list:
            if sym not in t_dict:
                t_dict[sym] = 0

        r = 0
        for i in range(0, len(self.f_s)):
            r += self.f_s[i].evalf(subs=t_dict) / self.g_s[i].evalf(subs=t_dict)
        return r

    def to_sym_value_dict(self, values: List[float]) -> Dict[sympy.core.symbol.Symbol, float]:
        sym_value_dict = {}
        for i in range(0, len(self.sym_list)):
            sym_value_dict[self.sym_list[i]] = values[i]
        return sym_value_dict
