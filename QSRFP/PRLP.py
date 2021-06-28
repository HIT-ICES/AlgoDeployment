#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：AlgoGroupDeployment 
@File ：PRLP.py
@Author ：septemberhx
@Date ：2021/4/4
@Description: Parametric relaxation linear programming (PRLP)
"""
import sympy
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, PULP_CBC_CMD


def y(p_jk, y_l, y_h):
    """
    Function definition for the series of y_j(pjk) at page 8
    :param p_jk: p
    :param y_l: low bound of y
    :param y_h: high bound of y
    """
    return y_l + p_jk * (y_h - y_l)


def varphi(y_j, y_k):
    """
    Function definition for varphi at page 8
    """
    return y_j * y_k


def varphi_l(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk):
    """
    Function definition for varphi_l at page 8
    """
    y_j_p = y(p_jk, y_j_l, y_j_h)
    y_k_p = y(p_jk, y_k_l, y_k_h)
    return y_j_p * y_k_p + y_k_p * (y_j - y_j_p) + y_k_p * (y_k - y_k_p)


def varphi_h(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk):
    """
    Function definition for varphi_h at page 8
    """
    y_j_p = y(p_jk, y_j_l, y_j_h)
    y_k_p = y(p_jk, y_k_l, y_k_h)
    return y_j_p * y_k_p + y(1 - p_jk, y_k_l, y_k_h) * (y_j - y_j_p) + y(1 - p_jk, y_j_l, y_j_h) * (y_k - y_k_p)


def phi_l(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk, v_jk):
    """
    Function definition for phi_l at page 10
    """
    if v_jk > 0:
        return v_jk * varphi_l(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk)
    elif v_jk < 0:
        return v_jk * varphi_h(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk)
    else:
        return 0


def psi_h(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk, v_jk):
    """
    Function definition for psi_h at page 10
    """
    if v_jk < 0:
        return v_jk * varphi_l(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk)
    elif v_jk > 0:
        return v_jk * varphi_h(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk)
    else:
        return 0


def lambda_l(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk, v_jk):
    """
    Function definition for lambda_l at page 10
    """
    if v_jk > 0:
        return v_jk * varphi_l(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk)
    elif v_jk < 0:
        return v_jk * varphi_h(y_j, y_j_l, y_j_h, y_k, y_k_l, y_k_h, p_jk)
    else:
        return 0


class PRLP:

    def __init__(self, fL_s, gU_hat_s, HL_s, sym_list, Y):
        self.fL_s = fL_s
        self.gU_hat_s = gU_hat_s
        self.HL_s = HL_s
        self.sym_list = sym_list
        self.vars = []
        for i in range(0, len(self.sym_list)):
            self.vars.append(LpVariable(name=self.sym_list[i].name, lowBound=Y[i][0], upBound=Y[i][1]))
        self.Y = Y
        assert len(fL_s) == len(gU_hat_s)

    def solve(self):
        model = self.to_PuLP()
        model.solve(PULP_CBC_CMD(msg=False))
        # print('=' * 20)
        # print(model)
        # print('=' * 20)
        # print(f"status: {model.status}, {LpStatus[model.status]}")
        # print(f"objective: {model.objective.value()}")

        opt_value = 0
        var_values = {}
        if model.status == 1:
            opt_value = model.objective.value()
            for var in model.variables():
                var_values[sympy.simplify(var.name)] = var.value()
        return model.status, opt_value, var_values

    def to_PuLP(self):
        model = LpProblem(name='PRLP from QSRFP', sense=LpMinimize)
        H0L = 0
        for i in range(0, len(self.fL_s)):
            H0L_i = (self.fL_s[i] / self.gU_hat_s[i]).expand()
            H0L += self.to_PuLP_single(H0L_i)

        # set object
        model += H0L

        # calculate constraint expressions in PuLP format
        for i in range(0, len(self.HL_s)):
            model += (self.to_PuLP_single(self.HL_s[i]) <= 0, 'Constraint {0}'.format(i))
        return model

    def to_PuLP_single(self, single):
        r = 0
        if isinstance(single, sympy.core.add.Add):
            for arg in single.args:
                r += self.to_PuLP_unit(arg)
        else:
            r += self.to_PuLP_unit(single)
        return r

    def to_PuLP_unit(self, unit):
        if unit.is_number:
            return float(unit)
        elif isinstance(unit, sympy.core.symbol.Symbol):
            return self.vars[self.sym_list.index(unit)]
        elif isinstance(unit, sympy.core.mul.Mul):
            coe = list(unit.as_coefficients_dict().values())[0]
            sym = self.vars[self.sym_list.index(list(unit.free_symbols)[0])]
            return sym * coe
        else:
            raise Exception('Unexpected unit {0}'.format(unit))
