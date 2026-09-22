import gurobipy as gp
from gurobipy import GRB
import numpy as np

from OnlineAlgorithm import OnlineAlgorithm


class MyAlg(OnlineAlgorithm):
    def __init__(self, capacities, time_window):
        super().__init__(
            algName="myAlg", capacities=capacities, time_window=time_window
        )

        # define additional parameters here (if needed)
        self.beta = 0  # TODO
        self.eta = 0  # TODO

    def set_beta(self, beta):
        self.beta = beta

    def set_eta(self, eta):
        self.eta = eta

    def decide(self, ev):
        arrival = ev["arrival"]
        departure = ev["departure"]
        demand = ev["demand"]
        rate_limit = ev["rateLimit"]
        value = ev["value"]

        if arrival < 0 or departure > self.time_window or arrival > departure:
            return False, {}
        if demand <= 0 or rate_limit <= 0:
            return False, {}

        beta = float(self.beta)
        eta = float(self.eta)
        if eta <= 0 or beta <= 0:
            return False, {}

        def phi(w, C):
            return eta * (beta ** (w / C) - 1.0)

        # def waterfill_gurobi(util_row, C, arrival, departure, D, R, time_limit=10.0):
        #     """
        #     Capped water-filling via Gurobi general constraints:
        #
        #     y_t = min(b_t, max(0, ell - w_t))
        #     sum_t y_t == D
        #     b_t = min(R, C - w_t)
        #
        #     Returns: dict {t: y_t} or None if infeasible / no solution.
        #     """
        #
        #     times = list(range(arrival, departure + 1))
        #     w = {t: float(util_row[t]) for t in times}
        #     C = float(C)
        #     D = float(D)
        #     R = float(R)
        #
        #     # Per-slot cap b_t = min(R, C - w_t), but never negative
        #     b = {t: max(0.0, min(R, C - w[t])) for t in times}
        #
        #     # Quick infeasibility check
        #     if D > sum(b.values()) + 1e-9:
        #         return None
        #
        #     m = gp.Model("waterfill")
        #     m.Params.OutputFlag = 0  # silent
        #     m.Params.TimeLimit = time_limit
        #
        #     # Variables
        #     ell = m.addVar(lb=-GRB.INFINITY, name="ell")  # water level
        #     y = m.addVars(times, lb=0.0, ub=[b[t] for t in times], name="y")
        #
        #     # Helpers: u_t = ell - w_t, r_t = max(0, u_t)
        #     u = m.addVars(times, lb=-GRB.INFINITY, name="u")
        #     r = m.addVars(times, lb=0.0, name="r")
        #
        #     for t in times:
        #         m.addConstr(u[t] == ell - w[t], name=f"u_def[{t}]")
        #
        #         # r[t] = max(0, u[t])
        #         m.addGenConstrMax(r[t], [u[t], 0.0], name=f"relu[{t}]")
        #
        #         # y[t] = min(b[t], r[t])
        #         # (This enforces the cap, including rateLimit.)
        #         m.addGenConstrMin(y[t], [r[t], b[t]], name=f"cap[{t}]")
        #
        #     # Meet exactly the demand (water-filling solution is at equality)
        #     m.addConstr(gp.quicksum(y[t] for t in times) == D, name="demand")
        #
        #     # Objective: minimize the water level (encourages filling low-utilization slots first)
        #     # This yields the canonical capped water-filling solution.
        #     m.setObjective(ell, GRB.MINIMIZE)
        #
        #     m.optimize()
        #
        #     if m.Status != GRB.OPTIMAL:
        #         return None
        #
        #     sched = {t: float(y[t].X) for t in times if y[t].X > 1e-9}
        #     return sched

        def waterfill_implemented(util_row, C, arrival, departure, D, R):
            util_row = np.asarray(util_row, dtype=float)
            indices = np.arange(arrival, departure + 1)

            # Maximum feasible increase per index
            max_inc = np.minimum(R, C - util_row[indices])
            max_inc = np.maximum(max_inc, 0.0)

            # Quick feasibility check
            if D < 0 or max_inc.sum() + 1e-12 < D:
                return None

            # Sort by current utilization (water-filling)
            order = np.argsort(util_row[indices])
            sorted_idx = indices[order]
            sorted_util = util_row[sorted_idx]
            sorted_cap = max_inc[order]

            n = len(sorted_idx)
            delta = np.zeros(n)
            remaining = D

            for i in range(n):
                width = i + 1

                # Next water level
                next_level = sorted_util[i + 1] if i + 1 < n else float("inf")

                gap = next_level - sorted_util[i]

                # Per-element limit still available
                feasible = min(gap, sorted_cap[:width].min())

                required = feasible * width

                if remaining >= required:
                    delta[:width] += feasible
                    remaining -= required
                else:
                    delta[:width] += remaining / width
                    remaining = 0
                    break

            # Numerical check
            if remaining > 1e-8:
                return None

            # Build result dictionary
            result = {
                int(sorted_idx[i]): float(delta[i]) for i in range(n) if delta[i] > 0
            }

            return result

        best_station = None
        best_schedule = None
        best_net = float("-inf")

        for s in range(self._number_of_stations_):
            C = float(self.capacities[s])

            # 1) minimization step (water-filling) to get candidate schedule
            # if self.optimizer == 'gurobi':
            #     temp_schedule = waterfill_gurobi(
            #         util_row=self.utilizations[s],
            #         C=C,
            #         arrival=arrival,
            #         departure=departure,
            #         D=demand,
            #         R=rate_limit
            #     )
            # else:

            temp_schedule = waterfill_implemented(
                util_row=self.utilizations[s],
                C=C,
                arrival=arrival,
                departure=departure,
                D=demand,
                R=rate_limit,
            )
            if temp_schedule is None:
                continue  # infeasible at this station
            xi = 0.0
            for t, y in temp_schedule.items():
                w_after = self.utilizations[s][t] + y
                xi += phi(w_after, C) * y

            if value[s] < xi:
                continue

            net = value[s] - xi
            if net > best_net:
                best_net = net
                best_station = s
                best_schedule = temp_schedule

        if best_station is None:
            return False, {}

        return True, {best_station: best_schedule}
