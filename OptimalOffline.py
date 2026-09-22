import gurobipy as gp
from gurobipy import GRB


class OptimalOffline:

    def __init__(self, capacities, time_window):
        self.capacities = capacities
        self.time_window = time_window

    def solve_offline(self, evs, v=None, R=None, time_limit=60 * 2, verbose=False):
        """
        Offline MILP (ignore price term):
            max sum_{n,k} v[n,k] * x[n,k]
            s.t.  sum_{t in T_n} y[n,t,k] >= D_n * x[n,k]         for all n,k
                sum_{n active at t} y[n,t,k] <= C_k             for all t,k
                y[n,t,k] <= R[n,k] * x[n,k]                     for all n,k,t in T_n
                sum_k x[n,k] <= 1                               for all n
                x[n,k] in {0,1}, y[n,t,k] >= 0

        Inputs:
        evs: list of dicts, each ev has:
            ev['arrival'] (int), ev['departure'] (int, exclusive),
            ev['demand'] (float)
        capacities: list length K, C_k per station per time
        time_horizon: int T, times are 0..T-1
        v: optional value matrix. If None, uses ev['value'] same for all stations.
            - If provided: v is a 2D list/array shape (N,K) or dict (n,k)->value.
        R: optional rate limit matrix. If None, uses ev['rateLimit'] same for all stations.
            - If provided: R is a 2D list/array shape (N,K) or dict (n,k)->rate.
        time_limit: optional seconds for solver
        verbose: if True, prints solver output

        Returns:
        model, x_sol, y_sol, admitted
            x_sol: dict (n,k)->0/1
            y_sol: dict (n,t,k)->float (only positive ones stored)
            admitted: list of (n,k) pairs where x=1
        """
        N = len(evs)
        K = len(self.capacities)
        T = self.time_window

        def get_v(n, k):
            if v is None:
                print("None value found!")
                return float(evs[n]["value"][k])
            if isinstance(v, dict):
                print("Dict type found!")
                return float(v[(n, k)])
            return float(v[n][k])

        def get_R(n, k):
            if R is None:
                return float(evs[n].get("rateLimit", 0.0))
            if isinstance(R, dict):
                return float(R[(n, k)])
            return float(R[n][k])

        m = gp.Model("offline_ignore_prices")
        m.Params.OutputFlag = 1 if verbose else 0
        if time_limit is not None:
            m.Params.TimeLimit = float(time_limit)

        # ---------------- Variables ----------------
        x = m.addVars(N, K, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        y = {}
        for n in range(N):
            a = int(evs[n]["arrival"])
            d = int(evs[n]["departure"])
            a = max(0, a)
            d = min(T - 1, d)
            for t in range(a, d + 1):
                for k in range(K):
                    y[(n, t, k)] = m.addVar(
                        lb=0.0, vtype=GRB.CONTINUOUS, name=f"y[{n},{t},{k}]"
                    )

        m.update()

        # ---------------- Constraints ----------------

        # (1) each EV assigned to at most one station
        for n in range(N):
            m.addConstr(
                gp.quicksum(x[n, k] for k in range(K)) <= 1, name=f"assign[{n}]"
            )

        # (2) demand satisfaction if assigned: for each n,k
        for n in range(N):
            a = int(evs[n]["arrival"])
            d = int(evs[n]["departure"])
            a = max(0, a)
            d = min(T - 1, d)
            Dn = float(evs[n]["demand"])
            for k in range(K):
                m.addConstr(
                    gp.quicksum(y[(n, t, k)] for t in range(a, d + 1)) == Dn * x[n, k],
                    name=f"demand[{n},{k}]",
                )

        # (3) per-slot station capacity: for each t,k
        # Only include EVs active at time t (those for which y exists)
        for t in range(T):
            for k in range(K):
                expr = gp.LinExpr()
                for n in range(N):
                    a = int(evs[n]["arrival"])
                    d = int(evs[n]["departure"])
                    a = max(0, a)
                    d = min(T - 1, d)
                    if a <= t <= d:
                        expr += y[(n, t, k)]
                m.addConstr(expr <= float(self.capacities[k]), name=f"cap[{t},{k}]")

        # (4) per-slot rate limit tied to assignment: y[n,t,k] <= R[n,k]*x[n,k]
        for n in range(N):
            a = int(evs[n]["arrival"])
            d = int(evs[n]["departure"])
            a = max(0, a)
            d = min(T - 1, d)
            for k in range(K):
                Rnk = float(get_R(n, k))
                # If Rnk is 0, this forces y=0 unless x=0; usually you'd avoid 0.
                for t in range(a, d + 1):
                    m.addConstr(
                        y[(n, t, k)] <= Rnk * x[n, k], name=f"rate[{n},{t},{k}]"
                    )

        # ---------------- Objective (IGNORE price term) ----------------
        m.setObjective(
            gp.quicksum(get_v(n, k) * x[n, k] for n in range(N) for k in range(K)),
            GRB.MAXIMIZE,
        )

        # Solve
        m.optimize()

        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return m, {}, {}, []

        # Extract solutions
        x_sol = {(n, k): float(x[n, k].X) for n in range(N) for k in range(K)}

        y_sol = {}
        for (n, t, k), var in y.items():
            val = float(var.X)
            if val > 1e-9:
                y_sol[(n, t, k)] = val

        admitted = None
        return m.ObjVal, x_sol, y_sol, admitted

    def optimize(self, instance):
        total_utility, x_sol, y_sol, admitted = self.solve_offline(instance)

        return total_utility
