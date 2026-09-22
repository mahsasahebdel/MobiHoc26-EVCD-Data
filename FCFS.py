from OnlineAlgorithm import OnlineAlgorithm


class FCFS(OnlineAlgorithm):
    def __init__(self, capacities, time_window):
        super().__init__(algName="FCFS", capacities=capacities, time_window=time_window)

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

        for s in range(self._number_of_stations_):
            C = float(self.capacities[s])

            remaining = float(demand)
            temp_schedule = {}

            times = list(range(arrival, departure + 1))
            times.sort(key=lambda t: self.utilizations[s][t])

            for t in times:
                if remaining <= 1e-9:
                    break

                cap_left = C - self.utilizations[s][t]
                if cap_left <= 1e-10:
                    continue

                y = min(rate_limit, cap_left, remaining)
                if y > 1e-12:
                    temp_schedule[t] = y
                    remaining -= y

            if remaining > 1e-9:
                continue

            return True, {s: temp_schedule}  # assigning to the first available station

        return False, {}
