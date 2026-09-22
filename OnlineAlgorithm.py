from abc import ABC, abstractclassmethod


class OnlineAlgorithm(ABC):
    def __init__(self, algName, capacities, time_window):
        self.algName = algName
        self._number_of_stations_ = len(capacities)
        self.capacities = capacities
        self.time_window = time_window

        self.reset()

    def reset(self):
        self.utilizations = [
            [0 for _ in range(self.time_window)]
            for __ in range(self._number_of_stations_)
        ]

    def admit(self, schedule):
        for station in schedule:
            for t in schedule[station]:
                self.utilizations[station][t] += schedule[station][t]

    @abstractclassmethod
    def decide(self, ev):
        """_summary_

        gets information of ev and returns a boolean, showing whether to admit the ev or not.
        in the case of admission, it returns dictionary of scheduling.
        Keys in the scheudling dictionary shows the index of charging stations.
        Each value is also andother dictionary that each key in the sub-dictionary
        is the time index, and value is the unit of energy for that time.

        example:
        schedule = {
        0: {10: 0.1, 11:0.2, 12: 0.1}
        }

        shows ev has admitted to station index 0 and there are three charging for times 10, 11, and 12

        """
        schedule = {}
        return False, schedule
