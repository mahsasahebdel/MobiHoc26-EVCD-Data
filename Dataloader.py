from datetime import datetime, timedelta
import pytz
import math
import random
import csv


class DataLoader:
    def __init__(
        self,
        filepath,
        delta_minutes=15,
        instance_length_hours=24,
        L=1,
        U=10,
        ev_multiplier=20,
        num_stations=3,
        value_mode="fixed",
        fav_alpha=0.3,
        seed=42,
        timezone_str="America/Los_Angeles",
    ):
        self.delta = timedelta(minutes=delta_minutes)
        self.delta_minutes = self.delta.total_seconds() / 60.0
        self.instance_length = timedelta(hours=instance_length_hours)

        self.L, self.U = float(L), float(U)
        self.ev_multiplier = int(ev_multiplier)
        self.value_delta = (self.U - self.L) / self.ev_multiplier

        self.num_stations = int(num_stations)
        self.value_mode = value_mode
        self.fav_alpha = float(fav_alpha)

        self.X = self.fav_alpha * (self.U - self.L)
        self.fav_L = self.L + self.X
        self.fav_delta = (self.U - self.fav_L) / self.ev_multiplier

        self.timezone = pytz.timezone(timezone_str)

        self.meta = {}
        self.instances = []
        self.time_window = 0
        self.total_evs = 0
        self.theta = self.U
        self.D_max = 0
        self.D_min = float("inf")

        random.seed(seed)

        self._load(filepath)

    def _reset(self):
        self.meta = {}
        self.instances = []
        self.time_window = 0
        self.total_evs = 0
        self.theta = self.U
        self.D_max = 0
        self.D_min = float("inf")

    def __iter__(self):
        for inst in self.instances:
            yield inst

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _parse_datetime(self, time_str):
        dt = datetime.fromisoformat(str(time_str).strip())
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        else:
            dt = dt.astimezone(self.timezone)
        return dt

    def _time_to_index(self, dt, start_time):
        return int((dt - start_time).total_seconds() // self.delta.total_seconds())

    def _safe_float(self, x, default=0.0):
        try:
            if x is None or x == "":
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def get_value(self, total_req, batch_idx):
        low = self.L + batch_idx * self.value_delta
        high = self.L + (batch_idx + 1) * self.value_delta
        high = min(high, self.U)

        v = random.uniform(low, high)
        return [v * total_req for _ in range(self.num_stations)]

    def get_value_favorite(self, total_req, batch_idx, favorite_station):
        values = []

        for s in range(self.num_stations):
            if s == favorite_station:
                low = self.fav_L + batch_idx * self.fav_delta
                high = self.fav_L + (batch_idx + 1) * self.fav_delta
                high = min(high, self.U)
                v = random.uniform(low, high)
            else:
                upper_nonfav = min(
                    self.L + self.X,
                    self.L + (batch_idx + 1) * self.value_delta,
                )
                v = random.uniform(self.L, upper_nonfav)

            values.append(v * total_req)

        return values

    # --------------------------------------------------
    # Main loader
    # --------------------------------------------------

    def _load(self, filepath):
        self._reset()

        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            raise ValueError("The CSV file is empty.")

        required_cols = [
            "arrival_time",
            "charging_end_time",
            "energy_consumed_kWh",
            "charging_power_kW",
            "vehicle_id",
        ]

        missing_cols = [c for c in required_cols if c not in rows[0]]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        parsed_rows = []
        for row in rows:
            try:
                arrival = self._parse_datetime(row["arrival_time"])
                departure = self._parse_datetime(row["charging_end_time"])
            except Exception:
                continue

            if departure <= arrival:
                continue

            parsed_rows.append((row, arrival, departure))

        if len(parsed_rows) == 0:
            raise ValueError("No valid charging sessions were found.")

        global_start = min(arrival for _, arrival, _ in parsed_rows)
        global_end = max(departure for _, _, departure in parsed_rows)

        self.meta = {
            "site": "CSV_EV_dataset",
            "start": global_start,
            "end": global_end,
        }

        total_seconds = (global_end - global_start).total_seconds()
        num_instances = math.ceil(total_seconds / self.instance_length.total_seconds())

        instance_windows = []
        for i in range(num_instances):
            inst_start = global_start + i * self.instance_length
            inst_end = min(inst_start + self.instance_length, global_end)
            instance_windows.append((inst_start, inst_end))

        self.instances = [[] for _ in range(num_instances)]

        for row, arrival, departure in parsed_rows:
            inst_idx = int(
                (arrival - global_start).total_seconds()
                // self.instance_length.total_seconds()
            )

            inst_start, inst_end = instance_windows[inst_idx]

            arrival = max(arrival, inst_start)
            departure = min(departure, inst_end)

            arrival_idx = self._time_to_index(arrival, inst_start)
            departure_idx = self._time_to_index(departure, inst_start)

            if departure_idx > arrival_idx + 15:
                departure_idx = arrival_idx + 15

            if departure_idx <= arrival_idx:
                departure_idx = arrival_idx + 1

            self.D_max = max(self.D_max, departure_idx - arrival_idx + 1)
            self.D_min = min(self.D_min, departure_idx - arrival_idx + 1)
            self.time_window = max(self.time_window, arrival_idx + 1, departure_idx + 1)

            energy = self._safe_float(row.get("energy_consumed_kWh"), default=5.0)
            energy /= 5.0

            charging_power = self._safe_float(row.get("charging_power_kW"), default=0.0)
            rateLimit = charging_power * (self.delta_minutes / 60.0)
            rateLimit /= 5.0

            if rateLimit <= 0:
                rateLimit = 3.0

            available_slots = departure_idx - arrival_idx + 1

            # make sure the EV is individually feasible
            min_feasible_rate = energy / available_slots if available_slots > 0 else energy
            rateLimit = max(rateLimit, min_feasible_rate)

            ev = {
                "arrival": arrival_idx,
                "departure": departure_idx,
                "demand": energy,
                "rateLimit": rateLimit,
                "id": row.get("vehicle_id"),
                "batch": 0,
            }

            fav_station = random.randrange(self.num_stations)

            for i in range(self.ev_multiplier):
                ev_i = dict(ev)
                ev_i["batch"] = i + 1

                if self.value_mode == "stationwise":
                    ev_i["favorite_station"] = fav_station
                    ev_i["value"] = self.get_value_favorite(
                        energy, batch_idx=i, favorite_station=fav_station
                    )
                else:
                    ev_i["value"] = self.get_value(energy, batch_idx=i)

                self.instances[inst_idx].append(ev_i)
                self.total_evs += 1

        for inst_idx in range(len(self.instances)):
            self.instances[inst_idx] = sorted(
                self.instances[inst_idx],
                key=lambda x: (x["batch"], x["arrival"], x["departure"]),
            )

        if self.D_min == float("inf"):
            self.D_min = 0

        print(
            f"[EVDataLoader] Data loading completed.\n"
            f"  • Number of instances      : {len(self.instances)}\n"
            f"  • Total EVs loaded         : {self.total_evs}\n"
            f"  • Instance length (hours)  : {self.instance_length.total_seconds() / 3600}\n"
            f"  • Time discretization (min): {self.delta_minutes}\n"
            f"  • Time window (# of slots) : {self.time_window}\n"
            f"  • D_min                    : {self.D_min}\n"
            f"  • D_max                    : {self.D_max}\n"
        )