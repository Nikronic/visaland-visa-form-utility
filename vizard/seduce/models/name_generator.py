from random import choice, choices, uniform

dir_path = "vizard/seduce/models/names/"

class RecordGenerator:
    """
    A class to generate fake records for the purpose of UI/UX.
    """
    def __init__(self, acceptance_rate, n=5):
        self.acceptance_rate = acceptance_rate
        self.n = n

    def first_name(self, gender):
        if gender.lower() in ["m", "male"]:
            filename = "male_name.txt"
        elif gender.lower() in ["f", "female"]:
            filename = "female_name.txt"
        elif gender.lower() in ["r", "random"]:
            filename = choice(["male_name.txt", "female_name.txt"])
        else:
            raise ValueError
        with open(f"{dir_path}{filename}", "r", encoding="utf8") as f:
            names = f.read().split("\n")
        return choice(names)

    def last_name(self):
        with open(f"{dir_path}last_name.txt", "r", encoding="utf8") as f:
            names = f.read().split("\n")
        return choice(names)

    def acceptance_rates(self, acceptance_rate, n=5, distance_radius=0.1):
        acceptance_rates_list = []
        for i in range(n):
            acceptance_rates_list.append(
                round(
                    uniform(
                        acceptance_rate - distance_radius,
                        acceptance_rate + distance_radius,
                    ),
                    2,
                )
            )
        return acceptance_rates_list

    def record_generator(self):
        n = self.n
        records = []
        acceptance_rate = self.acceptance_rate
        acceptance_statuses = choices(
            [True, False], weights=[acceptance_rate, 1 - acceptance_rate], k=n
        )
        acceptance_rates_list = self.acceptance_rates(acceptance_rate, n)
        # acceptence rate should be between 0 and 1
        acceptance_rates_list = [
            0.99 if rate > 1 else 0.02 if rate < 0 else rate for rate in acceptance_rates_list
        ]

        for i in range(n):
            records.append(
                {
                    "first_name": self.first_name("r"),
                    "last_name": self.last_name(),
                    "acceptance_rate": acceptance_rates_list[i],
                    "acceptance_status": acceptance_statuses[i],
                }
            )
        return records
