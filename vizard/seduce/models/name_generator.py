from random import choice, choices, uniform

dir_path = "vizard/seduce/models/names/"


class RecordGenerator:
    """
    A class to generate fake records for the purpose of UI/UX.
    """

    def __init__(self, acceptance_rate, n=5):
        self.acceptance_rate = acceptance_rate
        self.n = n

    def first_name_fa(self, gender):
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

    def last_name_fa(self):
        with open(f"{dir_path}last_name.txt", "r", encoding="utf8") as f:
            names = f.read().split("\n")
        return choice(names)

    def last_name_en(self):
        with open(f"{dir_path}last_name_en.txt", "r", encoding="utf8") as f:
            names = f.read().split("\n")
        return choice(names)

    def first_name_en(self, gender):
        if gender.lower() in ["m", "male"]:
            filename = "male_name_en.txt"
        elif gender.lower() in ["f", "female"]:
            filename = "female_name_en.txt"
        elif gender.lower() in ["r", "random"]:
            filename = choice(["male_name_en.txt", "female_name_en.txt"])
        else:
            raise ValueError
        with open(f"{dir_path}{filename}", "r") as f:
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
            0.99 if rate > 1 else 0.02 if rate < 0 else rate
            for rate in acceptance_rates_list
        ]

        for i in range(n):
            records.append(
                {
                    "first_name": self.first_name_fa("r"),
                    "last_name": self.last_name_fa(),
                    "acceptance_rate": acceptance_rates_list[i],
                    "acceptance_status": acceptance_statuses[i],
                }
            )
        return records

    def name_chance_generator(self, n=20):
        cases = []
        for i in range(n):
            if uniform(0, 1) < 0.7:
                first_name = self.first_name_fa("r")
                last_name = self.last_name_fa()
            else:
                first_name = self.first_name_en("r")
                last_name = self.last_name_en()

            chance = round(uniform(0.03, 0.98), 2)
            cases.append(
                {"first_name": first_name, "last_name": last_name, "chance": chance}
            )
        if uniform(0, 1) < 0.8:
            cases.append({"first_name": "asd", "last_name": "sad", "chance": 0.99})
        return cases
