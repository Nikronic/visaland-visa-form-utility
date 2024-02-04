from random import choice, choices
import numpy as np

dir_path = "vizard/seduce/models/names/"


def first_name(gender):
    if gender.lower() in ["m", "male"]:
        filename = "male_name.txt"
    elif gender.lower() in ["f", "female"]:
        filename = "female_name.txt"
    elif gender.lower() in ["r", "random"]:
        filename = choice(['male_name.txt','female_name.txt'])
    else:
        raise ValueError
    with open(f"{dir_path}{filename}", "r", encoding="utf8") as f:
        names = f.read().split("\n")
    return choice(names)


def last_name():
    with open(f"{dir_path}last_name.txt", "r", encoding="utf8") as f:
        names = f.read().split("\n")
    return choice(names)


def record_generator(acceptance_rate, n=5):
    records = []
    acceptance_statuses = choices(
        [True, False], weights=[acceptance_rate, 1 - acceptance_rate], k=n
    )
    for i in range(n):
        records.append(
            {
                "first_name": first_name("r"),
                "last_name": last_name(),
                "acceptance_status": acceptance_statuses[i],
            }
        )
    return records
print(record_generator(0.5, 5))