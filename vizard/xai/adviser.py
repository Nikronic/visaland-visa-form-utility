import json

with open("vizard/xai/adv.json", "r", encoding="utf-8") as f:
    adv = json.load(f)


def case_handling(inputs):
    re = []

    if inputs["sex"] == "Male":
        re.append("man")
    else:
        re.append("woman")

    job_title = inputs["occupation_title1"]
    if job_title == "employee":
        re.append("employee-main")
    elif job_title == "student":
        re.append("student-main")
    elif job_title == "OTHER":
        re.append("unemployed-main")
    elif job_title == "manager":
        re.append("employer-main")

    age = inputs["date_of_birth"]
    if age < 30:
        re.append("under-30")
    elif age < 40:
        re.append("30-40")
    elif age < 50:
        re.append("40-50")
    else:
        re.append("upper-50")

    child_count = inputs["child_count"]
    if child_count == 0:
        re.append("no-child")
    else:
        re.append("has-child")

    parent_accompany = inputs["parent_accompany"]
    spouse_accompany = inputs["spouse_accompany"]
    sibling_accompany = inputs["sibling_accompany"]
    child_accompany = inputs["child_accompany"]

    if parent_accompany or spouse_accompany or sibling_accompany or child_accompany:
        if spouse_accompany:
            if child_accompany == 0:
                re.append("with-spouse")
            elif child_accompany == child_count:
                re.append("with-spouse-all-childs")
            else:
                re.append("with-spouse-some-childs")
        else:
            if child_accompany != 0:
                if child_accompany == child_count:
                    re.append("with-all-childs")
                else:
                    re.append("with-some-childs")
    else:
        re.append("alone")

    insurance_history = inputs["occupation_period"]
    if insurance_history < 2:
        re.append("under-2")
    else:
        re.append("upper-2")

    applicant_marital_status = inputs["applicant_marital_status"]
    re.append(applicant_marital_status)

    invitation_letter = inputs["invitation_letter"]
    if invitation_letter == "none":
        re.append("no-invitation")
    else:
        re.append("has-invitation")
        if invitation_letter in ["parent", "siblings", "child"]:
            re.append("family")
        elif invitation_letter in ["f2", "f3"]:
            re.append("relatives-2-3")
        elif invitation_letter == "friend":
            re.append("friend")

    travel_history = inputs["travel_history"]
    if travel_history == "none":
        re.append("no-trip")
    else:
        other = True
        for item in travel_history:
            if item == "us_uk_au":
                re.append("us-aus-en-ja")
                other = False

            if item == "schengen_once":
                re.append("shcengen-1")
                re.append("shcengen")
                other = False

            if item == "schengen_twice":
                re.append("shcengen-2-upper-2")
                re.append("shcengen")
                other = False

            if other:
                re.append("other-countries")

    re.extend(
        [
            "employee-main",
            "under-2",
            "married",
            "has-child",
        ]
    )
    return re


def adviser(input):
    rn = []
    for item in adv:
        flag = True
        for condition in item["conditions"]:
            if isinstance(condition, list):
                all_false = True
                for con in condition:
                    if con in case_handling(input):
                        all_false = False
                        break
                if all_false:
                    flag = False
            if condition not in case_handling(input):
                flag = False
                break
        if flag:
            rn.append({"result:": item["result"], "advice": item["advice"]})
    return rn
