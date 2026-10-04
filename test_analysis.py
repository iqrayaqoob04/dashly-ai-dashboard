import pandas as pd
from api.analyze import profile, plan, execute

df=pd.DataFrame({"AGE":[20,30,40,50],"PEER_PRESSURE":[1,2,4,3],"LUNG_CANCER":["Yes","No","YES","No"],"GENDER":["Male","Female","Male","Female"],"INCOME":[10,40,20,30]})
schema=profile(df)
def run(q): return execute(df,plan(q,schema))
assert run("How many patients are in the dataset?")["kpis"][0][1] == 4
assert run("What percentage of patients have lung cancer?")["kpis"][2][1] == "50.00%"
assert run("How many patients have lung cancer?")["kpis"][0][1] == 2
assert run("What is the average peer pressure?")["kpis"][0][1] == "2.50"
assert run("Compare peer pressure by lung cancer")["chart"]["values"] == [2.5,2.5]
assert run("Show the number of patients by gender")["chart"]["values"] == [2,2]
assert run("What percentage of patients are male?")["kpis"][2][1] == "50.00%"
assert run("What is the average age?")["kpis"][0][1] == "35.00"

conditional=pd.DataFrame({"SMOKING":["Yes","Yes","Yes","No","No"],"LUNG_CANCER":["Yes","No","Yes","No","Yes"]})
conditional_schema=profile(conditional)
def conditional_run(q): return execute(conditional,plan(q,conditional_schema))
assert conditional_run("What percentage of smokers have lung cancer?")["kpis"][2][1] == "66.67%"
assert conditional_run("How many smokers have lung cancer?")["kpis"][0][1] == 2
assert conditional_run("What percentage of all patients both smoke and have lung cancer?")["kpis"][2][1] == "40.00%"
try:
    conditional_run("What percentage have lung cancer due to smoking?")
    raise AssertionError("Causal wording should ask for clarification")
except ValueError as error:
    assert "cannot establish causation" in str(error)
print("All deterministic and conditional-analysis tests passed.")
