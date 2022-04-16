import json
import pandas as pd
import os

def load__Patient_INFO():
    with open("patient_info_dict.json", "r+") as f:
        return json.load(f)

def load_csv(patient_info, mass, calc, patient_id):
    criterion = calc['patient_id'].map(lambda x: x=="P_"+patient_id)
    if(calc[criterion]["pathology"].shape[0] > 0):
        return calc[criterion]["pathology"].iloc[0]
    criterion = mass['patient_id'].map(lambda x: x=="P_"+patient_id)
    if(mass[criterion]["pathology"].shape[0] > 0):
        return mass[criterion]["pathology"].iloc[0]


if __name__ == "__main__":
    #generate_Patient_INFO()
    patient_info = load__Patient_INFO()
    mass = pd.read_csv("mass_case_description_train_set.csv")
    calc = pd.read_csv("calc_case_description_train_set.csv")
    for i in patient_info:
        pathology = load_csv(patient_info, mass, calc, i)
        patient_info[i]["pathology"] = pathology
    
    with open("patient_info_dict_.json", "w+") as f:
        json.dump(patient_info, f)