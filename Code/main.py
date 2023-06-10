import pandas as pd

import sys
import shutil
import os

from preprocessing import preprocessing
from Code.build_models import method_SSL
from introducing_leakages import introducing_leakages_function

import time



def array_P(aux):
    if aux == "0.5":
        return [0.5]

    elif aux == "0.6":
        return [0.6]

    elif aux == "0.7":
        return [0.7]

    elif aux == "0.8":
        return [0.8]

    elif aux == "0.9":
        return [0.9]

    elif aux == "1.0":
        return [1.0]

    elif aux == "All":
        return [0.5,0.6,0.7,0.8,0.9,1.0]


P = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#por o factory_salary e body_perform

#Save file in folder
def save_file_folder(file_path):
    try:
        # Copy the file to the target folder
        folder_path = os.getcwd()
        dest_path = os.path.join(folder_path, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        print("Successfully!")
        return dest_path
    except FileNotFoundError:
        print("File doesn't exist.")

#Read file ".csv" with delimiter: ","
def read_csv(file_path):

    data = pd.read_csv(file_path, on_bad_lines='skip')

    target = data.columns[-1]

    return data, target

if __name__ == "__main__":
    if len(sys.argv) > 2:
        file_path = sys.argv[1]
        classf = int(sys.argv[2])
        dest_path = save_file_folder(file_path)
        data, target = read_csv(file_path)
        os.system("clear")
        print("What do you intend to do?")
        print("1) Check for leakage")
        print("2) Introducing leakage and check behaviour")
        val = int(input())
        if val == 1 or val == 2:
            if val == 1:
                print("Option 1 chosen")
                os.system("clear")
                data_aux = preprocessing(data,target, classf)
                method_SSL(data_aux,target,classf)
            else:
                print("Option 2 chosen")
                os.system("clear")
                attrs_categoric = [x for x in list(data.columns) if data[x].dtype == object and x != target]
                print("Add: ")
                print("0.5")
                print("0.6")
                print("0.7")
                print("0.8")
                print("0.9")
                print("1.0")
                print("All")
                aux = input("Write: ")
                os.system("clear")

                if aux == "0.5" or aux == "0.6" or aux == "0.7" or aux == "0.8" or aux == "0.9" or aux == "1.0" or aux == "All":
                    P = array_P(aux)
                    dict_p  = {}
                    overfitting = {}
                    for k in P:
                        dict_p[k] = 0

                    for c in data.columns:
                        if c == target:
                            continue

                        print("----------- Column: " + c + "   -------------")
                        data_leakage = data.copy()
                        for p in P:
                            print("----------- Percentage of leakage : " + str(p) + "   -------------")
                            data_leakage = introducing_leakages_function(data,c,target,p,attrs_categoric, classf)
                            print(data_leakage)
                            data_leakage_aux = preprocessing(data_leakage,target,classf)
                            ssl, val = method_SSL(data_leakage_aux,target,classf)

                            dict_p[p] += ssl
                            overfitting[c] = val
                            time.sleep(0.5)

                    for key in dict_p.keys():
                        print("With " + str(key) + " the leakage percentage entered, we have " + str(dict_p[key]) + "/" + str(len(data.columns[:-1])) +" variables that the method can detect with leakage")

                    print("-----------------------------------------------------------------------------------")

                    for key in overfitting.keys():
                        if key == "overfitting":
                            continue
                        else:
                            print("In column: " + key + "Model 2 and Model 3 both achieved score = 1.0")
                else:
                    print("None of the options are valid!")

        else:
            print("None of the options are valid!")


    elif len(sys.argv) == 2:
        print("No task number provided.")
    else:
        print("No file path provided.")
