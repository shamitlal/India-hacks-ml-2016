import numpy as np
import pandas as pd
import pylab as p
import csv as csv

df=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv")
problem_csv=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/problems.csv")
test_csv=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")


predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/meta_test_labels.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","solved_status"])

userdict=dict()
for i in range(len(problem_csv.index)):
    userdict[problem_csv.problem_id[i]]=(problem_csv.solved_count[i]*problem_csv.solved_count[i]*100)/(problem_csv.solved_count[i] + \
        problem_csv.error_count[i])

for i in range(len(df.index)):
    if userdict[test_csv.problem_id[i]]>=22000:
        open_file_object.writerow([i, "1"])
    else:
        open_file_object.writerow([i, df.solved_status[i]])

predictions_file.close()    


