import numpy as np
import pandas as pd
import pylab as p
import csv as csv

predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/csv/test16.csv", "wb")

train_submissions=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")


data_train_x=list()
data_train_x= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(1,16):
    data_train_x[i] = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/csv/test"+(str)(i)+".csv")
    print data_train_x[i].info()

open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","solved_status"])
for i in range(len(train_submissions.index)):
    sum=0
    for j in range(1,16):
        sum+=data_train_x[j].solved_status[i]
    if sum>=7:
        open_file_object.writerow([i,"1"])
    else:
        open_file_object.writerow([i,"0"])



predictions_file.close()