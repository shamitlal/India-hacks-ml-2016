
import numpy as np
import pandas as pd
import pylab as p
import csv as csv







# prerpocessing problems.csv starts




df=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv")
cnt=0
for i in range(len(df.index)):
	if df.solved_status[i]==1:
		cnt+=1
print "cnt:",cnt
'''
test_file_csv = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")
ids = test_file_csv.Id.values
predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow([ids[0] , "0"])
'''