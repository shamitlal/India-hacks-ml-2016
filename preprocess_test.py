
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pylab as p
import csv as csv







# prerpocessing problems.csv starts

df=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/problems.csv")


for i in range(1,6):
    temp='tag'+(str)(i)
    del df[temp]

df['level_numeric']=df['solved_count']

for i in range(len(df.index)):
    if df.level[i]=='V-E':
        df.level_numeric[i]=0
    elif df.level[i]=='E':
        df.level_numeric[i]=1
    elif df.level[i]=='E-M':
        df.level_numeric[i]=2
    elif df.level[i]=='M':
        df.level_numeric[i]=3
    elif df.level[i]=='M-H':
        df.level_numeric[i]=4
    elif df.level[i]=='H':
        df.level_numeric[i]=5
    else:
        df.level_numeric[i]=6


median_solved_for_level=np.zeros(7)
for i in range(6):
    median_solved_for_level[i]=df[df['level_numeric']==i]['solved_count'].dropna().median()

# fill missing values for level for each row
for i in range(len(df.index)):
    if df.level_numeric[i]==6:
        temp=df.solved_count[i]
        mini=10000    # to find minimum distance of solved_cnt with median values
        select=0
        for j in range(6):
            compute_dist=abs(temp-median_solved_for_level[j])
            if compute_dist<mini:
                mini=compute_dist
                select=j

        df.level_numeric[i]=select


#df['solved-error']=df['solved_count']-df['error_count']

df['new_accuracy']=(df['solved_count']+1)/(df['solved_count']+df['error_count']+1) # created new feature for accuracy
df['accuracy']=(df.accuracy*100)
df.new_accuracy=(df.new_accuracy*100)
df.rating=df.rating*10
df.level_numeric=df.level_numeric*100

#print df.info()
#preprocessing problems.csv ends
















#preprocessing of users.csv starts

train_user=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/users.csv")
del train_user['skills']
train_user['user_accuracy']=(train_user['solved_count']+1)/(train_user['solved_count']+train_user['attempts']+1) #created new feature
train_user.user_accuracy=(train_user.user_accuracy*100)

train_user['solved_count_user'] = train_user['solved_count']
del train_user['solved_count']
del train_user['user_type']

#preprocessing of users.csv ends




















#preprocessing of test.csv starts

train_submissions=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")

#preprocessing of test.csv ends



final1 = pd.merge(df, train_submissions, on='problem_id', how='inner')

final = pd.merge(final1, train_user, on='user_id', how='inner')




labels=np.zeros(2000000)  # for prediction labels
print final.info()

print "hello"







# making csv file storing attributes corresponding to user_id and problem_id in csv file
file_location="/Users/shamitlal/Desktop/india hacks ml/test_attributes.csv"
predictions_file = open(file_location, "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["accuracy", "solved_count", "error_count" , "rating", "level_numeric" , "new_accuracy" , "solved_count_user" , "attempts" , "user_accuracy" ])


for i in range(len(final.index)): 
    open_file_object.writerow([ final.accuracy[i] , final.solved_count[i] ,final.error_count[i] , final.rating[i] , 
        final.level_numeric[i] , final.new_accuracy[i] , final.solved_count_user[i] ,  final.attempts[i] ,final.user_accuracy[i]])

predictions_file.close()




