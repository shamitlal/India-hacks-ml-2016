
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pylab as p
import csv as csv







# prerpocessing problems.csv starts

df=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/train/problems.csv")


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
for i in range(1002):
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




df['new_accuracy']=(df['solved_count']+1)/(df['solved_count']+df['error_count']+1) # created new feature for accuracy

df['accuracy']=(df.accuracy*100)
df.new_accuracy=(df.new_accuracy*100)
df.rating=df.rating*10
df.level_numeric=df.level_numeric*100
#print df.info()
#preprocessing problems.csv ends
















#preprocessing of users.csv starts

train_user=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/train/users.csv")
del train_user['skills']
train_user['user_accuracy']=(train_user['solved_count']+1)/(train_user['solved_count']+train_user['attempts']+1) #created new feature
train_user.user_accuracy=(train_user.user_accuracy*100)
#print train_user.info()
train_user['solved_count_user'] = train_user['solved_count']
del train_user['solved_count']

del train_user['user_type']


#preprocessing of users.csv ends



















#preprocessing of submissions.csv starts

train_submissions=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/train/submissions.csv")
train_submissions = train_submissions.drop(['solved_status', 'language_used', 'execution_time'], axis=1) 

#preprocessing of submissions.csv ends

final1 = pd.merge(df, train_submissions, on='problem_id', how='inner')

final = pd.merge(final1, train_user, on='user_id', how='inner')

labels=np.zeros(1444699)  # for prediction labels

print final.info()

print 'hello tuple starts\n\n'

# for cases having same user_id solving same problem_id consider only AC cases for all

tup=()
userdict = dict()
for i in range(1198131):
    tup=(train_submissions.user_id[i],train_submissions.problem_id[i])
    if tup not in userdict:
        userdict[tup]=i

    else:
        if train_submissions.result[userdict[tup]]!="AC":
            userdict[tup]=i










# making csv file storing attributes corresponding to user_id and problem_id in csv file
file_location="/Users/shamitlal/Desktop/india hacks ml/training_attributes.csv"
predictions_file = open(file_location, "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["labels", "accuracy", "solved_count", "error_count" , "rating", "level_numeric" , "new_accuracy" , "solved_count_user" , "attempts" , "user_accuracy" ])


for i in range(len(final.index)): #1198131
    tup=(final.user_id[i],final.problem_id[i])
    row=userdict[tup]
    if train_submissions.result[row]=='AC':
        labels[i]=int(1)
    else:
        labels[i]=int(0)
    open_file_object.writerow([labels[i], final.accuracy[i] , final.solved_count[i] ,final.error_count[i] , final.rating[i] , 
        final.level_numeric[i] , final.new_accuracy[i] , final.solved_count_user[i] ,  final.attempts[i] ,final.user_accuracy[i]])

predictions_file.close()


