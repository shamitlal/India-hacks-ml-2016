import numpy as np
import pandas as pd
import pylab as p
import csv as csv


df=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/train/submissions.csv")

class baz():
    def __init__(self, ac, wa):
        self.ac = ac
        self.wa = wa


problem_hash=list()
tup=tuple()
userdictac=dict()
userdictwa=dict()

for i in range(1444699):
    problem_hash.append(baz(0,0))
    

for i in range(len(df.index)):
    tup=(df.user_id[i],df.problem_id[i])
    if df.result[i]=='AC':        
        if tup in userdictac:
            continue
        elif tup in userdictwa:
            problem_hash[df.problem_id[i]].wa-=1
            userdictac[tup]=1
        else:
            problem_hash[df.problem_id[i]].ac+=1
            userdictac[tup]=1
    else:
        if tup in userdictac:
            continue
        elif tup in userdictwa:
            continue
        else:
            problem_hash[df.problem_id[i]].wa+=1
            userdictwa[tup]=1
    
df=pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/train/problems.csv")

for i in range(len(df.index)):
    print 'i=',i+1, ' solved_count=',df.solved_count[i],' accuracy=',df.accuracy[i],'new_accuracy=',df.solved_count[i]*100/(df.solved_count[i]+df.error_count[i]),\
      ' ac=',problem_hash[df.problem_id[i]].ac,' wa=',problem_hash[df.problem_id[i]].wa







