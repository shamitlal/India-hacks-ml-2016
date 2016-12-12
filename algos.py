import numpy as np
import pandas as pd
import pylab as p
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model, decomposition, datasets
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import xgboost as xgb
# dont delete error count , level_numeric, rating


def randomforest():  # 4   best for n=200 ,max_depth=6 data points=4 lac   0.773
    #predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv", "wb")
    predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/csv/test10.csv", "wb")

    data_train_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_attributes.csv")
    #data_train_y = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_labels.csv")
    
    #data_train_y_numpy=data_train_y.values
    data_test_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/test_attributes.csv")
    data_train_x = data_train_x.drop(['user_accuracy','new_accuracy','rating'],axis=1)
    data_test_x = data_test_x.drop(['user_accuracy','new_accuracy','rating'],axis=1)
    data_test_x_numpy = data_test_x.values
    data_train_x_numpy=data_train_x.values
    test_file_csv = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")
    ids = test_file_csv.Id.values


    forest = RandomForestClassifier(n_estimators=1 , verbose =1 , max_depth=5)
    forest = forest.fit( data_train_x_numpy[0::,1::], data_train_x_numpy[0::,0])
    print "\n\n predicting .....\n\n"
    output = forest.predict(data_test_x_numpy).astype(int)

    
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","solved_status"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    
    #return output




def gradientboostingtrees():   # 1  # works best at n_estimators=20 nd max_depth = 3  acc = 0.767
    #predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv", "wb")
    predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/csv/test15.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","solved_status"])

    data_train_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_attributes.csv")
    #data_train_y = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_labels.csv")
    data_train_x_numpy=data_train_x.values
    #data_train_y_numpy=data_train_y.values
    data_test_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/test_attributes.csv")
    data_test_x_numpy = data_test_x.values
    test_file_csv = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")
    ids = test_file_csv.Id.values

    clf = GradientBoostingClassifier(n_estimators=25 , max_depth=3 ,verbose=1)
    clf = clf.fit(data_train_x_numpy[200000::,1::], data_train_x_numpy[200000::,0])
    print "\n\n predicting ... \n\n"
    output = clf.predict(data_test_x_numpy).astype(int) 
    
    
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()    

    #return output


def adaboost():   # 2  works best for n=25    acc = 0.755 data=100000
    predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","solved_status"])

    data_train_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_attributes.csv")
    #data_train_y = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_labels.csv")
    data_train_x_numpy=data_train_x.values
    #data_train_y_numpy=data_train_y.values
    data_test_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/test_attributes.csv")
    data_test_x_numpy = data_test_x.values
    test_file_csv = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")
    ids = test_file_csv.Id.values

    clf = AdaBoostClassifier(n_estimators=5)
    clf = clf.fit(data_train_x_numpy[0:100000,1::] , data_train_x_numpy[0:100000,0])
    print "\n\n predicting ... \n\n"
    output = clf.predict(data_test_x_numpy).astype(int) 
    
    
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()    

    #return output



def randomforest_adaboost_gradientboosting_ensemble():   # 2
    op = adaboost()
    print 'adaboost done\n'
    op1 = gradientboostingtrees()
    print 'gradient boosting done\n'
    op2 = randomforest()
    print 'random forest done\n'

    test_file_csv = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")
    ids = test_file_csv.Id.values
    predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","solved_status"])

    for i in range(len(op)):
        su=op[i]+op1[i]+op2[i]
        if su>=2:
            output=1
        else:
            output=0
        open_file_object.writerow([ids[i], output])        
    predictions_file.close()

    return output


def supportvector():
    predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv", "wb")
    data_train_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_attributes.csv")
    data_train_y = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_labels.csv")
    data_train_x_numpy=data_train_x.values
    data_train_y_numpy=data_train_y.values
    data_test_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/test_attributes.csv")
    data_test_x_numpy = data_test_x.values
    test_file_csv = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")
    ids = test_file_csv.Id.values

    clf = svm.SVC()
    clf = clf.fit(data_train_x_numpy , data_train_y_numpy[0::,0])
    print "\n\n predicting ... \n\n"
    output = clf.predict(data_test_x_numpy).astype(int) 
    
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","solved_status"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()

    return output
    


def xgboosting():  # gives best performance at n_estimator=25 , learning_rate=0.05 , max_depth = 3  acc=0.784 data=10000

    predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","solved_status"])
    data_train_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_attributes.csv")
   
    data_test_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/test_attributes.csv")
    

    data_test_x_numpy = data_test_x.values
    data_train_x_numpy=data_train_x.values
    test_file_csv = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")
    ids = test_file_csv.Id.values  #35617

    print "\n\n predicting ... \n\n"
    #, learning_rate=0.05
    gbm = xgb.XGBClassifier(max_depth=2, n_estimators=2000 ).fit(data_train_x_numpy[0::,1::], data_train_x_numpy[0::,0])
    output = gbm.predict(data_test_x_numpy).astype(int)
    
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()    



def logisticregressionclassifier():
    predictions_file = open("/Users/shamitlal/Desktop/india hacks ml/test_labels.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","solved_status"])

    data_train_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_attributes.csv")
    data_train_y = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/training_labels.csv")
    data_train_x_numpy=data_train_x.values
    data_train_y_numpy=data_train_y.values
    data_test_x = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/test_attributes.csv")
    data_test_x_numpy = data_test_x.values
    test_file_csv = pd.read_csv("/Users/shamitlal/Desktop/india hacks ml/will_bill_solve_it/test/test.csv")
    ids = test_file_csv.Id.values
    
    clf = linear_model.LogisticRegression(verbose = 1 , C = 0.001 , solver = 'newton-cg' , penalty = 'l2')
    clf.fit(data_train_x_numpy, data_train_y_numpy[0::,0])

    print "\n\n predicting ... \n\n"
    output = clf.predict(data_test_x_numpy).astype(int) 
    
    
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()


randomforest()

