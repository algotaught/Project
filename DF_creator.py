'''this file will be used to load the data'''

import pandas as pd
import os
from sklearn.model_selection import train_test_split
'''this function will take 2 parameters: 1) File path and 2) Target column
    it will then return 4 outputs which will be xtrain,xtest,ytrain,ytest
 '''


def df(file_path,target_column,cat_cols,mapping_dict,drop_columns):

    #loading csv

    csv=pd.read_csv(file_path)
    print(csv.isnull().sum())
    target_column=target_column
    map_cols=cat_cols+[target_column]
    for x in map_cols:
        csv[x]=csv[x].map(mapping_dict)
    print(type(csv))
    csv.drop(drop_columns,axis=1,inplace=True)
    print(type(csv))


    #defining x and y column
    x=csv.drop(target_column,axis=1)
    y=csv.loc[:,target_column].values.reshape(-1,1)


    #x=x.drop(drop_columns,axis=1,inplace=True)
    #doing the train and test split using the sklearn library

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
    print(xtrain,xtest,ytrain,ytest)

    #defining the return values that this function will give

    return xtrain,xtest,ytrain,ytest