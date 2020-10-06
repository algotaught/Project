import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier




'''Linear Regression model building'''
class model_trainer:
    def __init__(self,xtrain,ytrain,xtest,ytest):
        self.xtrain=xtrain
        self.ytrain=ytrain
        self.xtest=xtest
        self.ytest=ytest
        pass

    def logisticModel(self):

        lr=LogisticRegression()
        lr.fit(self.xtrain,self.ytrain)

        prediction=lr.predict(self.xtest)
        Testscore=accuracy_score(self.ytest,prediction)
        report=classification_report(self.ytest, prediction)
        print(lr.score(self.xtrain,self.ytrain))
        print(Testscore)
        print(report)
        joblib.dump(lr, 'saved_model/lr_model.pkl')

    def decisionTree(self):
        dtc=DecisionTreeClassifier()
        dtc.fit(self.xtrain,self.ytrain)
        print(dtc.score(self.xtrain,self.ytrain))

        preds=dtc.predict(self.xtest)

        print(classification_report(self.ytest,preds))

        joblib.dump(dtc, 'saved_model/dtc_model.pkl')


    def xgb(self):
        xg=XGBClassifier()
        xg.fit(self.xtrain,self.ytrain)
        prediction=xg.predict(self.xtest)
        Testscore=accuracy_score(self.ytest,prediction)
        report=classification_report(self.ytest, prediction)
        print(xg.score(self.xtrain,self.ytrain))
        print(Testscore)
        print(report)
        joblib.dump(xg, 'saved_model/xg_model.pkl')

