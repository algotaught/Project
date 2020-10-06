#from preprocessing import DataProcessor
from DF_creator import df



'''defining the parameters'''
target_col='went_on_backorder'
file_path='raw_csv/InputFile.csv'

categorical_columns=['potential_issue','deck_risk', 'oe_constraint',
                     'ppap_risk', 'stop_auto_buy', 'rev_stop',
                     ]
mapping_dictionary= {"Yes": 1, "No": 0}

columns_to_drop=['sku','Index_Product']
non_categorical_columns= ['national_inv',
                      'lead_time',
                      'in_transit_qty',
                      'forecast_3_month',
                      'forecast_6_month',
                      'forecast_9_month',
                      'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
                      'min_bank', 'pieces_past_due', 'perf_6_month_avg',
                      'perf_12_month_avg', 'local_bo_qty'
                          ]
pca_components=7



'''Loading the csv file in pandas dataframe format, and dividing the xtrain,xtest,ytrain,ytest'''

xtrain,xtest,ytrain,ytest=df(file_path=file_path,target_column=target_col,cat_cols=categorical_columns,mapping_dict=mapping_dictionary,drop_columns=columns_to_drop)

print('this is xtrain column\n',xtrain)

print('this is xtest column\n',xtest)

print('this is ytrain column\n',ytrain)

print('this is ytest column\n',ytest)



'''creating a log of above file'''
'''import  the logger class'''

'''from logger import log_creator

creating a object of log_creator so that we can use the methods that are written in it.

log_writer=log_creator.log('./lo')'''


'''importing the class preprocessing to do the neccesary transformations '''

from preprocessing import transform

'''Creating object of the class to use it's methods to apply it onto dataset'''
dataTransformer=transform()

#running the data through the pipeline
xtrain,xtest,ytrain,ytest= dataTransformer.transform_data(xtrain,xtest,ytrain,ytest)

print(xtrain,ytrain)


'''importing methods to train the ml model'''

from training_logic import model_trainer


model_trainer=model_trainer(xtrain,ytrain,xtest,ytest)


print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)
classifierModel=model_trainer.logisticModel()

xgboost=model_trainer.xgb()

decisiontree=model_trainer.decisionTree()




