'''this python file will be responsible for
    applying all the steps of the preprocessing logic
    onto the dataset

'''

#defining the constants


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




from preprocessing_logic import process_data




class transform:
    def __init__(self):
        pass

    def transform_data(self,xtrain,xtest,ytrain,ytest):
        self.xtrain=xtrain
        self.xtest=xtest
        self.ytrain=ytrain
        self.ytest=ytest

        '''Creating object of the class 
            to use it's methods to apply it onto dataset
        '''
        process=process_data()



        #imputing the missing values

        xtrain_imp=process.imputer(self.xtrain)


        xtest_imp=process.imputer(self.xtest)

        #scaling the data using the scaler method

        xtrain_scale= process.scaler(xtrain_imp,categorical_columns=categorical_columns)
        xtest_scale=process.scaler(xtest_imp,categorical_columns=categorical_columns)

        print(xtrain_scale.isnull().sum())
        xtrain_pca=process.pca(xtrain_scale,pca_components=pca_components,non_cat_columns=non_categorical_columns,cat_cols=categorical_columns)
        xtest_pca=process.pca(xtest_scale,pca_components=pca_components,non_cat_columns=non_categorical_columns,cat_cols=categorical_columns)

        #balancing the training dataset

        print(xtrain_pca.isnull().sum())
        xtrain_bal,ytrain_bal=process.balancing(xtrain_pca,self.ytrain)


        return xtrain_bal,xtest_pca,ytrain_bal,self.ytest
