from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

class process_data:
    def __init__(self):
        pass

    def imputer(self,df):
        self.df=df
        print(self.df.isnull().sum())
        imputer_obj=KNNImputer(n_neighbors=3, weights='uniform')
        imputed_df=imputer_obj.fit_transform(self.df)
        df=pd.DataFrame(imputed_df,columns=df.columns,index=df.index)

        return df

    def scaler(self,df,categorical_columns):
        self.df=df
        self.categorical_columns=categorical_columns
        ss=StandardScaler()
        df_cont= self.df.drop(self.categorical_columns,axis=1)
        print(type(df_cont))
        for x in df_cont.columns:
            self.df[x] = ss.fit_transform(df_cont[x].values.reshape(-1,1))
            pass

        return self.df

    def pca(self, df, pca_components, non_cat_columns, cat_cols):
        self.df=df
        self.pca_components=pca_components
        self.non_cat_columns=non_cat_columns
        self.cat_cols=cat_cols

        pca_obj=PCA(n_components=self.pca_components)
        print('pca df')
        print(df.shape)
        #pca_df=df.drop(cat_cols,axis=1)
        #print(type(pca_df))
        #print(pca_df)
        pca_columns= pca_obj.fit_transform((self.df.drop(self.cat_cols,axis=1)))
        print('pca done')
        names=[]

        for x in range(self.pca_components):
            name= 'pca'+str(x)
            names.append(name)

        pca_df=pd.DataFrame(pca_columns,columns=names)

        cat_df= df.drop(non_cat_columns,axis=1)
        cat_df=cat_df.reset_index(drop=True)
        pca_df=pca_df.reset_index(drop=True)
        pca_df= pd.concat([pca_df, cat_df], axis=1)

        print(pca_df.shape,pca_df)
        return pca_df

    def balancing(self,xtrain,ytrain):
        self.xtrain=xtrain
        self.ytrain=ytrain
        sm=SMOTETomek()

        balanced_xtrain, balanced_ytrain= sm.fit_resample(self.xtrain,self.ytrain)

        xtrain_df= pd.DataFrame(balanced_xtrain, columns=self.xtrain.columns)
        balanced_ytrain=balanced_ytrain.reshape(-1,1)
        return xtrain_df, balanced_ytrain




