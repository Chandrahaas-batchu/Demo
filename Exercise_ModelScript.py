# -*- coding: utf-8 -*-
"""
Created on Jun 4 12:43:34 2019

@author: Chandrahaas batchu
"""
#Importing the required libraries
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging
import dill as pickle

#Function to read the training data from CSV file
def load_data(filepath):
    df = pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
    except IOError:
        logging.exception('')
    if df.empty:
        raise ValueError('No data available')
    return df

# Function to clean and pre-process the training dataframe
def preprocessing(df, remove_outliers=True):
    #Replacing null values in categorical features with mode values of the respective features
    for column in ['yr', 'season', 'mnth', 'hr', 'holiday','weekday','workingday','weathersit']:
        df[column].fillna(df[column].mode()[0], inplace=True)

    #Replacing null values in numeric features with mean values of the respective features
    for column in ['temp','atemp','hum','windspeed']:
        df[column].fillna(df[column].mean(), inplace=True)
        
    # Converting the categorical features' datatypes to string datatype, to create dummy variables
    convert_dict = {'season':str,
                   'mnth':str,
                   'hr':str,
                   'weekday':str,
                   'weathersit':str}
    df = df.astype(convert_dict)
    
    # Creating dummy variables for all categorical features
    df = pd.get_dummies(df, columns=['season','mnth','hr','weekday','weathersit'],drop_first=True)
    
    # Removing the feature "instant" since it is not useful
    df.drop(columns='instant',inplace=True, axis=1)
    
# Using LinearRegression to impute zero 'windspeed' values
    # Separating records having '0' windspeed and non-zero windspeeds
    df_wind0 = df[df["windspeed"]==0]
    df_windNot0 = df[df["windspeed"]!=0]
        
    # Creating a list of feature names that might influence the 'windspeed'
    windColumns = ['yr','temp','atemp','hum','season_2', 'season_3',
           'season_4', 'mnth_10', 'mnth_11', 'mnth_12', 'mnth_2', 'mnth_3',
           'mnth_4', 'mnth_5', 'mnth_6', 'mnth_7', 'mnth_8', 'mnth_9', 'hr_1',
           'hr_10', 'hr_11', 'hr_12', 'hr_13', 'hr_14', 'hr_15', 'hr_16',
           'hr_17', 'hr_18', 'hr_19', 'hr_2', 'hr_20', 'hr_21', 'hr_22',
           'hr_23', 'hr_3', 'hr_4', 'hr_5', 'hr_6', 'hr_7', 'hr_8', 'hr_9','weathersit_2', 'weathersit_3', 'weathersit_4']
    
    #Instantiating LinearRegression model
    linreg = LinearRegression()
    # Training the linear regression model on records with non-zero windspeed values
    linreg.fit(df_windNot0[windColumns], df_windNot0["windspeed"])
    
    #Predicting the windspeed values for records originally with '0'
    wind_0Values = linreg.predict(X= df_wind0[windColumns])
    df_wind0["windspeed"] = wind_0Values
    df = df_windNot0.append(df_wind0)
    df.reset_index(inplace=True)
    df.drop('index',inplace=True,axis=1)
    
    if(remove_outliers):
    # Removing outliers from 'cnt' feature
        # Calculating inter-quartile range
        iqr = df['cnt'].quantile(0.75) - df['cnt'].quantile(0.25)
        
        # Calculating the value of 'cnt' at upper-whisker in the box-plot
        outlier_UL = df['cnt'].quantile(0.75) + (iqr)*1.5
            
        # Creating a new dataset without the outliers
        df = df[df['cnt'] <= outlier_UL]

# Separating features (x), dependent variable (y)
    x = df.drop(columns=['cnt','dteday','casual','registered','atemp'],axis=1)
    y = df['cnt']
    #x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=7)
    
# Standardizing the numeric features
    numeric_features = ['temp','hum','windspeed']
    x_std = x.copy()
    scaler = StandardScaler()
    x_std[numeric_features] = scaler.fit_transform(x_std[numeric_features])

# Applying natural logarithm on 1 + cnt variable values
    y_log = np.log1p(y)
    
    return x_std, y_log;

# Function to train and build a XGBoost model
def train_and_build(x_std,y_log, grid_search_boolean=False):
    
    if(grid_search_boolean):
        gbm = GradientBoostingRegressor()
        parameters={'n_estimators':[100, 200, 2000, 4000], 
            'alpha': [0.05, 0.02, 0.01]
           }
        MAE_scorer = metrics.make_scorer(mean_absolute_error, greater_is_better=False)
        
        grid_gbm = GridSearchCV(estimator=gbm, cv=5, param_grid=param_grid, n_jobs=n_jobs, scoring=MAE_scorer)
        grid_gbm.fit(x_std,y_log)
        
        gbm_best = grid_gbm.best_estimator_
        print ("Best MAE:", gbm_best.best_score_)
        print ("Best hyper-parameters:", gbm_best.best_params_)
    else:
        #Instantiating a GradientBoosting Regressor model with hyperparameters selected using GridSearch & Cross-validation
        gbm_best = GradientBoostingRegressor(n_estimators=4000,alpha=0.01)
        
        #Cross-Validation scores
        scores = cross_val_score(gbm_best, x_std,y_log, scoring='neg_mean_absolute_error', cv=5)
        
        print("The cross-validation scores(MAE) are:", scores)
        print("The mean cross-validation score(MAE) is:", str(scores.mean()))
        
        #Training the model on entire training dataset
        gbm_best.fit(x_std,y_log)
    
    return gbm_best;

if __name__ == '__main__':
    
    def train():
        # Calling the load_data function to read the CSV file
        df = load_data('train.csv')
        
        # Calling the preprocessing function to clean and transofmr the data
        x_std, y_log = preprocessing(df)
        
        # Training a model on the cleaned data
        model = train_and_build(x_std,y_log)
    
        filename = 'model_v1.pk'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
        print("Trained and save the model as 'model_v1.pk'")
    
    def test_and_predict():
        # Calling the load_data function to read the test CSV file
        df = load_data('test.csv')
        
        # Calling the preprocessing function to clean and transofmr the data
        x_std, y_log = preprocessing(df)#, remove_outliers=False)
        
        # Loading the model that was saved after training on train data
        with open('model_v1.pk' ,'rb') as f:
            loaded_model = pickle.load(f)
            
        print(df.shape)
        print(x_std.shape)
            
        preds_gbm = loaded_model.predict(x_std)
        print ("Test MAE Value For GradientBoost Regression: ",mean_absolute_error(np.exp(y_log),np.exp(preds_gbm)))
        predictions = pd.DataFrame(np.expm1(preds_gbm), columns=['predicted_cnt']).reset_index(drop=True)
        x_std_noIndex = x_std.reset_index(drop=True)
        y_log_noIndex = pd.DataFrame(y_log,columns=['cnt']).reset_index(drop=True)
        test_predictions = pd.concat([x_std_noIndex, np.expm1(y_log_noIndex), predictions], axis=1)
        return test_predictions
    
    # Training the model again
    #train()
    
    #Predicting on the test data
    predictions = test_and_predict()
    predictions.to_csv('Predictions_Test.csv',index=False)
    
    ###---- End of FIle ------###
    
        