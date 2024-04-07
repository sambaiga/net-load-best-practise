import numpy as np
import pandas as pd


def clean_data(data):
    #Drop meters with more than 50% missing values
    data = data.dropna(axis=1, thresh=int(0.9*len(data)))
    #fill missing values
    data = data.fillna(method='ffill')
    data  = data.fillna(value=data.median()) 
    return data  


def prepare_data(data):
    df = clean_data(data)
    df_hourly = df.resample("H").sum()
    df_daily  = df.resample("D").sum()
    df_weekly  = df.resample("W").sum()
    df_monthly  = df.resample("M").sum()
    
    return {"data":df, 
            "hourly":df_hourly, 
            "daily":df_daily, 
            "weekly":df_weekly, 
            "monthly":df_monthly
            }
def prepare_feature(data, feature_type="diff", diff=True):
    
    if diff:
        diff_data = {}
        diff_data["data"] = data['data'].diff().fillna(0.0)
        diff_data["hourly"]= data['hourly'].diff().fillna(0.0)
        diff_data['daily']= data['daily'].diff().fillna(0.0)
        diff_data['weekly']= data['weekly'].diff().fillna(0.0)
        diff_data['monthly']= data['monthly'].diff().fillna(0.0)
    
    if feature_type=="diff" and diff == True:
        features = np.concatenate([diff_data["data"].values, 
                           diff_data["hourly"].values, 
                           diff_data['daily'].values,
                          diff_data['weekly'].values,
                          diff_data['monthly'].values], 0).T
    elif feature_type=="org": 
        features = np.concatenate([data['hourly'].values, 
                           data['daily'].values, 
                           data['weekly'].values,
                          data['monthly'].values,
                          data['data'].values], 0).T  
    
    elif feature_type=="combined": 
        features = np.concatenate([data['hourly'].values, 
                           data['daily'].values, 
                           data['weekly'].values,
                          data['monthly'].values,
                          data['data'].values], 0).T 
        
        features_dff = np.concatenate([diff_data["data"].values, 
                           diff_data["hourly"].values, 
                           diff_data['daily'].values,
                          diff_data['weekly'].values,
                          diff_data['monthly'].values], 0).T
        
        features = np.concatenate([features, features_dff, 1])  
    
    elif  feature_type in ["hourly", "daily", "weekly", "monthly", 'data']: 
        if diff==False:
            features =  data[feature_type].T.values
        else:
             features =  diff_data[feature_type].T.values    
    
    return features    
        