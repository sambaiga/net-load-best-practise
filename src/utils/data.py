
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from utils.data_processing import AbsoluteScaler, fourier_series_t, compute_netload_ghi
from utils.load_data import load_uk_data, load_substation_data
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pytorch_lightning as pl

class DatasetObjective(object):


    def __init__(self, hparams, 
                window=slice('2019-01-01', '2019-12-31'), 
                data=None, dataset='netload', add_ghi=False,  scaler=MinMaxScaler(), target_scaler= AbsoluteScaler(), fit_scaler=True, exog_periods=None):

        
        self.hparams = hparams
        print(f'Load {dataset} dataset')
        
        if data is None:
            if dataset=='pt_dataset':
                self.data = load_substation_data()   
                    
            elif dataset=='uk_dataset':
                self.data = load_uk_data()
                
            else:
                raise AssertionError(f"{dataset} Dataset name not define")
                     

        else:
            self.data = data.copy()
         
        
        self.scaler = scaler
        self.target_transformer = target_scaler
        self.target = self.data[hparams['targets']].values
        self.installed_capacity=np.abs(self.target).max(0)
       
        

        if fit_scaler:
            self.target_transformer.fit(self.target)
        self.target = self.target_transformer.transform(self.target)
        self.data[[f+"_target" for f in hparams['targets']]] = self.target
       
        self.data = self.data.dropna()
        self.data = self.data.sort_index()
        
        
            
        
        self.numerical_features = self.hparams['time_varying_unknown_feature'] + self.hparams['time_varying_known_feature']
        if fit_scaler:
            self.scaler.fit_transform(self.data[self.numerical_features])
        if self.numerical_features:
            self.data[self.numerical_features] = self.scaler.transform(self.data[self.numerical_features])
        columns=hparams['targets']+hparams['time_varying_known_feature']+hparams['time_varying_unknown_feature']+hparams['time_varying_known_categorical_feature']+[f+"_target" for f in hparams['targets']]
        
        
        self.data[hparams['time_varying_known_categorical_feature']] = self.data[hparams['time_varying_known_categorical_feature']].astype(np.float32).values
        if self.data.index.name!='timestamp':
            self.data.index.name='timestamp'
            
       
        exog = self.data[hparams['time_varying_known_categorical_feature']].values
        self.data=self.data[columns]
        if exog_periods is None:
            exog_periods=[len(np.unique(exog[:, l])) for l in range(exog.shape[-1])]
        self.exog_periods = exog_periods
        seasonalities =np.hstack([fourier_series_t(exog[:,i], exog_periods[i], 1) for i in range(len(exog_periods))])
    
        
        self.seasonalities=pd.DataFrame(seasonalities)
        self.seasonalities.index = self.data.index
        
        self.seasonality_columns= [f"{i+1}" for i in range(seasonalities.shape[-1])]
        for i, column in enumerate(self.seasonality_columns):
            self.data[column]=seasonalities[:, i]
            
        
   
    def set_index_to_start_at_midnight(self, data, hour_init_prediction, hparams):
        data=data.sort_index()
        dummy_steps = hparams['SAMPLES_PER_DAY'] - (hour_init_prediction + 1)
        steps = dummy_steps + hparams['SAMPLES_PER_DAY']
        for datetime in data.index[data.index.hour == hour_init_prediction]:
            if len(data[:datetime]) >= hparams['window_size']:
                datetime_init_backtest = datetime
                #print(f"Backtesting starts at day: {datetime_init_backtest}")
                break

        days_backtest = np.unique(data[datetime_init_backtest:].index.date)
        days_backtest = pd.to_datetime(days_backtest)
        days_backtest = days_backtest[1:]
        numerical_columns=[f+"_target" for f in hparams['targets']]+hparams['time_varying_unknown_feature']
        #print(f"Days predicted in the backtesting: {days_backtest.strftime('%Y-%m-%d').values}")
        print('')
        start_window = (days_backtest[0] - pd.Timedelta(int(hparams['window_size']//hparams['horizon']), unit='day')).replace(hour=hour_init_prediction)
        end_window  = days_backtest[-1] - pd.Timedelta(30, unit='minutes')
        return data.loc[start_window:end_window]
    
    def get_data(self,  window= None, data=None):
        
        if data is None and window is not None:
            data = self.data.loc[window]
        elif data is None and window is None:
            data = self.data
        data = data.set_index('timestamp')
        #data = self.set_index_to_start_at_midnight(data, 0, self.hparams)

        numerical_columns=[f+"_target" for f in self.hparams['targets']]+\
                               self.hparams['time_varying_unknown_feature']
       
           
               
        unkown_features = data[numerical_columns].values.astype(np.float64) 
        target = data[[f+"_target" for f in self.hparams['targets']]].values.astype(np.float64)

        seasonalities=data[self.seasonality_columns].values
        if len(self.hparams['time_varying_known_feature'])>=1:
            known_features = np.concatenate([data[self.hparams['time_varying_known_feature']].values, seasonalities], 1).astype(np.float64)
        else:
            known_features = seasonalities.astype(np.float64) 
        unkown_features = np.concatenate([unkown_features, known_features], 1).astype(np.float64)
        
        return target, known_features, unkown_features
    
    

class TimeSeriesDataset(object):   
    def __init__(self, unknown_features, kown_features, targets, window_size=96, horizon=48, batch_size=64, shuffle=False, test=False, drop_last=True):
        self.inputs = unknown_features
        self.covariates = kown_features
        self.targets = targets
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.test = test
        self.drop_last= drop_last
        
    def frame_series(self):
        
        nb_obs, nb_features = self.inputs.shape
        features, targets, covariates = [], [], []
        

        list_range = range(0, nb_obs - self.window_size - self.horizon+1, self.horizon) if self.test else range(0, nb_obs - self.window_size - self.horizon+1)
        with tqdm(len(list_range)) as pbar:
            for i in list_range:
                features.append(torch.FloatTensor(self.inputs[i:i + self.window_size, :]).unsqueeze(0))
                targets.append(
                        torch.FloatTensor(self.targets[i + self.window_size:i + self.window_size + self.horizon]).unsqueeze(0))
                covariates.append(
                        torch.FloatTensor(self.covariates[i + self.window_size:i + self.window_size + self.horizon,:]).unsqueeze(0))

                pbar.set_description('processed: %d' % (1 + i))
                pbar.update(1)
            pbar.close() 

        features = torch.cat(features)
        targets, covariates = torch.cat(targets), torch.cat(covariates)
        
        
        
        #padd covariate features with zero
        diff = features.shape[2] - covariates.shape[2]
        B, N, _ = covariates.shape
        diff = torch.zeros(B, N, diff, requires_grad=False)
        covariates = torch.cat([diff, covariates], dim=-1)
        features = torch.cat([features, covariates], dim=1)
        
        del covariates
        del diff
        
       

        return TensorDataset(features,  targets)
        
    def get_loader(self):
        dataset = self.frame_series()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last, num_workers=8, pin_memory=True)
        return loader


class TimeSeriesLazyDataset(Dataset):   
    def __init__(self, unknown_features, kown_features, targets, window_size=96, horizon=48):
        self.inputs = torch.FloatTensor(unknown_features)
        self.covariates = torch.FloatTensor(kown_features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size
        self.horizon = horizon
        
        
    def __len__(self):
        return self.inputs.shape[0]-self.window_size-self.horizon
    
    def __getitem__(self, index):
        features = self.inputs[index:index + self.window_size]
        target = self.targets[index+self.window_size:index+self.window_size+self.horizon]
        covariates = self.covariates[index+self.window_size:index+self.window_size+self.horizon]
        
        #padd covariate features with zero
        diff = features.shape[1] - covariates.shape[1]
        N, _ = covariates.shape
        diff = torch.zeros(N, diff, requires_grad=False)
        covariates = torch.cat([diff, covariates], dim=-1)
        features = torch.cat([features, covariates], dim=0)
        del covariates
        del diff
        return features, target
            

class TimeseriesDataModule(pl.LightningDataModule):
    def __init__(self, hparams, experiment, train_df, test_df, test=False):
        super().__init__()
       
        target, known_features, unkown_features = experiment.get_data(data=train_df)
        if test:
            self.train_dataset = TimeSeriesDataset(unkown_features, known_features, target, window_size=hparams['window_size'], horizon=hparams['horizon'],
                                                    batch_size=hparams['batch_size'], shuffle=True, test=False, drop_last=True)
        else:
            self.train_dataset = TimeSeriesLazyDataset(unkown_features, known_features, target, window_size=hparams['window_size'], horizon=hparams['horizon'])
      

        target, known_features, unkown_features = experiment.get_data(data=test_df)
        
        if test:
            self.test_dataset = TimeSeriesDataset(unkown_features, known_features, target, window_size=hparams['window_size'], horizon=hparams['horizon'],
                                                    batch_size=hparams['batch_size'], shuffle=False, test=False, drop_last=False)
        else:
            self.test_dataset = TimeSeriesLazyDataset(unkown_features, known_features, target, window_size=hparams['window_size'], horizon=hparams['horizon'])
        del target
        del known_features
        del unkown_features
        self.batch_size=hparams['batch_size']
        self.test = test

    def train_dataloader(self):
        if self.test:
            return self.train_dataset.get_loader()
        else:
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        if self.test:
            return self.test_dataset.get_loader()
        else:
            return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)