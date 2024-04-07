import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
from darts import TimeSeries
import numpy as np
from utils.hparams import PT_params, UK_params, hparams
from utils.load_data import load_hybrid_res_data, load_uk_data, load_substation_data, load_albania_data
from utils.data import DatasetObjective
from baselines.backtesting import BacktestingForecast
from utils.data_processing import add_time_features
from utils.data_processing import get_periods_for_exog_variable, fit_scaling_on_train_df, get_index
import matplotlib
matplotlib.use('Agg')


def run_backtesting(forecast_len=6, 
                incremental_len = 1,
                min_train_len=12,
                n_splits=10,
                    seed=777,
                    max_epochs=200,
                    autotune=False,
                    experiment_type='benchmark',
                    window_type='expanding',
                    data_path='../dataset/',
                    dataset="UK_dataset",
                    exp_name='short_forecasting',
                    num_trials=50
                   ):

    """
        Run backtesting for time series forecasting with configurable parameters.

        Args:
            forecast_len (int, optional): Length of the forecast horizon. Defaults to 6.
            incremental_len (int, optional): Length of incremental steps. Defaults to 1.
            min_train_len (int, optional): Minimum length of the training dataset. Defaults to 12.
            seed (int, optional): Seed for reproducibility. Defaults to 777.
            max_epochs (int, optional): Maximum number of epochs for training. Defaults to 200.
            autotune (bool, optional): Whether to perform hyperparameter autotuning. Defaults to False.
            baseline (bool, optional): Whether to run baseline models. Defaults to False.
            window_type (str, optional): Type of window for cross-validation. Defaults to 'expanding'.
            data_path (str, optional): Path to the dataset. Defaults to '../dataset/'.
            dataset (str, optional): Name of the dataset. Defaults to 'UK_dataset'.
            exp_name (str, optional): Experiment name. Defaults to 'short_forecasting'.
            conformal_loss (bool, optional): Whether to use conformal loss. Defaults to True.
            conf_level (float, optional): Confidence level for conformal loss. Defaults to 0.9.
            num_trials (int, optional): Number of trials. Defaults to 50.
            conformalize (bool, optional): Whether to apply conformalization. Defaults to False.

        Returns:
            None
    """
    hparams.update({'max_epochs':max_epochs})
    hparams.update({'num_trials':num_trials}) 
    hparams.update({'latent_size': 256, 'depth': 2, 'batch_size': 64,  'dropout': 0.1786342451901633,
                'alpha': 0.1786342451901633, 'emb_size': 32, 'embed_type': 'RotaryEmb', 'comb_type':'weighted-comb',
                })
   
    
    if  dataset=='pt_dataset':
        data  = load_substation_data(add_ghi=True)['2018-09':'2022-03'].select_dtypes(exclude='object')
        data=data.reset_index().drop_duplicates(subset='timestamp')
        data=data.drop(columns=[ 'index'])

        columns = list(data.columns)[1:]
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True).dt.tz_localize(None)
        data=TimeSeries.from_dataframe(data, 'timestamp', columns, fill_missing_dates=True).pd_dataframe()
        hparams.update({'time_varying_known_categorical_feature': [ 'DAYOFWEEK', 'DAY', 'HOUR', 'Session']}),
        hparams.update({'time_varying_known_feature': ['Ghi']})
        hparams.update({'time_varying_unknown_feature': [ ]})
        hparams.update({'targets':['NetLoad']}) 
        
    elif  dataset=='albania_dataset':
        data  =load_albania_data().select_dtypes(exclude='object')
        hparams.update({'time_varying_known_categorical_feature': ['DayOfWeek', 'Hour', 'Holiday', 'NightHour', 'AnnualHoliday']}),
        hparams.update({'time_varying_known_feature': ['Temp', 'Humid']})
        hparams.update({'time_varying_unknown_feature': []})
        hparams.update({'targets':['NetLoad']}) #'Load',
        
    
    elif dataset=='uk_dataset': 
        data = load_uk_data(add_ghi=True).select_dtypes(exclude='object')
        data=data.reset_index().drop_duplicates(subset='timestamp')
        data=data.drop(columns=[ 'index'])

        columns = list(data.columns)[1:]
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True).dt.tz_localize(None)
        data=TimeSeries.from_dataframe(data, 'timestamp', columns, fill_missing_dates=True).pd_dataframe()
        hparams.update({'time_varying_known_categorical_feature': ['DAYOFWEEK', 'DAY', 'HOUR', 'Session']}),
        hparams.update({'time_varying_known_feature': ['Ghi']})
        hparams.update({'time_varying_unknown_feature': []})
        hparams.update({'targets':['NetLoad']})
        
    
    
    if  dataset=='pv_generation':
        data = load_hybrid_res_data()
        hparams.update({'time_varying_known_categorical_feature': ['DAYOFYEAR','WEEK','HOUR', 'Session','Season']})
        hparams.update({'time_varying_known_feature': ['Radiation']})
        hparams.update({'time_varying_unknown_feature': [ ]})
        hparams.update({'targets':['PVGen(MWh)']}) 
        
    elif  dataset=='wind_generation': 
        data = load_hybrid_res_data()
        hparams.update({'time_varying_known_categorical_feature': ['DAYOFYEAR','WEEK','HOUR','Season']})
        hparams.update({'time_varying_known_feature': ['WindSpeed']})
        hparams.update({'time_varying_unknown_feature': [ ]})
        hparams.update({'targets':['WindGen(MWh)']}) 
        
    elif  dataset=='combined_generation':
        data = load_hybrid_res_data()
        data['NetLoad']=data['PVGen(MWh)'].values +data['WindGen(MWh)'].values
        hparams.update({'time_varying_known_categorical_feature': ['DAYOFYEAR','WEEK','HOUR', 'Session','Season']})
        hparams.update({'time_varying_known_feature': ['Radiation', 'WindSpeed']})
        hparams.update({'time_varying_unknown_feature': ['PVGen(MWh)', 'PVGen(MWh)' ]})
        hparams.update({'targets':['NetLoad']}) 
        
   
   
    if experiment_type=='benchmark':
        file_name=f'{exp_name}_{dataset}_{experiment_type}_{window_type}' 
        bactesting= BacktestingForecast(hparams,  data,  file_name,
                                        n_splits=n_splits, forecast_len = forecast_len, 
                                        incremental_len = incremental_len, min_train_len=min_train_len,
                                        window_type=window_type, 
                                        max_epochs=max_epochs)
        list_models=[ 'MLPForecast', 'NBEATS', 'NHiTS', 'LSTM', 'MSTL' ,'CATBOOST', 'MSTL', 'SeasonalNaive', 'RF', 'LREGRESS', 'PatchTST','TimesNet', 'FEDformer']
        for encoder_type in  list_models: 
            hparams.update({'encoder_type':encoder_type})
            metrics=bactesting.fit(hparams, autotune=autotune)
            
    elif experiment_type=='short-long':
        hparams.update({'time_varying_known_categorical_feature': [ 'DAYOFWEEK', 'DAY', 'HOUR', 'Session']}),
        hparams.update({'time_varying_known_feature': ['Ghi']})
        hparams.update({'time_varying_unknown_feature': []})
        file_name=f'{exp_name}_{dataset}_{experiment_type}_{window_type}' 
        list_models=['MLPForecast']#'NHiTS',  'PatchTST', 'TimesNet'
        for window_size in [ 48]: # 48, 96, 144, 192, 336, 384, 432:
            for horizon in [24, 12, 2]: #[2, 12, 24, 48, 96, 192, 336]

                file_name=f'{exp_name}_{dataset}_{experiment_type}_{window_size}_{horizon}_{window_type}' 
                
                bactesting= BacktestingForecast(hparams,  data,  file_name,
                                            n_splits=n_splits, forecast_len = forecast_len, 
                                            incremental_len = incremental_len, min_train_len=min_train_len,
                                            window_type=window_type, 
                                            max_epochs=max_epochs)
                for encoder_type in  list_models: 
                    hparams.update({'encoder_type':encoder_type})
                    print(f"Training model {encoder_type}, with horizon ={horizon} and window={window_size}")
                    file_path=f"../results/PT-Benchmark_expanding/{hparams['encoder_type']}/best_params.npy"
                    if os.path.exists(file_path):
                        hparams.update(np.load(file_path, allow_pickle=True).item())
                        
                    hparams.update({'window_size':window_size})
                    hparams.update({'horizon':horizon})
                    metrics=bactesting.fit(hparams, autotune=False)
                    
                    
    elif experiment_type=='mlpf-model-design':
        hparams.update({'encoder_type':'MLPForecast'})
        for emb_type in ['CombinedEmb']:#"None", 'PosEmb', 'RotaryEmb',
            
            for comb_type in ['attn-comb', 'weighted-comb', 'addition-comb']:
                
                file_name=f'{exp_name}_{dataset}_{experiment_type}_{emb_type}_{comb_type}' 
                bactesting= BacktestingForecast(hparams,  data,  exp_name=file_name,
                                                n_splits=n_splits, forecast_len = forecast_len, 
                                                incremental_len = incremental_len, min_train_len=min_train_len,
                                                window_type=window_type, 
                                                max_epochs=max_epochs)
                print(f"Training model MLPF, with {comb_type} and {emb_type}")
                file_path=f"../results/PT-Benchmark_expanding/{hparams['encoder_type']}/best_params.npy"
                if os.path.exists(file_path):
                    hparams.update(np.load(file_path, allow_pickle=True).item())
                
                hparams.update({'embed_type': emb_type})
                hparams.update({'comb_type':comb_type})
                metrics=bactesting.fit(hparams, autotune=False)
            
        
                    
    
   
    
    
  
    
       
    



                    



       
if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    

 
    argparser = argparse.ArgumentParser(description='TEST_NEW_PIPELINE')
    argparser.add_argument('--dataset',       type=str, default='pt_dataset', help='Type of dataset')
    argparser.add_argument('--exp_name',       type=str, default="PT-Benchmark", help='Type of dataset')
    argparser.add_argument('--exp_type',       type=str, default="short-long", help='Type of dataset')
    argparser.add_argument('--data_path',       type=str, default="../dataset/", help='data_path')
    argparser.add_argument('--baseline',        action="store_true", help="A boolean conformal_loss")
    argparser.add_argument('--epochs',       type=int, default=20, help='number of iterations')
    argparser.add_argument('--seed',       type=int, default=200, help='number of seeds')
    argparser.add_argument('--trials',       type=int, default=50, help='number of trials')
    argparser.add_argument("--autotune", action="store_true", help="A boolean conformalize")
    args = argparser.parse_args()
    
    
    run_backtesting(forecast_len=6, 
                    incremental_len = 2,
                     min_train_len=12,
                    n_splits=10,
                    window_type='expanding',
                    seed=args.seed,
                    max_epochs=args.epochs,
                    autotune=args.autotune,
                    experiment_type=args.exp_type,
                    dataset=args.dataset,
                    exp_name=args.exp_name,
                    num_trials=args.trials
                   )
    
    