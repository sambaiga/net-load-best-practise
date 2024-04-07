
from .baselines import DeterministicBaselineForecast
from MLPF.model import MLPForecast
from darts import TimeSeries
import os
from copy import deepcopy
import numpy as np
import pandas as pd
# from net.utils import   get_latest_checkpoint
from .evaluation import evaluate_point_forecast
from orbit.constants.constants import TimeSeriesSplitSchemeKeys
from orbit.diagnostics.backtest import  TimeSeriesSplitter
import matplotlib_inline.backend_inline
from utils.data import DatasetObjective
import matplotlib_inline.backend_inline
import arviz as az
import matplotlib.pyplot as plt
from utils.data_processing import get_periods_for_exog_variable, fit_scaling_on_train_df, get_index
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")
az.style.use(["science", "grid", "arviz-doc", 'tableau-colorblind10'])


class BacktestingForecast(object):
    def __init__(self, hparams,  data,  exp_name=None, file_name=None, 
                        n_splits=10, forecast_len = 3, 
                        incremental_len = 1, min_train_len=12,
                        window_type='expanding', 
                        max_epochs=50,
                        root_dir="../"):
        
        hparams.update({ 'max_epochs': max_epochs})
        min_train_len = int(hparams['SAMPLES_PER_DAY']*30*min_train_len) # minimal length of window length
        forecast_len = int(hparams['SAMPLES_PER_DAY']*30*forecast_len) # length forecast window
        incremental_len = int(hparams['SAMPLES_PER_DAY']*30*incremental_len) # step length for moving forward)
        self.hparams=hparams
        self.exp_name=exp_name
        self.file_name=file_name
        self.len = len(data)
        self.exog_period = get_periods_for_exog_variable(hparams, data)
        self.generator = TimeSeriesSplitter(df=data.reset_index(),
                                                    min_train_len=min_train_len,
                                                    incremental_len=incremental_len,
                                                    forecast_len=forecast_len,
                                                    window_type=window_type,
                                                    n_splits=n_splits,
                                            date_col='timestamp')
        
        
    def fit_train_test(self, hparams, train_df, test_df, key, train_ratio=0.90):
        file_path=f"../results/{self.exp_name}/{hparams['encoder_type']}/best_params.npy"
        if os.path.exists(file_path):
            hparams.update(np.load(f"../results/{self.exp_name}/{hparams['encoder_type']}/best_params.npy", allow_pickle=True).item())
            
        print(f"---------------Fit {hparams['encoder_type']} Backtesting expanding-{key} train test Training --------------------------")
        print("")
        print(f"Train_window: from {train_df['timestamp'].iloc[0]} to  {train_df['timestamp'].iloc[-1]} ")
        print("")
        print(f"Test_window: from {test_df['timestamp'].iloc[0]} to  {test_df['timestamp'].iloc[-1]}")
        print("")
        input_scaler, target_scaler=fit_scaling_on_train_df(hparams, train_df)
        experiment= DatasetObjective(hparams=hparams, 
                                            data=train_df.set_index('timestamp', drop=True), add_ghi=False, 
                                            scaler=input_scaler, 
                                            target_scaler=target_scaler, 
                                            fit_scaler=False,
                                            exog_periods=self.exog_period)

        train_df = experiment.data
        train_df.attrs['freq'] = pd.to_timedelta(int(24*60/hparams['horizon']), unit='T')

        train_df=train_df.reset_index()
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp']).dt.tz_localize(None)
        

        test_experiment= DatasetObjective(hparams=hparams, 

                                            data=test_df.set_index('timestamp', drop=True), add_ghi=False, 
                                            scaler=input_scaler, 
                                            target_scaler=target_scaler, 
                                            fit_scaler=False,
                                            exog_periods=self.exog_period)
        test_df = test_experiment.data
      
        test_df.attrs['freq'] = pd.to_timedelta(int(24*60/hparams['horizon']), unit='T')
        test_df=test_df.reset_index()

        test_df['timestamp'] = pd.to_datetime(test_df['timestamp']).dt.tz_localize(None)
        columns = list(test_df.columns)[1:]
        df = TimeSeries.from_dataframe(test_df, 'timestamp', columns, fill_missing_dates=True)
        test_df=df.pd_dataframe().reset_index()

            

        file_name=f'{key}'
        #file_name=f'{self.file_name}_{key}'

        if hparams['encoder_type']=='MLPForecast':
            model=MLPForecast(exp_name=self.exp_name, file_name=file_name, hparams=hparams)
            size = int(len(train_df)*train_ratio)
            train_df, val_df = train_df.iloc[:size], train_df[size:]
            train_size= (train_ratio*len(train_df))/self.len
            train_walltime=model.fit(train_df, val_df,  experiment)
            outputs=model.predict_from_df(test_df, experiment)
            outputs['train-time']=train_walltime
        else:
            model = DeterministicBaselineForecast(exp_name=self.exp_name, file_name=file_name, hparams=hparams) 
            
            if hparams['encoder_type'] in ['D-LINEAR', 'TCN', 'TFT']: #'NHiTS', 'NBEATS', 'RNN'
                size = int(len(train_df)*train_ratio)
                train_df, val_df = train_df.iloc[:size], train_df[size:]
                model.prepare_data(experiment, train_df, test_df, val_df)
                train_size= (train_ratio*len(train_df))/self.len
            else:
                model.prepare_data(experiment, train_df, test_df, None)
                train_size= len(train_df)/self.len
            
            outputs=model.fit(None, hparams, file_name=self.file_name)
                    
       
        outputs['train-size']=train_size
        np.save(f"../results/{self.exp_name}/{hparams['encoder_type']}/{file_name}_processed_results.npy", outputs)
        metrics=outputs['NetLoad_metrics']
        metrics[f'folds']=key
        return metrics
       
                
        
    def fit(self,  hparams, train_ratio = 0.90, autotune=True):
        bactestingting_metrics=[]
        
        file_path=f"../results/{self.exp_name}/{hparams['encoder_type']}/best_params.npy"
        if os.path.exists(file_path):
            hparams.update(np.load(f"../results/{self.exp_name}/{hparams['encoder_type']}/best_params.npy", allow_pickle=True).item())
            #autotune=False
        #else:
           #autotune=True

        for train_df, test_df, scheme, key in self.generator.split():
    
            print(f"---------------Fit {hparams['encoder_type']} Backtesting expanding-{key+1} Cross validation Training --------------------------")
            print("")
            print(f"Train_window: from {train_df['timestamp'].iloc[0]} to  {train_df['timestamp'].iloc[-1]} ")
            print("")
            print(f"Test_window: from {test_df['timestamp'].iloc[0]} to  {test_df['timestamp'].iloc[-1]}")
            print("")
            #train_df=train_df.drop(columns=['index'])
            #test_df=test_df.drop(columns=['index'])
            #train_df.attrs['freq'] = pd.to_timedelta(30, unit='T')
            input_scaler, target_scaler=fit_scaling_on_train_df(hparams, train_df)

            experiment= DatasetObjective(hparams=hparams, 
                                            data=train_df.set_index('timestamp', drop=False), add_ghi=False, 
                                            scaler=input_scaler, 
                                            target_scaler=target_scaler, 
                                            fit_scaler=False,
                                            exog_periods=self.exog_period)

            train_df = experiment.data
            train_df.attrs['freq'] = pd.to_timedelta(int(24*60/hparams['horizon']), unit='T')

            train_df=train_df.reset_index()
            train_df['timestamp'] = pd.to_datetime(train_df['timestamp']).dt.tz_localize(None)
            columns = list(train_df.columns)[1:]
            df = TimeSeries.from_dataframe(train_df, 'timestamp', columns, fill_missing_dates=True)
            train_df=df.pd_dataframe().reset_index()

            test_experiment= DatasetObjective(hparams=hparams, 

                                            data=test_df.set_index('timestamp', drop=True), add_ghi=False, 
                                            scaler=input_scaler, 
                                            target_scaler=target_scaler, 
                                            fit_scaler=False,
                                            exog_periods=self.exog_period)
            test_df = test_experiment.data
            test_df.attrs['freq'] = pd.to_timedelta(int(24*60/hparams['horizon']), unit='T')
            test_df=test_df.reset_index()

            test_df['timestamp'] = pd.to_datetime(test_df['timestamp']).dt.tz_localize(None)
            columns = list(test_df.columns)[1:]
            df = TimeSeries.from_dataframe(test_df, 'timestamp', columns, fill_missing_dates=True)
            test_df=df.pd_dataframe().reset_index()

            


            file_name=f'{key+1}_cross_validation'


            if hparams['encoder_type']=='MLPForecast':
                model=MLPForecast(exp_name=self.exp_name, file_name=file_name, hparams=hparams)
                size = int(len(train_df)*train_ratio)
                train_df, val_df = train_df.iloc[:size], train_df[size:]
                train_size= (train_ratio*len(train_df))/self.len
                if autotune:
                    model.auto_tune_model(train_df=train_df, val_df=val_df, experiment=experiment, num_trials=hparams['num_trials'])
                    break
                else:
                    training_time=model.fit(train_df, val_df,  experiment)
                    outputs=model.predict(test_df, experiment, test=True)
                    outputs['Train-time']=training_time
                    
                
            else:
                
                hparams.update({'autotune': autotune})

                model = DeterministicBaselineForecast(exp_name=self.exp_name, file_name=file_name, hparams=hparams) 
                if hparams['encoder_type'] in ['D-LINEAR', 'TCN', 'TFT']: #'NHiTS', 'NBEATS', 'RNN'
                    size = int(len(train_df)*train_ratio)
                    train_df, val_df = train_df.iloc[:size], train_df[size:]
                    model.prepare_data(experiment, train_df, test_df, val_df)
                    train_size= (train_ratio*len(train_df))/self.len
                else:
                    model.prepare_data(experiment, train_df, test_df, None)
                    train_size= len(train_df)/self.len
                
                
            
                if autotune and hparams['encoder_type'] in ['D-LINEAR', 'TCN', 'TFT']: #'NHiTS', 'NBEATS', 'RNN'
                    model.auto_tune_model(num_trials=hparams['num_trials'])
                    break
                else:
                    outputs=model.fit(None, hparams, file_name=self.file_name)
                    if autotune and hparams['encoder_type'] in ['NHiTS', 'NBEATS', 'RNN', 'LSTM', 'TimesNet', 'PatchTST', 'FEDformer']:
                        break
                
            
            outputs['train-size']=train_size
            np.save(f"../results/{self.exp_name}/{hparams['encoder_type']}/{file_name}_processed_results.npy", outputs)
            metrics=outputs['NetLoad_metrics']
            metrics['train-size']=train_size
            metrics['folds']=key+1
            bactestingting_metrics.append(metrics)
        if not autotune:
            bactestingting_metrics=pd.concat(bactestingting_metrics)
            return bactestingting_metrics
        else:
            return None
        

        
    def plot(self, ax=None,  middle = 10, large = 12, train_ratio=0.8):
        tr_start = list()
        tr_len = list()
        # technically should be just self.forecast_len
        tt_len = list()
        yticks = list(range(1,self.generator.n_splits + 1))
        val_len = list()
        for idx, scheme in self.generator._split_scheme.items():
            # fill in indices with the training/test groups
            tr_start.append(list(scheme[TimeSeriesSplitSchemeKeys.TRAIN_IDX.value])[0])
            train_len=len(list(scheme[TimeSeriesSplitSchemeKeys.TRAIN_IDX.value]))
            tr_len.append(int(train_ratio*train_len))
            val_len.append(int((1-train_ratio)*train_len))
            tt_len.append(self.generator.forecast_len)

        tr_start = np.array(tr_start)
        tr_len = np.array(tr_len)
        val_len=np.array(val_len)

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(9,3))
        ax.barh(
                    yticks,
                    tr_len,
                    align="center",
                    height=0.5,
                    left=tr_start,
                    label="train",
                )

        ax.barh(
                    yticks,
                    val_len,
                    align="center",
                    height=0.5,
                    left=tr_start + tr_len,
                    label="val",
                )
        ax.barh(
                    yticks,
                    tt_len,
                    align="center",
                    height=0.5,
                    left=tr_start + tr_len +val_len,
                    label="test",
                )

        strftime_fmt="%Y-%m-%d"
        xticks_loc = np.array(ax.get_xticks(), dtype=int)
        new_xticks_loc = np.linspace(
                        0, len(self.generator.dt_array) - 1, num=len(xticks_loc)
                    ).astype(int)
        dt_xticks = self.generator.dt_array[new_xticks_loc]
        dt_xticks = dt_xticks.strftime(strftime_fmt)
        ax.set_xticks(new_xticks_loc)
        ax.set_xticklabels(dt_xticks)

        # some formatting parameters
       

        ax.set_yticks(yticks)
        ax.set_ylabel("Folds", fontsize=large)
        ax.invert_yaxis()
        # ax.grid(which="both", color='grey', alpha=0.5)
        ax.tick_params(axis="x", which="major", labelsize=middle)
        ax.set_title("Train/Test Split Scheme", fontsize=large)

        return ax

