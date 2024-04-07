from utils.data import  TimeSeriesDataset, TimeSeriesLazyDataset, DataLoader
from darts import TimeSeries
import optuna
from tqdm import tqdm
from timeit import default_timer
from pytorch_lightning.callbacks import Callback, EarlyStopping
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
from utils.data_processing import get_index
from .evaluation import evaluate_point_forecast
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import numpy as np
from .optuna_pruner import PyTorchLightningPruningCallback
from .baseline_models import BaselineDNNModel




class DeterministicBaselineForecast(object):
    def __init__(self, hparams, exp_name="Tanesco", seed=42, root_dir="../", file_name=None):

        results_path = Path(f"{root_dir}/results/{exp_name}/{hparams['encoder_type']}/")
        logs = Path(f"{root_dir}/logs/{exp_name}/{hparams['encoder_type']}/")
        figures = Path(f"{root_dir}/figures/{exp_name}/{hparams['encoder_type']}/")
        figures.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)
        
        self.logs = logs
        self.results_path=results_path
        self.figures =  figures
        self.file_name=file_name
        self.hparams=hparams
        self.exp_name=exp_name
        self.root_dir=root_dir
        self.baseline_models = BaselineDNNModel()



    def process_df(self, data):
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
        columns = list(data.columns)[1:]
        df = TimeSeries.from_dataframe(data, 'timestamp', columns)
        data = df.pd_dataframe().reset_index()
        return data
    
    def inverse_scaling(self, target, scaler):
        B, T, C = target.shape
        target = scaler.inverse_transform(target.reshape(B*T, C)) 
        return target.reshape( B, T, C)
    
    def fit(self, trial=None, params=None, file_name=None):

        if trial is not None:
            file_name=f"{file_name}_{trial.number}"
            checkpoints = Path(f"{self.root_dir}/checkpoints/{self.exp_name}/{self.hparams['encoder_type']}/{self.file_name}_{trial.number}")
        else:
            checkpoints = Path(f"{self.root_dir}/checkpoints/{self.exp_name}/{self.hparams['encoder_type']}/{self.file_name}")
            file_name=self.file_name
        checkpoints.mkdir(parents=True, exist_ok=True)
        
        if params is None and trial is not None :
            params = self.baseline_models.get_hyparams(trial, self.hparams)
            callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        else:
            callback = None
        model = self.baseline_models.get_model(params, checkpoints, callback, self.exp_name)
            
        start_time = default_timer()
        if self.hparams['encoder_type'] in ['TFT', 'D-LINEAR', 'TRANSFORMER']: #'RNN'
            model.fit(self.train, future_covariates=self.train_cov, val_series=self.val, val_future_covariates=self.val_cov)
            
        # elif self.hparams['encoder_type'] in ['NHiTS', 'NBEATS']:
        #     model.fit(self.train, past_covariates=self.train_cov, val_series=self.val, val_past_covariates=self.val_cov)
            
        elif self.hparams['encoder_type'] in ['TCN']:
            model.fit(self.train,  val_series=self.val)
            
        elif self.hparams['encoder_type'] in [ 'CATBOOST',  "RF", 'LREGRESS']:
            model.fit(self.train, future_covariates=self.train_cov)
            
        elif self.hparams['encoder_type'] in ['MSTL', 'SeasonalNaive', 'AutoARIMA']:
            train_df=self.transform_darts_data(self.train, self.target_columns)
            model.fit(train_df)

        elif self.hparams['encoder_type'] in ['NHiTS', 'NBEATS', 'RNN', 'LSTM', 'TimesNet', 'PatchTST', 'FEDformer']:
            train_df=self.transform_neuralforecast_data(self.train, self.hparams)
            model.fit(df=train_df, val_size=len(self.test))
            model.save(path=str(checkpoints), save_dataset=True, overwrite=True)
            if self.hparams['autotune']:
                study = model.models[0].results
                print(f"Best value: {study.best_value}, Best params: {study.best_params}")
                self.hparams.update(study.best_params)
                np.save(f"../results/{self.exp_name}/{self.hparams['encoder_type']}/best_params.npy", self.hparams)        
            
        train_walltime = default_timer() - start_time

        
        start_time = default_timer()
        if self.hparams['encoder_type'] in ['TFT', 'D-LINEAR', 'TRANSFORMER']: #'RNN'
            best_model =  model.load_from_checkpoint(model_name=self.hparams['encoder_type'], 
                                                         work_dir=checkpoints, best=True)

            pred =  best_model.predict(n=len(self.test),
                                        series=self.train.concatenate(self.val),
                                        future_covariates=self.covariates).values()
        # elif self.hparams['encoder_type'] in  ['NHiTS', 'NBEATS']:
        #     best_model =  model.load_from_checkpoint(model_name=self.hparams['encoder_type'], 
        #                                                  work_dir=checkpoints, best=True)

        #     pred =  best_model.predict(n=len(self.test),
        #                                 series=self.train.concatenate(self.val),
        #                                 past_covariates=self.covariates).values()
        elif self.hparams['encoder_type'] in ['TCN']:
            best_model =  model.load_from_checkpoint(model_name=self.hparams['encoder_type'], 
                                                         work_dir=checkpoints, best=True)

            pred =  best_model.predict(n=len(self.test),
                                        series=self.train.concatenate(self.val)).values()
            
        elif self.hparams['encoder_type'] in ['CATBOOST',  "RF", 'LREGRESS']:
            pred = model.predict(len(self.test), future_covariates=self.test_cov).values()
            
        elif self.hparams['encoder_type'] in ['MSTL', 'SeasonalNaive', 'AutoARIMA']:
            pred = model.predict(h=len(self.test))
            pred = pred[self.hparams['encoder_type']].values
            
        elif self.hparams['encoder_type'] in ['NHiTS', 'NBEATS', 'RNN', 'LSTM', 'TimesNet', 'PatchTST', 'FEDformer']:
            # if not self.hparams['autotune']:
            #     model = model.load(str(checkpoints))
            test_df=self.transform_neuralforecast_data(self.test, self.hparams)
            pred = self.get_prediction_from_nixtlamodel(test_df, model, self.hparams)
            
        test_walltime = default_timer() - start_time
        if self.hparams['encoder_type'] in ['NHiTS', 'NBEATS', 'RNN', 'LSTM', 'TimesNet', 'PatchTST', 'FEDformer']:
            ouputs=self.post_process_nixtla_pred(pred, train_walltime, test_walltime, file_name)
        else:
            ouputs=self.post_process_pred(pred, train_walltime, test_walltime, file_name)
        ouputs['train-time']=train_walltime
        if trial is not None:
            return ouputs['NetLoad_metrics']['mae'].mean()
        else:
            return ouputs
        
        
    def prepare_data(self, experiment, train_df, test_df, val_df=None):
        
        
       

        # Handle for Mac GPU
        if torch.backends.mps.is_available():
            train_df[train_df.select_dtypes(np.float64).columns] = train_df.select_dtypes(np.float64).astype(np.float32)
            test_df[test_df.select_dtypes(np.float64).columns] = test_df.select_dtypes(np.float64).astype(np.float32)
            test_df[test_df.select_dtypes(np.float64).columns] = test_df.select_dtypes(np.float64).astype(np.float32)
            # train_df = train_df.astype(np.float32)
            # test_df = test_df.astype(np.float32)
            if val_df is not None:
                val_df[val_df.select_dtypes(np.float64).columns] = val_df.select_dtypes(np.float64).astype(np.float32)
                # val_df = val_df.astype(np.float32)

        #check if train_df has no NAN
        assert train_df.isnull().sum().sum()==0, "Train data has NAN"
        assert test_df.isnull().sum().sum()==0, "Test data has NAN"
        if self.hparams['encoder_type'] not in ['NHiTS', 'NBEATS', 'RNN', 'LSTM', 'TimesNet', 'PatchTST', 'FEDformer', 'RF', 'CATBOOST', 'LREGRESS', 'MSTL', 'SeasonalNaive']:
            assert val_df.isnull().sum().sum()==0, "Val data has NAN"

        self.target_transformer = experiment.target_transformer
        
        self.target_columns=[f"{self.hparams['targets'][i]}_target" for i in range(len(self.hparams['targets']))]
        #train_df = self.process_df(train_df)
        #test_df  = self.process_df(test_df)
        #if val_df is not None:
            #val_df  = self.process_df(val_df)
        self.index=get_index(test_df,self. hparams, test=True)
        self.target =  test_df.iloc[self.hparams['window_size']:][self.target_columns].values
        #self.target_range = self.target.max(0)-np.where(self.target.min(0)<0, self.target.min(0), 0)
        self.target_range = experiment.installed_capacity
        
        covariates=experiment.seasonality_columns + self.hparams['time_varying_known_feature']
    
        if self.hparams['encoder_type'] not in ['NHiTS', 'NBEATS', 'RNN', 'LSTM', 'TimesNet', 'PatchTST', 'FEDformer']:
            self.train = TimeSeries.from_dataframe(train_df, 
                                                    'timestamp', 
                                                    self.target_columns,
                                                    )
            self.test = TimeSeries.from_dataframe(test_df, 
                                                        'timestamp', 
                                                        self.target_columns)
            if val_df is not None:
                self.val = TimeSeries.from_dataframe(val_df, 
                                                        'timestamp', 
                                                        self.target_columns)
                self.val_cov = TimeSeries.from_dataframe(
                                val_df, 'timestamp', covariates)

            self.train_cov = TimeSeries.from_dataframe(
                                train_df, 'timestamp', covariates)

            self.test_cov = TimeSeries.from_dataframe(
                                test_df, 'timestamp', covariates)
            if val_df is not None:
                self.covariates = self.train_cov.concatenate(self.val_cov).concatenate(self.test_cov)
                
        else:
            self.train = train_df
            self.test = test_df
        
        
    def transform_neuralforecast_data(self, data, hparams):
        columns = ["timestamp"]+hparams["targets"]+hparams["time_varying_known_feature"]+hparams["time_varying_unknown_feature"]+hparams["time_varying_known_categorical_feature"]
        df = data[columns]
        df=df.rename(columns={'timestamp':'ds'})
        #df.index.name='ds'
        #df=df.reset_index()
        df=df.rename(columns={hparams['targets'][0]:'y'})
        df.insert(0, 'unique_id', 'NetLoad')
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return df
    
    
    def transform_darts_data(self, data, target_columns):
        df = data.pd_dataframe()[target_columns].reset_index()
        df.columns = ['ds', 'y']
        df.insert(0, 'unique_id', 'NetLoad')
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return df


    def transform_darts_covariates(self, data, cov_columns):
        df = data.pd_dataframe()[cov_columns].reset_index()
        df=df.rename(columns={'timestamp':'ds'}) 
        df.insert(0, 'unique_id', 'NetLoad')
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return df
            
            
    def post_process_pred(self, pred, train_walltime, test_walltime, file_name):
        Ndays = len(self.target)//self.hparams['horizon']
        loc = pred[self.hparams['window_size']:][:Ndays*self.hparams['horizon']]
        target = self.target[:Ndays*self.hparams['horizon']].reshape(Ndays, self.hparams['horizon'], -1)
        loc = loc.reshape(Ndays,  self.hparams['horizon'], -1)
        
        outputs = {}
        
        outputs['pred'] = loc
        outputs['index']=self.index[:, self.hparams['window_size']:]
        outputs['true']=target
        outputs['train-time']=train_walltime
        outputs['test-time']=test_walltime
        outputs['target-range']=self.target_range
        #outputs['inputs']= features

        for k in ['pred',   'true']:
            outputs[k]=self.inverse_scaling(outputs[k], self.target_transformer)
            
        outputs = evaluate_point_forecast(outputs, outputs['target-range'], self.hparams, self.exp_name, file_name=file_name, show_fig=False)
        return outputs
    
    
    def post_process_nixtla_pred(self, pred_df, train_walltime, test_walltime, file_name):
        
        Ndays = len(pred_df)//self.hparams['horizon']
        if self.hparams['autotune']:
            if self.hparams['encoder_type'] == 'NHiTS':
                print(pred_df.columns)
                loc = pred_df['Auto'+self.hparams['encoder_type'].upper()].values
            else:
                loc = pred_df['Auto'+self.hparams['encoder_type']].values
        else:
            if self.hparams['encoder_type'] == 'NHiTS':
                loc = pred_df[self.hparams['encoder_type'].upper()].values
            else:
                loc = pred_df[self.hparams['encoder_type']].values
        target = pred_df['true'].values
        index = pred_df['ds'].values
        loc = loc.reshape(Ndays,  self.hparams['horizon'], -1)
        target = target.reshape(Ndays,  self.hparams['horizon'], -1)
        index = index.reshape(Ndays,  self.hparams['horizon'], -1)
        
        outputs = {}
        outputs['pred'] = loc
        outputs['index']=self.index[:, self.hparams['window_size']:]
        outputs['true']=target
        outputs['train-time']=train_walltime
        outputs['test-time']=test_walltime
        outputs['target-range']=self.target_range
        #outputs['inputs']= features
       
        outputs = evaluate_point_forecast(outputs, outputs['target-range'], self.hparams, self.exp_name, file_name=file_name, show_fig=False)
        return outputs
    
    
    def auto_tune_model(self, num_trials=10):
        
        def print_callback(study, trial):
            print(f"Current value: {trial.value}, Current params: {trial.params}")
            print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

        study_name=f"{self.exp_name}_{self.hparams['encoder_type']}"
        storage_name = "sqlite:///{}.db".format(study_name)
        
       
        pruner = pruner=optuna.pruners.SuccessiveHalvingPruner()
        study = optuna.create_study( direction="minimize", 
                                    study_name=study_name, 
                                    storage=storage_name,
                                    load_if_exists=True)
        study.optimize(self.fit, n_trials=num_trials, callbacks=[print_callback])
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
        self.hparams.update(study.best_trial.params)
        np.save(f"../results/{self.exp_name}/{self.hparams['encoder_type']}/best_params.npy", self.hparams)
       

        
    def get_prediction_from_nixtlamodel(self, test_df, model, hparams):
    
        Ndays = len(test_df)//hparams['horizon']
        all_predictions = []
        
        for i in range(Ndays):

            first_day_start = i*hparams['horizon']
            first_day_end = i*hparams['horizon']+ hparams['horizon']
            
            second_day_start = first_day_end
            second_day_end = second_day_start + hparams['horizon']

            historical_data = test_df.iloc[first_day_start:second_day_end]
            
            future_data = test_df.iloc[second_day_end:second_day_end+hparams['horizon']]

            if (len(historical_data)!=hparams['window_size']) or  (len(future_data)!=hparams['horizon']):
                continue

            pred_df = model.predict(historical_data, futr_df=future_data.drop(columns=['y']))

            pred_df = pred_df.reset_index(drop=False)
            pred_df['true']=future_data.y.values
            all_predictions.append(pred_df)

        all_predictions = pd.concat(all_predictions)
        return all_predictions
        
    