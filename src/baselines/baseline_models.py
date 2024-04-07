
from darts.models import   NHiTSModel, NBEATSModel, RNNModel, TCNModel, TFTModel, DLinearModel
from darts.models import TransformerModel, ExponentialSmoothing
from darts.models import  CatBoostModel,   LinearRegressionModel, RandomForest
from statsforecast.models import AutoARIMA, SeasonalNaive, MSTL
from timeit import default_timer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers
import numpy as np
from pathlib import Path
import torch
from .evaluation import evaluate_point_forecast
from statsforecast.models import AutoARIMA, SeasonalNaive, MSTL
from statsforecast import StatsForecast
from neuralforecast.models import TimesNet, PatchTST, FEDformer, NHITS, LSTM, NBEATS
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.auto import AutoTimesNet, AutoPatchTST, AutoFEDformer, AutoLSTM, AutoNBEATS, AutoNHITS
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) # Use this to disable training prints from optuna



class BaselineDNNModel(object):
    
    def get_model(self, hparams, path, callbacks, exp_name):

        self.exp_name = exp_name

        # if hparams['encoder_type']=='NHiTS':
        #     model = self.get_nhits_model(hparams, path, callbacks)
            
        # if hparams['encoder_type']=='NBEATS':
        #     model = self.get_nbeats_model(hparams, path, callbacks)
            
        # if hparams['encoder_type']=='RNN':
        #     model = self.get_rnn_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='TCN':
            model = self.get_tcn_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='TFT':
            model = self.get_tft_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='TRANSFORMER':
            model = self.get_transformer_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='D-LINEAR':
            model = self.get_dlinear_model(hparams, path, callbacks)
            
        if hparams['encoder_type'] in ['MSTL',  'SeasonalNaive', 'AutoARIMA']:
            model = self.get_statistical_baselines(hparams)
            
        if hparams['encoder_type'] in ['CATBOOST', 'RF', 'LREGRESS']:
            model = self.get_conventional_baseline_model(hparams)

        if hparams['encoder_type'] in ['NHiTS', 'NBEATS', 'RNN', 'LSTM', 'TimesNet', 'PatchTST', 'FEDformer']:
            model = self.get_neuralforecast_baseline(hparams, path)
        return model
            
    def get_hyparams(self, trial, hparams):
        # if hparams['encoder_type']=='NHiTS':
        #     params = self.get_nhits_search_params(trial, hparams)
            
        # if hparams['encoder_type']=='NBEATS':
        #     params = self.get_nbeats_search_params(trial, hparams)
            
        # if hparams['encoder_type']=='RNN':
        #     params = self.get_rnn_search_params(trial, hparams)
            
        if hparams['encoder_type']=='TCN':
            params = self.get_tcn_search_params(trial, hparams)
            
        if hparams['encoder_type']=='TFT':
            params = self.get_tft_search_params(trial, hparams)
        
        if hparams['encoder_type']=='D-LINEAR':
            params = self.get_dlinear_search_params(trial, hparams)
            
        if hparams['encoder_type']=='TRANSFORMER':
            params=self.get_transformer_search_params(trial, hparams)
        return params
    
    
    def get_conventional_baseline_model(self, hparams):
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        if hparams['encoder_type']=='CATBOOST': 
            model=CatBoostModel(lags=2,
                                lags_past_covariates=None,
                                lags_future_covariates=[1,2],
                                add_encoders =encoders,
                                output_chunk_length=hparams['horizon'])
                
        elif hparams['encoder_type']=='RF':
            model=RandomForest(lags=2,
                                lags_past_covariates=None,
                                lags_future_covariates=[1,2],
                                add_encoders =encoders,
                                output_chunk_length=hparams['horizon'])
        elif hparams['encoder_type']=='LREGRESS':
            model=LinearRegressionModel(lags=2,
                                lags_past_covariates=None,
                                lags_future_covariates=[1,2],
                                add_encoders =encoders,
                                output_chunk_length=hparams['horizon'])
        return model
    

    def get_neuralforecast_baseline(self, hparams, path, wandb=False, root_dir='../../'):
        self.future_exog=hparams["time_varying_unknown_feature"]+hparams["time_varying_known_categorical_feature"]
        self.hparams=hparams
        models = []
        period=int(24*60/hparams['SAMPLES_PER_DAY'])

        # callback=[]
        # checkpoint_callback = ModelCheckpoint(dirpath=path, monitor='val_mae', mode='min', save_top_k=2)   
        # callback.append(checkpoint_callback)
        # lr_logger = LearningRateMonitor()
        # callback.append(lr_logger)

        logs = Path(f"{root_dir}/logs/{self.exp_name}/{self.hparams['encoder_type']}/")
        if not wandb:
            logger  = loggers.TensorBoardLogger(logs,  name = self.exp_name) 
        else:
            logger = loggers.WandbLogger(project=self.exp_name, log_model="all")

        if hparams['encoder_type']=='TimesNet':
            if(hparams['autotune']):
                print('Running AutoTimesNet..')
                auto_timesnet = AutoTimesNet(
                        h=hparams['horizon'],
                        config=self.get_auto_timesnet_search_params,
                        search_alg=optuna.samplers.TPESampler(),
                        backend='optuna',
                        num_samples=hparams['num_trials']
                    )
                models.append(auto_timesnet)
            else:
                print('Running TimesNet..')
                timesnet = TimesNet(
                            h=hparams['horizon'],
                            input_size=hparams['window_size'],
                            hidden_size = hparams['hidden_size'],
                            conv_hidden_size = hparams['conv_hidden_size'],
                            learning_rate=hparams['learning_rate'],
                            random_seed=hparams['random_seed'],
                            dropout=hparams['dropout'],
                            futr_exog_list= self.future_exog,
                            max_steps=hparams['max_epochs'],
                            val_check_steps=50,
                            early_stop_patience_steps=5,
                            enable_checkpointing=True,
                            logger=logger)
                models.append(timesnet)
            
        if hparams['encoder_type']=='PatchTST':
            if(hparams['autotune']):
                print('Running AutoPatchTST..')
                auto_patchtst = AutoPatchTST(
                        h=hparams['horizon'],
                        config=self.get_auto_patchtst_search_params,
                        search_alg=optuna.samplers.TPESampler(),
                        backend='optuna',
                        num_samples=hparams['num_trials']
                    )
                models.append(auto_patchtst)
            else:
                print('Running PatchTST..')
                patchtst = PatchTST(h=hparams['horizon'],
                            input_size=hparams['window_size'],
                            patch_len=hparams['patch_len'],
                            stride=hparams['stride'],
                            revin=False,
                            hidden_size=hparams['hidden_size'],
                            n_heads=hparams['n_heads'],
                            dropout=hparams['dropout'],
                            activation=hparams['activation'],
                            random_seed=hparams['random_seed'],
                            learning_rate=hparams['learning_rate'],
                            max_steps=hparams['max_epochs'],
                            val_check_steps=50,
                            early_stop_patience_steps=5,
                            enable_checkpointing=True,
                            logger=logger)
                models.append(patchtst)

        if hparams['encoder_type']=='FEDformer':
            if(hparams['autotune']):
                print('Running AutoFEDformer..')
                auto_fedformer = AutoFEDformer(
                        h=hparams['horizon'],
                        config=self.get_auto_fedformer_search_params,
                        search_alg=optuna.samplers.TPESampler(),
                        backend='optuna',
                        num_samples=hparams['num_trials']
                    )
                models.append(auto_fedformer)
            else:
                print('Running FEDformer..')
                fedformer = FEDformer(h=hparams['horizon'],
                            input_size=hparams['window_size'],
                            futr_exog_list= self.future_exog,
                            hidden_size=hparams['hidden_size'],
                            conv_hidden_size = hparams['conv_hidden_size'],
                            dropout=hparams['dropout'],
                            random_seed=hparams['random_seed'],
                            learning_rate=hparams['learning_rate'],
                            max_steps=hparams['max_epochs'],
                            val_check_steps=50,
                            early_stop_patience_steps=5,
                            enable_checkpointing=True,
                            logger=logger)
                models.append(fedformer)

        if hparams['encoder_type']=='LSTM':
            if(hparams['autotune']):
                print('Running AutoLSTM..')
                auto_lstm = AutoLSTM(
                        h=hparams['horizon'],
                        config=self.get_auto_lstm_search_params,
                        search_alg=optuna.samplers.TPESampler(),
                        backend='optuna',
                        num_samples=hparams['num_trials']
                    )
                models.append(auto_lstm)
            else:
                print('Running LSTM..')
                lstm = LSTM(h=hparams['horizon'],
                            input_size=hparams['window_size'],
                            futr_exog_list= self.future_exog,
                            encoder_hidden_size = hparams['encoder_hidden_size'],
                            encoder_n_layers=hparams['encoder_n_layers'],
                            context_size=hparams['context_size'],
                            decoder_hidden_size=hparams['decoder_hidden_size'],
                            decoder_layers=hparams['decoder_layers'],
                            random_seed=hparams['random_seed'],
                            learning_rate=hparams['learning_rate'],
                            max_steps=hparams['max_epochs'],
                            val_check_steps=50,
                            early_stop_patience_steps=5,
                            enable_checkpointing=True,
                            logger=logger)
                models.append(lstm)

        if hparams['encoder_type']=='NBEATS':
            if(hparams['autotune']):
                print('Running AutoNBEATS..')
                auto_nbeats = AutoNBEATS(
                        h=hparams['horizon'],
                        config=self.get_auto_nbeats_search_params,
                        search_alg=optuna.samplers.TPESampler(),
                        backend='optuna',
                        num_samples=hparams['num_trials']
                    )
                models.append(auto_nbeats)
            else:
                print('Running NBEATS..')
                nbeats = NBEATS(h=hparams['horizon'],
                            input_size=hparams['window_size'],
                            windows_batch_size = hparams['windows_batch_size'],
                            random_seed=hparams['random_seed'],
                            learning_rate=hparams['learning_rate'],
                            max_steps=hparams['max_epochs'],
                            val_check_steps=50,
                            early_stop_patience_steps=5,
                            enable_checkpointing=True,
                            logger=logger)
                models.append(nbeats)

        if hparams['encoder_type']=='NHiTS':
            if(hparams['autotune']):
                print('Running AutoNHiTS..')
                auto_nhits = AutoNHITS(
                        h=hparams['horizon'],
                        config=self.get_auto_nhits_search_params,
                        search_alg=optuna.samplers.TPESampler(),
                        backend='optuna',
                        num_samples=hparams['num_trials']
                    )
                models.append(auto_nhits)
            else:
                print('Running NHiTS..')
                nhits = NHITS(h=hparams['horizon'],
                            input_size=hparams['window_size'],
                            futr_exog_list= self.future_exog,
                            hist_exog_list= self.future_exog,
                            n_pool_kernel_size=hparams['n_pool_kernel_size'],
                            n_freq_downsample=hparams['n_freq_downsample'],
                            random_seed=hparams['random_seed'],
                            learning_rate=hparams['learning_rate'],
                            max_steps=hparams['max_epochs'],
                            val_check_steps=50,
                            early_stop_patience_steps=5,
                            enable_checkpointing=True,
                            logger=logger)
                models.append(nhits)
       
        nf = NeuralForecast(
                    models=models,
                    freq=f"{period}T"
                )

        return nf

    
    def get_statistical_baselines(self, hparams):
        period=int(24*60/hparams['SAMPLES_PER_DAY'])
        if hparams['encoder_type']=='MSTL':
            mstl = MSTL(
                        season_length=[hparams['SAMPLES_PER_DAY'], hparams['SAMPLES_PER_DAY'] * 7], # seasonalities of the time series 
                        trend_forecaster=AutoARIMA() # model used to forecast trend
                    )
            model = StatsForecast(
                        models=[mstl], # model used to fit each time series 
                        freq=f'{period}T', # frequency of the data
                    )
        elif hparams['encoder_type']== 'SeasonalNaive':
            model = StatsForecast(models=[SeasonalNaive(season_length= hparams['SAMPLES_PER_DAY'])], # model used to fit each time series 
                        freq=f'{period}T')
            
        elif hparams['encoder_type']=='ARIMA':
            model = StatsForecast(models=[AutoARIMA(season_length= hparams['SAMPLES_PER_DAY'])], # model used to fit each time series 
                        freq=f'{period}T')
        
        elif hparams['encoder_type']=='AutoARIMA':
            model = StatsForecast(models=[AutoARIMA(season_length= hparams['SAMPLES_PER_DAY'])], # model used to fit each time series 
                        freq=f'{period}T')
        return model
            
    def set_callback(self, callbacks):
        # throughout training we'll monitor the validation loss for early stopping
        early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
        if callbacks is None:
            callbacks = [early_stopper]
        else:
            callbacks = [early_stopper] + callbacks


        # detect if a GPU is available
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            pl_trainer_kwargs = {
                "accelerator": "gpu",
                 "devices": [0],
                "callbacks": callbacks,
            }
            num_workers = 4
        else:
            pl_trainer_kwargs = {"callbacks": callbacks}
            num_workers = 0
        return callbacks, pl_trainer_kwargs, num_workers
    
    
    def get_transformer_model(self, hparams, path, callback=None):
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    

        model = TransformerModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        model_name=hparams['encoder_type'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        d_model=hparams['latent_size'],
                        nhead=4,
                        norm_type=hparams['norm_type'],
                        num_encoder_layers=hparams['depth'],
                        num_decoder_layers=hparams['depth'],
                        dim_feedforward=hparams['latent_size']*2,
                        dropout=hparams['dropout'],
                        activation=hparams['activation'],
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        work_dir=path, 
                        optimizer_kwargs={"lr": hparams['lr']},
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
   
        
    def get_nhits_model(self, hparams, path, callback=None):
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    

        model = NHiTSModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        model_name=hparams['encoder_type'],
                        num_stacks=hparams['num_stacks'],
                        num_blocks=hparams['num_blocks'],
                        #pooling_kernel_sizes=hparams['pooling_kernel_sizes'],
                        #n_freq_downsample=hparams['n_freq_downsample'],
                        dropout=hparams['dropout'],
                        num_layers=hparams['depth'],
                        MaxPool1d = hparams['MaxPool1d'],
                        activation=hparams['activation'],
                        layer_widths=hparams['latent_size'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        save_=True, 
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        work_dir=path, 
                        optimizer_kwargs={"lr": hparams['lr']},
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
   
    def get_nbeats_model(self, hparams, path, callback=None):
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)

       
        model = NBEATSModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        generic_architecture=hparams['generic_architecture'],
                        model_name=hparams['encoder_type'],
                        num_stacks=hparams['num_stacks'],
                        num_blocks=hparams['num_blocks'],
                        num_layers=hparams['depth'],
                        activation=hparams['activation'],
                        layer_widths=hparams['latent_size'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        dropout=hparams['dropout'],
                        save_=True, 
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        optimizer_kwargs={"lr": hparams['lr']},
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
    
    def get_rnn_model(self, hparams, path, callback=None):
        
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    
        model = RNNModel(
                        input_chunk_length=hparams['window_size'],
                        model=hparams['rnn_type'],
                        model_name=hparams['encoder_type'],
                        n_rnn_layers=hparams['depth'],
                        hidden_dim =hparams['latent_size'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        dropout=hparams['dropout'],
                        save_checkpoints=True, 
                        training_length=hparams['window_size']+hparams['horizon'],
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        optimizer_kwargs={"lr": hparams['lr']},
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
    def get_tcn_model(self, hparams, path, callback=None):
        
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    
        model = TCNModel(input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        kernel_size=hparams['kernel_size'],
                        num_filters=hparams['num_filters'],
                        model_name=hparams['encoder_type'],
                        weight_norm=hparams['weight_norm'],
                        dilation_base=hparams['dilation_base'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        save_checkpoints=True, 
                        dropout=hparams['dropout'],
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        optimizer_kwargs={"lr": hparams['lr']},
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
    def get_tft_model(self, hparams, path, callback=None):
        
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    
        model = TFTModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        model_name=hparams['encoder_type'],
                        hidden_size=hparams['latent_size'],
                        lstm_layers=hparams['depth'],
                        num_attention_heads=hparams['num_attn_head'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        add_relative_index=hparams['add_relative_index'],
                        save_checkpoints=True, 
                        dropout=hparams['dropout'],
                        optimizer_kwargs={"lr": hparams['lr']},
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
    def get_dlinear_model(self, hparams, path, callback=None):
        
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    
        model = DLinearModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        model_name=hparams['encoder_type'],
                        #shared_weights=hparams['shared_weights'],
                        #kernel_size=hparams['kernel_size'],
                        const_init=hparams['const_init'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        save_checkpoints=True, 
                        optimizer_kwargs={"lr": hparams['lr']},
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    def get_dlinear_search_params(self, trial, params):
    
        
    
        const_init = trial.suggest_int("const_init", 2, 4)
        params.update({'const_init': const_init})

    
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        return params
    
    
    def get_transformer_search_params(self, trial, params):
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        nhead = {'nhead':trial.suggest_categorical("nhead", [4,8,16])}
        params.update(nhead)
        

        norm_type = trial.suggest_categorical("norm_type", ["LayerNorm", "RMSNorm", "LayerNormNoBias", None])
        params.update({'norm_type': norm_type})
        
        activation = trial.suggest_categorical("activation", ["GLU", "Bilinear", "ReGLU", "GEGLU", "SwiGLU", "ReLU", "GELU"])
        params.update({'activation': activation})

        
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})


        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        return params
    
    def get_tft_search_params(self, trial, params):
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        num_attn_head = {'num_attn_head':trial.suggest_categorical("num_attn_head", [4,8,16])}
        params.update(num_attn_head)
      
    
        add_relative_index = trial.suggest_categorical("add_relative_index", [False, True])
        params.update({'add_relative_index': add_relative_index})
        
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        return params
    
    
    def get_tcn_search_params(self, trial, params):
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)


        
        kernel_size = trial.suggest_int("kernel_size", 5, 25)
        params.update({'kernel_size': kernel_size})
        
        num_filters = trial.suggest_int("num_filters", 5, 25)
        
        params.update({'num_filters': num_filters})
        weight_norm = trial.suggest_categorical("weight_norm", [False, True])
        params.update({'weight_norm': weight_norm})
        
        dilation_base = trial.suggest_int("dilation_base", 2, 4)
        params.update({'dilation_base': dilation_base})

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        return params
    
    def get_rnn_search_params(self, trial, params):
        
        rnn_type =trial.suggest_categorical("rnn_type", ['GRU','LSTM'])
        params.update({'rnn_type': rnn_type})
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

       
        return params
    
    def get_nhits_search_params(self, trial, params):
        
        num_stacks=trial.suggest_int('num_stacks', 2, 10)
        params.update({'num_stacks': num_stacks})
        num_blocks=trial.suggest_int('num_blocks', 1, 10)
        params.update({'num_blocks': num_blocks})
        pooling_kernel_sizes=trial.suggest_categorical('pooling_kernel_sizes', [(2, 2, 2), (16, 8, 1)])
        params.update({'pooling_kernel_sizes': pooling_kernel_sizes})
        n_freq_downsample=trial.suggest_categorical('n_freq_downsample', [[168, 24, 1], [24, 12, 1], [1, 1, 1]]) 
        params.update({'n_freq_downsample': n_freq_downsample})

        MaxPool1d =trial.suggest_categorical("MaxPool1d", [False, True])
        params.update({'MaxPool1d': MaxPool1d})

        activation=trial.suggest_categorical("activation", ['ReLU','RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid'])
        params.update({'activation': activation})

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        
        return params
    
    
    def get_auto_timesnet_search_params(self, trial):
        return {
            'max_steps': self.hparams["max_epochs"],
            'input_size': self.hparams["window_size"],
            'futr_exog_list':self.future_exog,
            'hidden_size':trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512] ),
            'conv_hidden_size':trial.suggest_categorical("conv_hidden_size", [16, 32, 64, 128, 256, 512] ),
            'learning_rate':trial.suggest_loguniform("learning_rate", 5e-4, 1e-3),
            'val_check_steps':25,
            'dropout': trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            'random_seed':trial.suggest_int('random_seed', 1, 20),
        }
    

    def get_auto_fedformer_search_params(self, trial):
        return {
            'max_steps': self.hparams["max_epochs"],
            'input_size': self.hparams["window_size"],
            'futr_exog_list':self.future_exog,
            'hidden_size':trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512] ),
            'conv_hidden_size':trial.suggest_categorical("conv_hidden_size", [16, 32, 64, 128, 256, 512] ),
            'learning_rate':trial.suggest_loguniform("learning_rate", 5e-4, 1e-3),
            'val_check_steps':25,
            'dropout': trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            'random_seed':trial.suggest_int('random_seed', 1, 20),
        }
        
        
    def get_auto_nhits_search_params(self, trial):
        return {
            'max_steps': self.hparams["max_epochs"],
            'input_size': self.hparams["window_size"],
            'futr_exog_list': self.future_exog,
            'hist_exog_list': self.future_exog,
            'val_check_steps': 25,
            'n_pool_kernel_size': trial.suggest_categorical("n_pool_kernel_size", [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]),
            'n_freq_downsample': trial.suggest_categorical("n_freq_downsample",[
                [168, 24, 1], [24, 12, 1], [180, 60, 1], [60, 8, 1], [40, 20, 1], [1, 1, 1]]),
            'windows_batch_size': trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            'learning_rate': trial.suggest_loguniform("learning_rate", 5e-4, 1e-3),
            'random_seed':trial.suggest_int('random_seed', 1, 20),
        }
    
    def get_auto_nbeats_search_params(self, trial):
        return {
            'max_steps': self.hparams["max_epochs"],
            'input_size': self.hparams["window_size"],
            'windows_batch_size': trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
            'val_check_steps':25,
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
            'random_seed':trial.suggest_int('random_seed', 1, 20),
        }
        
    def get_auto_patchtst_search_params(self, trial):
        return {
            'max_steps':self.hparams["max_epochs"],
            'input_size':self.hparams["window_size"],
            'patch_len':trial.suggest_categorical("patch_len", [16, 24, 32]),
            'n_heads':trial.suggest_categorical("n_heads", [4, 8, 16]),
            'revin':False,
            'val_check_steps':25,
            "activation":trial.suggest_categorical("activation", ['ReLU','GeLU']),
            'hidden_size':trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512] ),
            'learning_rate': trial.suggest_loguniform("learning_rate", 5e-4, 1e-3),
            'dropout': trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            'random_seed':trial.suggest_int('random_seed', 1, 20),
        }
    
    def get_auto_lstm_search_params(self, trial):
        return {
            'max_steps':self.hparams["max_epochs"],
            'input_size':self.hparams["window_size"],
            'futr_exog_list':self.future_exog,
            'encoder_hidden_size': trial.suggest_categorical("encoder_hidden_size", [50, 100, 200, 300]),
            'encoder_n_layers': trial.suggest_int('encoder_n_layers', 1, 4),
            'context_size': trial.suggest_categorical("context_size", [5, 10, 50]),
            'decoder_hidden_size':trial.suggest_categorical("decoder_hidden_size", [64, 128, 256, 512]),
            'decoder_layers':trial.suggest_categorical("decoder_layers", [2, 4, 8]),
            # 'dropout': trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
            'random_seed':trial.suggest_int('random_seed', 1, 20),
        }
    
    
    def get_nbeats_search_params(self, trial, params):
        num_stacks=trial.suggest_int('num_stacks', 2, 10)
        params.update({'num_stacks': num_stacks})
        num_blocks=trial.suggest_int('num_blocks', 1, 10)
        params.update({'num_blocks': num_blocks})
        

        generic_architecture =trial.suggest_categorical("generic_architecture", [False, True])
        params.update({'generic_architecture': generic_architecture})

        activation=trial.suggest_categorical("activation", ['ReLU','RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid'])
        params.update({'activation': activation})

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        
        return params
    
    
    
    
    