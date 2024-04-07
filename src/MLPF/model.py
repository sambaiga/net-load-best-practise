
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import optuna
import pytorch_lightning as pl
from copy import deepcopy
import torchmetrics
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning import loggers
from pathlib import Path
from .utils import get_latest_checkpoint, inverse_scaling, DictLogger, get_prediction_from_mlpf
from timeit import default_timer
from utils.data import  TimeSeriesDataset, TimeSeriesLazyDataset, DataLoader, TimeseriesDataModule
from utils.data_processing import get_index, add_exogenous_variables, fourier_series_t
from baselines.evaluation import evaluate_point_forecast
from .embending import Rotary
from .layers import FeedForward, activations, FutureEncoder, PastEncoder,  create_linear, MLPBlock, MLPForecastNetwork
torch.set_float32_matmul_precision('high')





class MLPForecastModel(pl.LightningModule):
    
    def __init__(self,  hparams):
        super().__init__()
        self.model = MLPForecastNetwork(hparams=hparams)
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        self.size = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(self.size))
        
        self.tra_metric_fcn=torchmetrics.MeanAbsoluteError()
        self.val_metric_fcn=torchmetrics.MeanAbsoluteError()

        self.save_hyperparameters()
        self.hparams.update(hparams)
        
    def forecast(self, x):
        return self.model.forecast(x)
    
    def training_step(self, batch, batch_idx):
        
        loss, metric = self.model.step(batch, self.tra_metric_fcn)
        self.log("train_loss",loss, prog_bar=True, logger=True)
        self.log("train_mae",metric, prog_bar=True, logger=True)

        return loss
            
    
    def validation_step(self, batch, batch_idx):
        
        loss, metric = self.model.step(batch, self.val_metric_fcn) 
        self.log("val_loss",loss, prog_bar=True, logger=True)
        self.log("val_mae",metric, prog_bar=True, logger=True)

     

    def configure_optimizers(self):
        p1 = int(0.75 * self.hparams.max_epochs)
        p2 = int(0.9 * self.hparams.max_epochs)

        
        params  = list(self.parameters())
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,  weight_decay=self.hparams.weight_decay)
           
        scheduler  = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[p1, p2], gamma=0.1)
        return [optim], [scheduler]
        
            
        
        

class MLPForecast(object):
    def __init__(self, hparams, exp_name="Tanesco", seed=42, root_dir="../", file_name=None, trial=None, metric='val_mae', wandb=False, rich_progress_bar=True):
 
        self.seed = seed,
        pl.seed_everything(seed, workers=True)
    
    
        self.rich_progress_bar=rich_progress_bar
        self.root_dir=root_dir
        self.file_name=file_name
        self.hparams=hparams
        self.exp_name=exp_name
        self.trial=trial
        self.metric=metric
        self.wandb=wandb
        self._create_folder()
        self._set_up_trainer()
        self.model = MLPForecastModel(hparams)
        

   
    def _create_folder(self):
        self.results_path = Path(f"{self.root_dir}/results/{self.exp_name}/{self.hparams['encoder_type']}/")
        self.logs = Path(f"{self.root_dir}/logs/{self.exp_name}/{self.hparams['encoder_type']}/")
        self.figures = Path(f"{self.root_dir}/figures/{self.exp_name}/{self.hparams['encoder_type']}/")
        self.figures.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        
    def _set_up_trainer(self):
        callback=[]
        if self.trial is not None:
            self.logger = True#DictLogger(self.logs,  version=self.trial.number)
            early_stopping = PyTorchLightningPruningCallback(self.trial, monitor=self.metric)
            file_name = f"{self.trial.number}"
            callback.append(early_stopping)
        else:
            early_stopping = EarlyStopping(monitor=self.metric, min_delta=0.0, patience=int(self.hparams['max_epochs']*0.5), verbose=False, mode="min", check_on_train_epoch_end=True)
            callback.append(early_stopping)
            exp_name=f"{self.exp_name}_{self.hparams['encoder_type']}_{self.file_name}" if self.file_name else f"{self.exp_name}_{self.hparams['encoder_type']}"
            if not self.wandb:
                self.logger  = loggers.TensorBoardLogger(self.logs,  name = exp_name) 
            else:
                self.logger = loggers.WandbLogger(project=exp_name, log_model="all")
                #self.logger.watch(self.model)
                # log gradients and model topology

               
            if self.file_name is not None:
                self.checkpoints = Path(f"../checkpoints/{self.exp_name}/{self.hparams['encoder_type']}/{self.file_name}")
            else:
                self.checkpoints = Path(f"../checkpoints/{self.exp_name}/{self.hparams['encoder_type']}")
        
        
            self.checkpoints.mkdir(parents=True, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(dirpath=self.checkpoints, monitor=self.metric, mode="min", save_top_k=2)   
            callback.append(checkpoint_callback)
            lr_logger = LearningRateMonitor()
            callback.append(lr_logger)
        
        if self.rich_progress_bar:
            progress_bar = RichProgressBar(
                                theme=RichProgressBarTheme(
                                    description="green_yellow",
                                    progress_bar="green1",
                                    progress_bar_finished="green1",
                                    progress_bar_pulse="#6206E0",
                                    batch_progress="green_yellow",
                                    time="grey82",
                                    processing_speed="grey82",
                                    metrics="grey82",
                                )
                            )
        else:
            progress_bar = TQDMProgressBar()
        
        callback.append(progress_bar)
        self.trainer = pl.Trainer(logger = self.logger,
                                #accumulate_grad_batches=0,
                                gradient_clip_val=self.hparams['clipping_value'],
                                max_epochs = self.hparams['max_epochs'],
                                callbacks=callback,
                                accelerator='auto'
                                )
        
        
    def fit(self, train_df, test_df, experiment, test=False):
        datamodule = TimeseriesDataModule(self.hparams, experiment, train_df, test_df, test=test)
        start_time = default_timer()
        print(f"---------------Training started ---------------------------")  
        if self.trial is not None:
            self.trainer.fit(self.model, datamodule.train_dataloader(), datamodule.val_dataloader())
            
            train_walltime = default_timer() - start_time
            print(f'training complete after {train_walltime} s')
            return self.trainer.callback_metrics[self.metric].item()
            #return self.logger.metrics[-1]['val_mae']
        else:
            self.trainer.fit(self.model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=get_latest_checkpoint(self.checkpoints))
            train_walltime = default_timer() - start_time
            print(f'training complete after {train_walltime} s')
            return train_walltime
        
    
    def predict_from_df(self, test_df,  experiment=None,  test=True):
        
        path_best_model = get_latest_checkpoint(self.checkpoints)
        self.model = self.model.load_from_checkpoint(path_best_model)
        start_time = default_timer()
        self.model.eval()  
        pred_df=get_prediction_from_mlpf(test_df, self.model, self.hparams, experiment)
        test_walltime = default_timer() - start_time
        
        
        Ndays = len(pred_df)//self.hparams['horizon']
        loc = pred_df['pred'].values
        target = pred_df['true'].values
        index = pred_df.index.values
        loc = loc.reshape(Ndays,  self.hparams['horizon'], -1)
        target = target.reshape(Ndays,  self.hparams['horizon'], -1)
        index = index.reshape(Ndays,  self.hparams['horizon'], -1)
        
        outputs = {}
        outputs['pred'] = loc
        outputs['index']=index
        outputs['true']=target
        
        outputs['test-time']=test_walltime
        outputs['target-range']=experiment.installed_capacity
        outputs = evaluate_point_forecast(outputs, outputs['target-range'], self.hparams, self.exp_name, file_name=self.file_name, show_fig=False)
        np.save(f"{self.results_path}/{self.file_name}_processed_results.npy", outputs)
        return outputs
        

    def predict(self, test_df,  experiment=None,  test=True):
    
        target, known_features, unkown_features = experiment.get_data(data=test_df)
        index=get_index(test_df, self.hparams, test=test)
        
        test_loader=TimeSeriesDataset(unkown_features,known_features, target, 
                                    window_size=self.hparams['window_size'], horizon=self.hparams['horizon'], 
                                    batch_size=self.hparams['batch_size'], shuffle=False, test=test, drop_last=False).get_loader()
        features, true = test_loader.dataset.tensors
       
        path_best_model = get_latest_checkpoint(self.checkpoints)
        self.model = self.model.load_from_checkpoint(path_best_model)
        start_time = default_timer()
        self.model.eval()  
        self.model.to(features.device)
        outputs = self.model.forecast(features)
        test_walltime = default_timer() - start_time
        outputs['test-time']=test_walltime
        outputs['inputs']= features
        outputs['index']=index[:, self.hparams['window_size']:]
        outputs['true']=true
        
        
        outputs['pred']=inverse_scaling(outputs['pred'], experiment.target_transformer)
        outputs['true']=inverse_scaling(outputs['true'], experiment.target_transformer)
        outputs['target-range']=experiment.installed_capacity
        outputs = evaluate_point_forecast(outputs, outputs['target-range'], self.hparams, self.exp_name, file_name=self.file_name, show_fig=False)
        np.save(f"{self.results_path}/{self.file_name}_processed_results.npy", outputs)
        return outputs
    
    
    def get_search_params(self, trial, params):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.

        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        dropout = {'dropout':trial.suggest_float("dropout", 0.1, 0.9)}
        params.update(dropout)

        activation  = {'activation':trial.suggest_categorical("activation", [0, 1, 2, 3, 4])}
        params.update(activation)
    
        emb_type = {'emb_type':trial.suggest_categorical("emb_type",["None", 'PosEmb', 'RotaryEmb', 'CombinedEmb'])}
        params.update(emb_type)
        
        emb_size = {'emb_size':trial.suggest_categorical("emb_size",[8,  16, 32, 64])}
        params.update(emb_size)
        
        comb_type = {'comb_type':trial.suggest_categorical("comb_type",['attn-comb', 'weighted-comb', 'addition-comb'])}
        params.update(comb_type)
        if comb_type=='attn-comb':
            num_head = {'num_head':trial.suggest_categorical("num_head",[2, 4,  8, 16])}
            params.update(num_head)
        

        
        alpha = {'alpha':trial.suggest_float("alpha", 0.01, 0.9)}
        params.update(alpha)
       
        return params
    
    
    
    def auto_tune_model(self, train_df, val_df, experiment, num_trials=10):
        self.train_df = train_df
        self.validation_df = val_df
        self.experiment= experiment
        
        
        def print_callback(study, trial):
            print(f"Trial No: {trial.number}, Current value: {trial.value}, Current params: {trial.params}")
            print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
            
            
        def objective(trial=None):
            self.hparams =  self.get_search_params(trial, self.hparams)
            model=MLPForecast(self.hparams, exp_name=f"{self.exp_name}", seed=42, trial=trial, rich_progress_bar=True)
            val_cost = model.fit(self.train_df, self.validation_df, self.experiment)
        
            return  val_cost

        study_name=f"{self.exp_name}_{self.hparams['encoder_type']}"
        storage_name = "sqlite:///{}.db".format(study_name)
        base_pruner = pruner=optuna.pruners.SuccessiveHalvingPruner()
        pruner=optuna.pruners.PatientPruner(base_pruner, patience=5)
        study = optuna.create_study( direction="minimize", pruner=pruner,  study_name=self.exp_name, 
                                    storage=storage_name,
                                    load_if_exists=True)
        study.optimize(objective, n_trials=num_trials, callbacks=[print_callback])
        self.hparams.update(study.best_trial.params)
        np.save(f"../results/{self.exp_name}/{self.hparams['encoder_type']}/best_params.npy", self.hparams)
    
    
    

        



        