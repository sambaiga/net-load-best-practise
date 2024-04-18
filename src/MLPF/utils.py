import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
import glob
import os
from timeit import default_timer


def get_prediction_from_mlpf(test_df, model, hparams, experiment):
    """
    Generate predictions using a multi-level prediction model.

    Args:
        test_df (pd.DataFrame): Test dataset.
        model: Multi-level prediction model.
        hparams (dict): Hyperparameters.
        experiment: Experiment details.

    Returns:
        pd.DataFrame: DataFrame containing predictions.

    """
    Ndays = len(test_df) // hparams['horizon']
    all_predictions = []

    for i in range(Ndays):
        # Extract historical and future data for the current day
        first_day_start = i * hparams['horizon']
        first_day_end = i * hparams['horizon'] + hparams['horizon']
        second_day_start = first_day_end
        second_day_end = second_day_start + hparams['horizon']
        historical_data = test_df.iloc[first_day_start:second_day_end]
        future_data = test_df.iloc[second_day_end:second_day_end + hparams['horizon']]

        # Check if the data has the correct length
        if (len(historical_data) != hparams['window_size']) or (len(future_data) != hparams['horizon']):
            continue

        # Prepare input data for the model
        numerical_columns = [f + "_target" for f in hparams['targets']] + hparams['time_varying_unknown_feature']
        covariate_columns = hparams['time_varying_known_feature'] + experiment.seasonality_columns
        unknown_features = historical_data[numerical_columns + covariate_columns].values.astype(np.float64)
        target = future_data[[f + "_target" for f in hparams['targets']]].values.astype(np.float64)
        known_features = future_data[covariate_columns].values.astype(np.float64)
        features = torch.FloatTensor(unknown_features).unsqueeze(0)
        covariates = torch.FloatTensor(known_features).unsqueeze(0)

        # Pad covariate features with zero
        diff = features.shape[2] - covariates.shape[2]
        B, N, _ = covariates.shape
        diff = torch.zeros(B, N, diff, requires_grad=False)
        covariates = torch.cat([diff, covariates], dim=-1)
        features = torch.cat([features, covariates], dim=1)
        model.to(features.device)

        # Make predictions and measure wall time
        start_time = default_timer()
        out = model.forecast(features)
        test_walltime = default_timer() - start_time

        # Inverse scaling of predictions and true values
        out['pred'] = inverse_scaling(out['pred'], experiment.target_transformer).flatten()
        out['true'] = experiment.target_transformer.inverse_transform(target).flatten()

        # Convert the output to a DataFrame and set the timestamp index
        out = pd.DataFrame.from_dict(out, orient='index').T
        out.index = future_data.timestamp
        out['test_walltime'] = test_walltime

        # Append the predictions to the list
        all_predictions.append(out)

        # Clean up variables
        del covariates
        del diff

    # Concatenate all predictions into a single DataFrame
    all_predictions = pd.concat(all_predictions)

    return all_predictions



def inverse_scaling(target, scaler):
    """
    Inverse scaling for the target using a specified scaler.

    Args:
        target (torch.Tensor): Scaled target tensor.
        scaler: Scaler object used for scaling.

    Returns:
        torch.Tensor: Inversely scaled target tensor.

    """
    B, T, C = target.shape
    target = scaler.inverse_transform(target.numpy().reshape(B * T, C))
    return target.reshape(B, T, C)


def get_latest_checkpoint(checkpoint_path):
    """
    Get the path of the latest checkpoint file in a directory.

    Args:
        checkpoint_path (str): Path to the directory containing checkpoint files.

    Returns:
        str or None: Path to the latest checkpoint file or None if no checkpoint is found.

    """
    checkpoint_path = str(checkpoint_path)
    list_of_files = glob.glob(checkpoint_path + '/*.ckpt')

    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file


class DictLogger(pl.loggers.TensorBoardLogger):
    """
    PyTorch Lightning `dict` logger.

    Subclass of `TensorBoardLogger` with additional functionality to store metrics as a list.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []

    def log_metrics(self, metrics, step=None):
        """
        Log metrics and store them in the metrics list.

        Args:
            metrics (dict): Dictionary of metrics.
            step (int, optional): Step at which the metrics are logged. Default is None.

        """
        super().log_metrics(metrics, step=step)
        self.metrics.append(metrics)