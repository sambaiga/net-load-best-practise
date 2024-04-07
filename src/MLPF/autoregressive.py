from tqdm import tqdm
import torch
import numpy as np
import pandas as  pd
import sys
from .utils import inverse_scaling


def get_autoregressive_forecast_from_MLPF(model, initial_historical_df, covariate_df, hparams, experiment,  max_days=7):
    """
    Get autoregressive forecasts using a Multi-Layer Perceptron Forecasting (MLPF) model.

    Parameters:
    - model: The MLPF model for forecasting.
    - initial_historical_df: DataFrame containing initial historical data.
    - covariate_df: DataFrame containing covariate data.
    - hparams: Dictionary of hyperparameters for the model.
    - experiment: An experiment object containing information about the forecasting task.
    - max_days: Maximum number of days for forecasting (default is 7).

    Returns:
    - predictions: DataFrame containing the autoregressive forecasts for each day.
    """


    # Prepare initial historical data for forecasting
    p_days=[]
    for i in range(0, hparams['window_size'], hparams['horizon']):
        p=initial_historical_df[i:hparams['horizon']+i].values
        p_days.append(p)
                
    day_1=p_days[0]
    day_2=p_days[1]
    
    
    predictions=[]
    # Iterate over the specified number of forecast days
    with tqdm(total=max_days, file=sys.stdout) as pbar:
        for k in range(0, max_days):
            first_day_start = k*hparams['horizon']
            first_day_end = k*hparams['horizon']+hparams['horizon']
                
            second_day_start = first_day_end
            second_day_end = second_day_start + hparams['horizon']
            historical_data = covariate_df.iloc[first_day_start:second_day_end]
            
            future_data =covariate_df.iloc[second_day_end:second_day_end+hparams['horizon']]
            
            historical_target=np.concatenate([day_1, day_2], axis=0)
            
            unkown_features = np.concatenate([historical_target, historical_data.values], axis=1).astype(np.float64) 
            
            known_features =  future_data.values.astype(np.float64)
            
            
            features = torch.FloatTensor(unkown_features).unsqueeze(0)
            covariates = torch.FloatTensor(known_features).unsqueeze(0)
            
            
            #padd covariate features with zero
            diff = features.shape[2] - covariates.shape[2]
            B, N, _ = covariates.shape
            diff = torch.zeros(B, N, diff, requires_grad=False)
            covariates = torch.cat([diff, covariates], dim=-1)
            features = torch.cat([features, covariates], dim=1)
            
            model.to(features.device)
            out=model.forecast(features)
            
            
            day_1=day_2
            day_2=out['pred'].numpy().reshape(-1, 1)
            out['pred']=inverse_scaling(out['pred'], experiment.target_transformer).flatten()
            out =pd.DataFrame.from_dict(out, orient='index').T
            predictions.append(out)
            
            del covariates
            del diff
            pbar.set_description('processed: %d' % (k))
            pbar.update(1)
        pbar.close()
        predictions=pd.concat(predictions)
    return predictions


def run_autoregressive_k_days(test_df, model, hparams, experiment, num_days=7):
    """
    Run autoregressive k-days forecasting on the given test dataset using an MLPF model.

    Parameters:
    - test_df: DataFrame containing the test dataset.
    - model: The MLPF model for forecasting.
    - hparams: Dictionary of hyperparameters for the model.
    - experiment: An experiment object containing information about the forecasting task.
    - num_days: Number of days to forecast (default is 7).

    Returns:
    - all_predictions: DataFrame containing the predictions for each k-days window.
    """
    
    all_predictions = []
    k = num_days + 2

    # Loop through the test dataset to produce k-days forecast
    for j in range(0, len(test_df), k * hparams['horizon']):
        # Extract windows of data
        w = slice(0 + j, j + k * hparams['horizon'])
        initial_df = test_df[w][[f"NetLoad_target"]].iloc[:hparams['window_size']]
        covariate_columns = hparams['time_varying_known_feature'] + experiment.seasonality_columns
        covariate_df = test_df[w][covariate_columns]

        # Skip if the window size or covariate length is not as expected
        if (len(initial_df) != hparams['window_size']) or (len(covariate_df) != k * hparams['horizon']):
            continue

        # Get autoregressive forecasts using the MLPF specified model
        predictions = get_autoregressive_forecast_from_MLPF(model, initial_df, covariate_df, hparams, experiment, max_days=num_days)
        
        # Extract true values from the test dataset
        true = test_df[w][['timestamp'] + hparams['targets']].iloc[hparams['window_size']:][:num_days * hparams['horizon']]

        # Set index and add true values to predictions DataFrame
        predictions.index = true['timestamp']
        predictions[f"true"] = true[hparams['targets']].values

        # Add the K-days information to predictions
        predictions[f"K-days"] = num_days
        all_predictions.append(predictions)

    return pd.concat(all_predictions)