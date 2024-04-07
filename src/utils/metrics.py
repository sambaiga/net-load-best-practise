import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score
import torch
import pandas as pd
from scipy import stats
def _divide_no_nan(a: float, b: float) -> float:
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div



def get_bias(y, y_hat):
    bias=(y - y_hat)/get_mae(y, y_hat)
    return bias

def get_residual(y, y_hat):
    res=(y - y_hat)
    return res

def get_nbias(y, y_hat):
    nbias=(y-y_hat)/(y + y_hat)
    return nbias


def get_mae(y, y_hat):
    return np.abs(y - y_hat)

def get_mse(y, y_hat):
    return np.square(y - y_hat)

def get_rmse(y, y_hat):
    return np.sqrt(get_mse(y, y_hat))

def get_mape(y, y_hat):
    delta_y = get_mae(y, y_hat)
    scale = np.abs(y)
    mape=_divide_no_nan(delta_y, scale)
    return mape

def get_smape(y, y_hat):
    delta_y = get_mae(y, y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = 2 *_divide_no_nan(delta_y, scale)
    return smape



def get_bias(y, y_hat):
    """
    Tracking Signal quantifies “Bias” in a forecast. No product can be planned from a severely biased forecast. Tracking Signal is the gateway test for evaluating forecast accuracy.
    Once this is calculated, for each period, the numbers are added to calculate the overall tracking signal. A forecast history entirely void of bias will return a value of zero
    """
    norm= (y - y_hat)
    denom=get_mae(y, y_hat)
    return  _divide_no_nan(norm, denom)
    

def get_nbias(y, y_hat):
    """
    this metric will stay between -1 and 1, with 0 indicating the absence of bias. Consistent negative values indicate a tendency to under-forecast whereas constant positive values indicate a tendency to over-forecast
    if the added values are more than 2, we consider the forecast to be biased towards over-forecast. Likewise, if the added values are less than -2, we find the forecast to be biased towards under-forecast.
    """
    norm= (y - y_hat)
    denom=(y + y_hat)
    return  _divide_no_nan(norm, denom)
    

def get_pointwise_metrics(pred:np.array, true:np.array, target_range:float=None, scale=1):
    """calculate pointwise metrics
    Args:   pred: predicted values
            true: true values
            target_range: target range          
    Returns:    rmse: root mean square error                


    """
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"
    target_range = true.max() - true.min() if target_range is None else target_range
    
    rmse = np.sqrt(mean_squared_error(true, pred))
    nrmse =min( rmse/target_range, 1)
    mae = mean_absolute_error(true, pred)/scale
    res =np.mean(true-pred)
    
    bias=get_bias(true, pred).sum()
    nbias=get_nbias(true, pred).sum()
    corr = np.corrcoef(true, pred)[0, 1]
    r2=r2_score(pred, true)
    smape=np.mean(get_smape(true, pred))
    mape=np.mean(get_mape(true, pred))
    return dict(nrmse=nrmse, mae=mae, r2=r2,
                corr=corr, rmse=rmse,  bias=bias, nbias=nbias, mape=mape, smape=smape, res=res)


def get_daily_pointwise_metrics(pred:np.array, true:np.array, target_range:float):
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"

    #get pointwise metrics
    metrics = get_pointwise_metrics(pred, true, target_range)
    metrics =pd.DataFrame.from_dict(metrics, orient='index').T
    return metrics
    





