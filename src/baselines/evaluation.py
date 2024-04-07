import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from utils.metrics import  get_pointwise_metrics
from utils.visual_functions import plot_prediction_with_scale
from utils.visual_functions import  plot_prediction_with_upper_lower
from utils.visual_functions import  plot_prediction_with_pi
import matplotlib_inline.backend_inline
import matplotlib.pyplot as plt
import arviz as az
import matplotlib.dates as mdates
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")
az.style.use(["science", "grid", "arviz-doc", 'tableau-colorblind10'])

def evaluate_point_forecast(outputs, target_range, hparams, exp_name, file_name,  ghi_dx=1, show_fig=False):
    
    pd_metrics, spilit_metrics = {}, {}
    logs = {}
    for j in range(outputs['true'].shape[-1]):
        metrics=[]
        for i in range(0, len(outputs['true'])):
         
            true = outputs['true'][i,:, j]
            pred = outputs['pred'][i,:, j]
            
            R = target_range[j]
            t_nmpic = true.std()/R
            
            df = pd.DataFrame(outputs['index'][i])
            df.columns=['Date']
            index=df.Date.dt.round("D").unique()[-1]
            point_scores = get_pointwise_metrics(pred, true, R, 1)
            point_scores =pd.DataFrame.from_dict(point_scores, orient='index').T
            point_scores['timestamp']=index
            metrics.append(point_scores)

        metrics = pd.concat(metrics)
        outputs[f"{hparams['targets'][j]}_metrics"]=metrics
        print(f"Results for {hparams['targets'][j]}")
        print(pd.DataFrame(metrics.median()).T[[  'mae', 'nrmse',  'corr',  'nbias']].round(3))

        bad=np.where(metrics['mae']==metrics['mae'].max())[0][0]
        good=np.where(metrics['mae']==metrics['mae'].min())[0][0]
        outputs[f"{hparams['targets'][j]}_bad"]=bad
        outputs[f"{hparams['targets'][j]}_good"]=good
        
        colors = ['C0', 'C3', 'C5']
        fig, ax = plt.subplots(1,2, figsize=(9,2))
        ax = ax.ravel()
        ax[0], lines, label=plot_prediction_with_upper_lower(ax[0],outputs['true'][good][:, j].flatten(),
                                     outputs['pred'][good][:, j].flatten(),
                                    None,
                                    None)
        
        

        met=metrics[['nrmse', 'mae',  'corr']].iloc[good].values
        ax[0].set_title("Good Day NMRSE: {:.2g}, \n MAE: {:.3g}, CORR: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
        leg = ax[0].legend(lines, label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        ax[1],lines, label=plot_prediction_with_upper_lower(ax[1], outputs['true'][bad][:, j].flatten(),
                                     outputs['pred'][bad][:, j].flatten(),
                                    None,
                                    None)
        
        met=metrics[['nrmse', 'mae',  'corr']].iloc[bad].values
        ax[1].set_title("Bad Day NMRSE: {:.2g}, \n MAE: {:.3g}, CORR: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
        leg = ax[1].legend(lines, label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        
        fig.tight_layout(pad=1.08, h_pad=0.5, w_pad=0.5)
        fig.savefig(f"../figures/{exp_name}/{hparams['encoder_type']}/{file_name}_{hparams['targets'][j]}_results.pdf", dpi=480)
        if not show_fig:
            plt.close()


   
    outputs["targets_range"]=target_range
    return outputs 

