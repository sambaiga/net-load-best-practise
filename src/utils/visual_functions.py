import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.dates as mdates
from matplotlib.dates import (rrulewrapper, RRuleLocator, drange, DayLocator, HourLocator, DateFormatter)
from statsmodels.graphics.api import qqplot#
import math
colors = [plt.cm.Blues(0.6), plt.cm.Reds(0.4), '#99ccff', '#ffcc99', plt.cm.Greys(0.6), plt.cm.Oranges(0.8), plt.cm.Greens(0.6), plt.cm.Purples(0.8)]
SPINE_COLOR="gray"
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import arviz as az
az.style.use(["science", "grid", "arviz-doc", 'tableau-colorblind10'])
import pandas as pd

nice_fonts = {
        # Use LaTeX to write all text
        "font.family": "serif",
        # Always save as 'tight'
        "savefig.bbox" : "tight",
        "savefig.pad_inches" : 0.05,
        "ytick.right" : True,
        "font.serif" : "Times New Roman",
        "mathtext.fontset" : "dejavuserif",
        "axes.labelsize": 15,
        "font.size": 15,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # Set line widths
        "axes.linewidth" : 0.5,
        "grid.linewidth" : 0.5,
        "lines.linewidth" : 1.,
        # Remove legend frame
        "legend.frameon" : False
}
#matplotlib.rcParams.update(nice_fonts)

def set_figure_size(fig_width=None, fig_height=None, columns=2):
    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES
    return (fig_width, fig_height)


def format_axes(ax):
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)
    return ax

def figure(fig_width=None, fig_height=None, columns=2):
    """
    Returns a figure with an appropriate size and tight layout.
    """
    fig_width, fig_height =set_figure_size(fig_width, fig_height, columns)
    fig = plt.figure(figsize=(fig_width, fig_height))
    return fig



def legend(ax, ncol=3, loc=9, pos=(0.5, -0.1)):
    leg=ax.legend(loc=loc, bbox_to_anchor=pos, ncol=ncol)
    return leg

def savefig(filename, leg=None, format='.eps', *args, **kwargs):
    """
    Save in PDF file with the given filename.
    """
    if leg:
        art=[leg]
        plt.savefig(filename + format, additional_artists=art, bbox_inches="tight", *args, **kwargs)
    else:
        plt.savefig(filename + format,  bbox_inches="tight", *args, **kwargs)
    plt.close()




def plot_prediction_with_pi(ax, true, mu, q_pred=None, date=None, true_max=None):
  
    date = np.arange(len(true)) if date is None else date
    h1 = ax.plot(date, true, ".", mec="#ff7f0e", mfc="None")
    h2 = ax.plot(date, mu,   '--',  c="#1f77b4", alpha=0.8)
    ax.set_ylabel('Power $(W)$')

    if q_pred is not None:
        N = q_pred.shape[0]
        alpha = np.linspace(0.1, 0.9, N//2).tolist() + np.linspace(0.9, 0.2, 1+N//2).tolist()
        
        for i in range(N):
            y1 = q_pred[i, :]
            y2 = q_pred[-1-i, :]
            h3 = ax.fill_between(date, y1.flatten(), y2.flatten(), color="lightsteelblue", alpha=alpha[1])
    ax.autoscale(tight=True)
    if true_max is None:
        true_max = true.max()


    
    if q_pred is not None:
        lines =[h1[0], h2[0], h3]
    else:
         lines =[h1[0], h2[0]]
    label = ["True", "Pred", "CONF"]
    
    
    return ax, lines, label

def plot_prediction_with_scale(ax, true, mu, scale=None, date=None, true_max=None):
  
    date = np.arange(len(true)) if date is None else date
    h1 = ax.plot(date, true, ".", mec="#ff7f0e", mfc="None")
    h2 = ax.plot(date, mu,   '--',  c="#1f77b4", alpha=0.8)
    ax.set_ylabel('Power $(W)$')

   
    h3 = ax.fill_between(date, (mu-scale).flatten(), (mu+scale).flatten(), color="lightsteelblue")
    ax.autoscale(tight=True)
    if true_max is None:
        true_max = true.max()


    
   
    lines =[h1[0], h2[0], h3]
    
    label = ["True", "Pred", "CONF"]
    
    
    return ax, lines, label

def visualize_results(outputs, id=0, j=0,  metrics=None):
    colors = ['C0', 'C3', 'C5']
    fig, ax = plt.subplots(2,2, figsize=(8,3))
    ax = ax.ravel()
    met=metrics[['nrmse', 'ciwe', 'ncrps',  'corr']].mean().values
   
    
       
    min_y=min(0, outputs['loc'][id][:, j].min(), outputs['true'][id][:, j].min())
    max_y=max(outputs['loc'][id][:, j].max(), outputs['true'][id][:, j].max())
   
                    
    ax[0].set_title("Uncalibrated NMRSE: {:.2g}, CWE: {:.3g}, CRPS: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
    ax[0].plot(outputs['loc'][id][:, j],'-',  c="#1f77b4", alpha=1, label="pred")
    ax[0].plot(outputs['true'][id][:, j],  ".",  mec="#ff7f0e", mfc="None", label="True")
    ax[0].set_ylabel('Power')
    ax[0].set_ylim(min_y, max_y)
    leg = ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    N=len((outputs['loc'][id]))
    ax[0].fill_between(np.arange(N), 
                       outputs['lower'][id][:, j], 
                       outputs['upper'][id][:, j],  
                       color="lightsteelblue", alpha=0.5, label="CONF")

    ax[1].set_title('Calibrated')
    ax[1].plot(outputs['loc'][id][:, j],'-',  c="#1f77b4", alpha=1, label="pred")
    ax[1].plot(outputs['true'][id][:, j],  ".",  mec="#ff7f0e", mfc="None", label="True")
    ax[1].fill_between(np.arange(N), 
                       outputs['lower_calib'][id][:, j], 
                       outputs['upper_calib'][id][:, j],  
                       color="lightsteelblue", alpha=0.5, label="90%")

    ax[1].set_ylabel('Power')
    ax[1].set_ylim(min_y, max_y)
    leg = ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    #ax[2].plot(outputs['index'][id][-N:], outputs['inputs'][id][-N:,3], '.', mec="#ff7f0e", mfc="None", label="Ghi")
    ax[2].plot(outputs['index'][id][-N:], outputs['epistemic'][id],  ".",  mfc="None", label="EP")
    hfmt = mdates.DateFormatter('%d %H')
    ax[2].xaxis.set_major_formatter(hfmt)
    plt.setp( ax[2].xaxis.get_majorticklabels(), rotation=90 );
    
    UNC=outputs['epistemic'][id]+outputs['aelotoric'][id]
    ax[3].plot(outputs['index'][id][-N:], outputs['aelotoric'][id],  ".",  mfc="None", label="AE")
    ax[3].xaxis.set_major_formatter(hfmt)
    plt.setp( ax[3].xaxis.get_majorticklabels(), rotation=90 );
    ax[3].set_ylabel('AE-UN')
    leg = ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout(pad=1.08, h_pad=0.5, w_pad=0.5)

    return fig, ax




def plot_prediction_with_upper_lower(ax, true, pred, lower, upper, date=None, true_max=None):
  
    date = np.arange(len(true)) if date is None else date
    h1 = ax.plot(date, true, ".", mec="#ff7f0e", mfc="None")
    h2 = ax.plot(date, pred,   '.-',  c="#1f77b4", alpha=0.9)
    ax.set_ylabel('Power $(W)$')

   
    #h3 = ax.fill_between(date, lower.flatten(), upper.flatten(), color="lightsteelblue")
    ax.autoscale(tight=True)
    if true_max is None:
        true_max = true.max()


    
   
    lines =[h1[0], h2[0]]
    
    label = ["True", "Pred"]
    
    
    return ax, lines, label
