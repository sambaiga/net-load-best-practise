# Setting the Standard: Best Practices for Benchmarking Net-Load Forecasting Approaches in Low-Voltage Distribution Substations

This repository is the official implementation of [Setting the Standard: Best Practices for Benchmarking Net-Load Forecasting Approaches in Low-Voltage Distribution Substationse](). 



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
- To begin training the model(s), start by optimizing the hyperparameters of the deep-learning-based forecasting model. This can be achieved by executing the code provided in the [SmartGridcomHyper](https://github.com/feelab-info/net-load-best-practices/blob/main/notebook/SmartGridcomHyper-params.ipynb).
- python experiment.py --dataset pt_dataset --exp_name PT-Benchmark --exp_type short-long --epochs 50

- Once the hyperparameters have been optimized, proceed to train the model for all experiments. This can be done by running the code available in the [SmartGridcomExperiment](https://github.com/feelab-info/net-load-best-practices/blob/main/notebook/SmartGridcomExperiment.ipynb) notebook.


## Evaluation
- After training the models and obtaining the results, the next step is to evaluate those results. You can follow the procedure outlined in the  [SmartGridcom-Results-Analysis](https://github.com/feelab-info/net-load-best-practices/blob/main/notebook/SmartGridcom-Results-Analysis.ipynb) notebook to analyze and evaluate the obtained outcomes.

## Results




