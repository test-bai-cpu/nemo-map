
# Run the experiments


## Setup environment
``` bash
conda env create -f environment.yml
conda activate mod
```


## For training
``` bash
python3 train.py  --model siren
```
The argument --model can be one of ["siren", "time_grid", "fourier"]. In the paper, we use siren as the main model.
For the ablation study, we provide two alternative temporal encodings: time grid and Fourier features.

The training runs for 100 epochs, and models are saved in `models/distri_gmm_siren`. For time_grid and fourier version, trained models are also provided `models/distri_gmm_feature_ff_time`, and `models/distri_gmm_feature_time`.

## For evaluation
``` bash
python3 evaluate_NLL.py --model siren
```
After training the model, we can evaluate it by computing the Negative Log Likelihood (NLL) value. Same here, the arg can be chosen from ["siren", "time_grid", "fourier"]. Detailed NLL results for each test sample will be saved in `nll_results/distri_gmm_siren/atc-all.csv`.


## For querying MoDs

``` bash
python3 generate_MoD_files.py --model siren
```
We can query the trained model to generated maps of dynamics for each hour of the ATC dataset. The generated MoDs are saved in `MoDs/distri_gmm_siren/<hour>.csv`


## For plotting
``` bash
python3 plot_MoD_files.py --model siren --version max
```
We can also plot the generated MoDs. Two version of plotting are provided. 
- Version `all` shows multimodality by rendering all SWGMM components with transparency proportional to their weights. 
- Version `max` more clearly shows the dominant flow, only displaying the mixture component with the largest weight.

The generated MoD figures are saved in `MoDs/distri_gmm_siren/all_png` and `MoDs/distri_gmm_siren/max_png` folders.