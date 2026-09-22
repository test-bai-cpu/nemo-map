<div align="center">

<h1>NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping</h1>

[Yufei Zhu](https://test-bai-cpu.github.io/index.html) · [Shih-Min Yang](https://shihminyang.github.io/) · [Andrey Rudenko](https://scholar.google.com/citations?user=BlSfMYwAAAAJ&hl=en) · [Tomasz P. Kucner](https://scholar.google.com/citations?user=ByjVNHoAAAAJ&hl=en) · [Achim J. Lilienthal](https://www.ce.cit.tum.de/pins/team/achim-j-lilienthal/) · [Martin Magnusson](https://scholar.google.com/citations?user=s9fPUg8AAAAJ&hl=en)

[RNP@ORU](https://www.oru.se/english/research/groups/nt/robot-navigation-and-perception-lab/)
</div>


The **official** implementation of ICLR'26 paper: "NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping"

## 📝 Overview
Safe and efficient robot operation in complex human environments can benefit
from good models of site-specific motion patterns. Maps of Dynamics (MoDs)
provide such models by encoding statistical motion patterns in a map, but existing
representations use discrete spatial sampling and typically require costly offline
construction. We propose a continuous spatio-temporal MoD representation based
on implicit neural functions that directly map coordinates to the parameters of a
Semi-Wrapped Gaussian Mixture Model. This removes the need for discretization and imputation for unevenly sampled regions, enabling smooth generalization across both space and time. Evaluated on two public datasets with real-world
people tracking data, our method achieves better accuracy of motion representation and smoother velocity distributions in sparse regions while still being computationally efficient, compared to available baselines. The proposed approach
demonstrates a powerful and efficient way of modeling complex human motion
patterns and high performance in the trajectory prediction downstream task.

## 🛠️ Run the experiment

### Clone the code
after `git clone`, also do `git lfs pull`, to pull the large data files in `atc/`


### Setup environment
``` bash
conda env create -f environment.yml
conda activate mod
```

### For training
``` bash
python3 train.py  --model siren
```
The argument --model can be one of ["siren", "time_grid", "fourier"]. In the paper, we use siren as the main model.
For the ablation study, we provide two alternative temporal encodings: time grid and Fourier features.

The argument --dataset selects the data: `ATC` (default) or one of `ETH-eth`, `ETH-hotel`, `UCY-students001`, `UCY-students003`, `UCY-zara01`. Per-dataset settings (normalization bounds, batch size, grid size, model variant) live in `dataset_config.yaml`.
Output folders are named `nemo_<scene>` for siren, with a `_time_grid` or `_fourier` suffix for the other two models (e.g. `models/nemo_atc`, `models/nemo_eth_fourier`).

The training runs for 100 epochs, and models are saved in `models/nemo_atc`. For time_grid and fourier version, trained models are also provided `models/nemo_atc_time_grid`, and `models/nemo_atc_fourier`.

### For evaluation
``` bash
python3 evaluate_NLL.py --model siren
```
After training the model, we can evaluate it by computing the Negative Log Likelihood (NLL) value. Same here, the arg can be chosen from ["siren", "time_grid", "fourier"]. Detailed NLL results for each test sample will be saved in `nll_results/nemo_atc/atc-all.csv`.


### For querying MoDs

``` bash
python3 generate_MoD_files.py --model siren
```
We can query the trained model to generated maps of dynamics for each hour of the ATC dataset. The generated MoDs are saved in `MoDs/nemo_atc/<hour>.csv`


### For plotting
``` bash
python3 plot_MoD_files.py --model siren --version max
```
We can also plot the generated MoDs. Two version of plotting are provided. 
- Version `all` shows multimodality by rendering all SWGMM components with transparency proportional to their weights. 
- Version `max` more clearly shows the dominant flow, only displaying the mixture component with the largest weight.

The generated MoD figures are saved in `MoDs/nemo_atc/all_png` and `MoDs/nemo_atc/max_png` folders.
### ATC SIREN component/grid sweep

Override the dataset configuration for an individual training run:

```bash
python train.py --dataset ATC --model siren --num-components 3 --grid-size 32
```

This saves checkpoints to `models/nemo-atc-com3-grid32/` and TensorBoard logs
under `runs/nemo-atc-com3-grid32/`. If either override is supplied, the folder
name includes both effective values. Without overrides, the existing config
and experiment name are used.

Run all component counts 1–15 and grid sizes 8, 16, 32, 64, 128, 256:

```bash
bash run_atc_siren_sweep.sh
```
