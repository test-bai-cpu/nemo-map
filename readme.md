<div align="center">

<h1>NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping</h1>

[Yufei Zhu](https://test-bai-cpu.github.io/index.html) · [Shih-Min Yang](https://shihminyang.github.io/) · [Andrey Rudenko](https://scholar.google.com/citations?user=BlSfMYwAAAAJ&hl=en) · [Tomasz P. Kucner](https://scholar.google.com/citations?user=ByjVNHoAAAAJ&hl=en) · [Achim J. Lilienthal](https://www.ce.cit.tum.de/pins/team/achim-j-lilienthal/) · [Martin Magnusson](https://scholar.google.com/citations?user=s9fPUg8AAAAJ&hl=en)

[RNP@ORU](https://www.oru.se/english/research/groups/nt/robot-navigation-and-perception-lab/)
</div>


The implementation of ICLR'26 paper: "NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping"

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

Make sure Git LFS is installed. After cloning the repository, run `git lfs install` and `git lfs pull` from the repository directory to download the files stored with Git LFS:

- `atc/`, `eth_ucy/`: datasets
- `models/`: trained model weights
- `nll_results/`: negative log-likelihood (NLL) evaluation results
- `MoDs/`: generated maps of dynamics (MoDs)


### Setup environment
``` bash
conda env create -f environment.yml
conda activate mod
```

### For training

To train the main SIREN model on ATC using the default dataset configuration:

```bash
python3 train.py --dataset ATC --model siren
```

 - The `--dataset` argument accepts `ATC` (default), `ETH-eth`, `ETH-hotel`, `UCY-students003`, or `UCY-zara01`. Per-dataset settings, including normalization bounds, batch size, number of mixture components, grid size, and SIREN variant, are defined in `dataset_config.yaml`. The default ATC configuration uses 3 components and a 64×64 grid.

- The `--model` argument accepts `siren` (default), `time_grid`, or `fourier`. SIREN is the main model used in the paper; time grid and Fourier features are alternative temporal encodings compared in the ablation study in Table 5.

Output folders are named `nemo_<scene>` for SIREN, with a `_time_grid` or `_fourier` suffix for the other models. The command above saves checkpoints to `models/nemo_atc/` and TensorBoard logs to `runs/nemo_atc/<timestamp>/`.

Use either or both of the following optional arguments to override the dataset configuration:

- `--num-components`: number of SWGMM mixture components.
- `--grid-size`: side length of the square spatial feature grid; for example, `32` creates a 32×32 grid.

For example, to train with 3 components and a 32×32 grid:

```bash
python3 train.py --model siren --dataset ATC --num-components 3 --grid-size 32
```

Training requires a CUDA-capable GPU and runs for 100 epochs. The checkpoint with the lowest validation loss is saved as `best.pt` in the corresponding model folder. Pretrained models are provided under `models/`, including `models/nemo_atc/` and `models/nemo_eth/`.

### For evaluation

Evaluate a trained ATC model on the test split using negative log-likelihood (NLL):

```bash
python3 evaluate_NLL.py --dataset ATC
python3 evaluate_NLL.py --dataset ETH-eth
python3 evaluate_NLL.py --dataset ETH-hotel
python3 evaluate_NLL.py --dataset UCY-students003
python3 evaluate_NLL.py --dataset UCY-zara01
```

### For querying MoDs

``` bash
python3 generate_MoD_files.py
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


## Correction of Reported NLL Values

In an earlier version of the paper, the reported negative log-likelihood (NLL) values were affected by an error in our implementation of the semi-wrapped normal density. All affected numbers have been corrected in the revised version of the paper.

The initial implementation shifted the observed orientation by $2\pi k$ first and wrapped the angular residual into $[-\pi, \pi)$ afterwards. Since wrapping cancels the shift, the three winding terms were identical, and the implemented density was three times the nearest winding term rather than the sum in Eq. (2) of the paper. As a result, the reported NLL values were lower than the correct values.

The error affected (i) the training objective of NeMo-map, including all ablation variants, (ii) the evaluation of NeMo-map, and (iii) the evaluation of CLiFF-map and Online CLiFF-map, which share the SWGMM likelihood code. STeF-map represents orientation as a histogram and does not use the winding sum, so it was unaffected by this implementation error.

We retrained all NeMo-map models (the main model, the temporal encoding variants in Section 4.5, and the hyperparameter ablations in Appendix E of the paper) with the corrected loss, using the same architecture, hyperparameters, and random seed. CLiFF-map and Online CLiFF-map were re-evaluated with the corrected density.

Separately, we updated the NLL computation for STeF-map. STeF-map models orientation only, so its NLL is computed over orientation. Its NLL was previously computed from the bin probabilities. We now divide these probabilities by the bin width ($2\pi/8$) to obtain a density. This changes its reported NLL values and ensures that all methods use densities rather than mixing densities and bin probabilities; STeF-map's NLL remains orientation-only.

We thank Iacopo Catalano (University of Turku) for identifying this error and for helpful discussions on the evaluation protocol.