############### For ATC dataset ###############


import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import numpy as np
import scipy.stats as stats
from pathlib import Path


def paired_ci(a, b, alpha=0.05):
    d = np.array(a) - np.array(b)
    n = len(d)
    mean_diff = np.mean(d)
    se = stats.sem(d, ddof=1)  # standard error of mean difference
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    return mean_diff, (ci_low, ci_high)

def cohens_dz(a, b):
    d = np.asarray(a) - np.asarray(b)
    return d.mean() / d.std(ddof=1)

all_rows = []


############# for ATC dataset - hour #############
# for hour in range(9, 21):
#     # res_file_1 = f"nll_results/cliff/cliff-map-hours/cliff-map-{hour}.csv"
#     # res_file_1 = f"nll_results/stef/stef_thres1/stef-map-{hour}.csv"
#     res_file_1 = f"nll_results/online/ATC1024_{hour}_{hour+1}_online_online.csv"
    
#     # res_file_1 = f"nll_results/NIR/distri_gmm_feature_ff_time_exp100/atc-{hour}.csv"
#     # res_file_1 = f"nll_results/NIR/distri_gmm_feature_time_exp100/atc-{hour}.csv"
    
#     res_file_2 = f"nll_results/NIR/distri_gmm_siren_exp100/atc-{hour}.csv"

#     # Load both files
#     df1 = pd.read_csv(res_file_1)
#     df2 = pd.read_csv(res_file_2)
    

#     merged_df = pd.merge(
#         df1, df2, on=["time", "x", "y", "speed", "motion_angle"], suffixes=("_A","_B"), validate="one_to_one"
#     )

#     all_rows.append(merged_df)
############################################


############# for ATC dataset #############
res_file_1 = f"nll_results/distri_gmm_feature_time_v2/atc-all.csv"
# res_file_1 = f"nll_results/distri_gmm_feature_ff_time_v2/atc-all.csv"
res_file_2 = f"nll_results/distri_gmm_siren_v2/atc-all.csv"

# Load both files
df1 = pd.read_csv(res_file_1)
df2 = pd.read_csv(res_file_2)

merged_df = pd.merge(
    df1, df2, on=["time", "x", "y", "speed", "motion_angle"], suffixes=("_A","_B"), validate="one_to_one"
)

all_rows.append(merged_df)
############################################



############# for ETH/UCY dataset, online cliffmap #############
# # version = "eth"
# # version = "hotel"
# version = "students003"
# # version = "students001"
# # version = "zara01"

# if version == "eth":
#     num_batches = 3
# elif version == "hotel":
#     num_batches = 3
# elif version == "students003":
#     num_batches = 2
# elif version == "students001":
#     num_batches = 1
# elif version == "zara01":
#     num_batches = 2

# for batch in range(num_batches):
#     res_file_1 = f"nll_results/online_cliff_ethucy/{version}/b{batch}.csv"
    
#     res_file_2 = f"nll_results/NIR_ethucy/{version}.csv"
    
#     # Load both files
#     df1 = pd.read_csv(res_file_1)
#     df2 = pd.read_csv(res_file_2)
    

#     merged_df = pd.merge(
#         df1, df2, on=["time", "x", "y", "speed", "motion_angle"], suffixes=("_A","_B"), validate="one_to_one"
#     )
#     all_rows.append(merged_df)
############################################


############# for ETH/UCY dataset, normal cliffmap #############
# version = "eth"
# # version = "hotel"
# # version = "students003"
# # version = "students001"
# # version = "zara01"

# res_file_1 = f"nll_results/cliff_ethucy/{version}.csv"
# res_file_2 = f"nll_results/NIR_ethucy/{version}.csv"

# # Load both files
# df1 = pd.read_csv(res_file_1)
# df2 = pd.read_csv(res_file_2)

# merged_df = pd.merge(
#     df1, df2, on=["time", "x", "y", "speed", "motion_angle"], suffixes=("_A","_B"), validate="one_to_one"
# )

# all_rows.append(merged_df)
############################################



############# for ETH/UCY dataset, STeF-map #############
# version = "eth"
# # version = "hotel"
# # version = "students003"
# # version = "zara01"

# if version == "eth":
#     num_batches = 15
# elif version == "hotel":
#     num_batches = 12
# elif version == "students003":
#     num_batches = 6
# elif version == "zara01":
#     num_batches = 9

# for batch in range(num_batches):
#     res_file_1 = f"nll_results/stef_ethucy_v2/{version}/b{batch}.csv"
    
#     res_file_2 = f"nll_results/NIR_ethucy/{version}.csv"
#     # res_file_2 = f"nll_results/cliff_ethucy/{version}.csv"

#     # Load both files
#     df1 = pd.read_csv(res_file_1)
#     df2 = pd.read_csv(res_file_2)
    

#     merged_df = pd.merge(
#         df1, df2, on=["time", "x", "y", "speed", "motion_angle"], suffixes=("_A","_B"), validate="one_to_one"
#     )

#     all_rows.append(merged_df)
############################################


all_df = pd.concat(all_rows, ignore_index=True)
# print(f"Total rows before cleaning: {len(all_df)}")
all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["nll_A", "nll_B"])
# check how many rows are left
# print(f"Total valid rows for comparison: {len(all_df)}")


# Paired t-test
tt = ttest_rel(all_df["nll_A"], all_df["nll_B"], alternative="greater")
mean_diff, ci = paired_ci(all_df["nll_A"], all_df["nll_B"])
dz = cohens_dz(all_df["nll_A"], all_df["nll_B"])

print(f"NLL paired t-test: statistic = {tt.statistic}, p = {tt.pvalue:.3e}")

print(f"A (variant): {Path(res_file_1).parent.name}")
print(f"B (ours): {Path(res_file_2).parent.name}")
print(
    f"NLL reduction by ours (A - B) = {mean_diff:.3f}, "
    f"95% CI of reduction = "
    f"[{ci[0]:.3f}, {ci[1]:.3f}]"
)
# print(f"Cohen's dz = {dz:.3f}")


mean_A = all_df["nll_A"].mean()
mean_B = all_df["nll_B"].mean()
std_A = all_df["nll_A"].std(ddof=1)
std_B = all_df["nll_B"].std(ddof=1)

print(f"Mean NLL_A = {mean_A:.3f} ± {std_A:.3f}")
print(f"Mean NLL_B = {mean_B:.3f} ± {std_B:.3f}")




########## between cliff-map-hour and siren ##########
# Total rows before cleaning: 5101093
# Total valid rows for comparison: 5101093
# NLL paired t-test: statistic = 631.2401053695055, p = 0.0
# Mean diff (A - B) = 1.1885109833182383, 95% CI = [(1.1848207246425961, 1.1922012419938806)]
# Cohen's dz = 0.2794878693937574
# Mean NLL_A = 1.9636140284433297 ± 4.953284898734786
# Mean NLL_B = 0.7751030451250897 ± 2.052713915794647


########## between stef-map-hour and siren ##########
# Total rows before cleaning: 5101093
# Total valid rows for comparison: 5101093
# NLL paired t-test: statistic = 1217.8110918067423, p = 0.0
# Mean diff (A - B) = 4.801368375403425, 95% CI = [(4.793640977132464, 4.809095773674386)]
# Cohen's dz = 0.5391980396649786
# Mean NLL_A = 5.576471420528516 ± 9.31421058092323
# Mean NLL_B = 0.7751030451250897 ± 2.052713915794647


########## between online and siren ##########
# Total rows before cleaning: 5101093
# Total valid rows for comparison: 5101093
# NLL paired t-test: statistic = 472.52070054158435, p = 0.0
# Mean diff (A - B) = 0.7523467392701585, 95% CI = [(0.7492260868605483, 0.7554673916797687)]
# Cohen's dz = 0.20921326562656792
# Mean NLL_A = 1.5274497843952497 ± 4.155519013526215
# Mean NLL_B = 0.7751030451250897 ± 2.052713915794647


########## between ff time and siren ##########
# Total rows before cleaning: 5101093
# Total valid rows for comparison: 5101093
# NLL paired t-test: statistic = 131.93726039866866, p = 0.0
# Mean diff (A - B) = 0.06292916620773614, 95% CI = [(0.06199433606015744, 0.06386399635531484)]
# Cohen's dz = 0.058416541485253125
# Mean NLL_A = 0.8380322113328263 ± 2.104755802616243
# Mean NLL_B = 0.7751030451250897 ± 2.052713915794647


########## between time grid and siren ##########
# Total rows before cleaning: 5101093
# Total valid rows for comparison: 5101093
# NLL paired t-test: statistic = 174.63810776620195, p = 0.0
# Mean diff (A - B) = 0.0820625700929601, 95% CI = [(0.08114158141482083, 0.08298355877109936)]
# Cohen's dz = 0.07732276868872581
# Mean NLL_A = 0.8571656152180497 ± 2.113225588463205
# Mean NLL_B = 0.7751030451250897 ± 2.052713915794647
