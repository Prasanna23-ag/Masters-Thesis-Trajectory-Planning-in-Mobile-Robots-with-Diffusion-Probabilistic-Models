import os
import glob
import numpy as np
import json
import pdb
import diffuser.utils as utils
import matplotlib.pyplot as plt

# DATASETS = [
#     f"{env}-{buffer}-v2"
#     for env in ["maze2d"]
#     for buffer in ["umaze", "medium", "large"]
# ]

DATASETS = ["birrt-dataset-v0"]
LOGBASE = "logs"
TRIAL = "*"
EXP_NAME = "plans*/*"
verbose = False



def load_results(paths):
    scores = []
    returns = []
    for i, path in enumerate(sorted(paths)):
        score, r = load_result(path)
        if verbose:
            print(path, score)
        if score is None:
            continue
        scores.append(score)
        returns.append(r)

    num_ = len(scores)
    if len(scores) > 0:
        mean = np.mean(scores)
        returns = np.array(returns)
        sus_rate = np.mean(np.array(scores) > 0)  # success rate = fraction of positive scores
    else:
        mean = np.nan
        sus_rate = np.nan

    if len(scores) > 1:
        err = np.std(scores) / np.sqrt(len(scores))
    else:
        err = 0
    return mean, err, scores, sus_rate



def load_result(path):
    """
    path : path to experiment directory; expects `rollout.json` to be in directory
    """
    # fullpath = os.path.join(path, 'rollout.json')

    if not os.path.exists(path):
        return None

    results = json.load(open(path, "rb"))
    score = results["score"] * 100
    r = np.sum(results["return"])
    return score, r


#######################
######## setup ########
#######################

if __name__ == "__main__":
    configs = ["config.maze2d_hl"]

    for cfg in configs:

        class Parser(utils.Parser):
            # dataset: str = "maze2d-medium-v1"
            dataset: str = "birrt-dataset-v0"
            config: str = cfg

        args = Parser().parse_args("plan")
        epochs = ["latest"]

        for dataset in [args.dataset] if args.dataset else DATASETS:
            subdir = os.path.join("logs", *args.savepath.split("/")[1:-1])


            reldir = subdir.split("/")[-1]
            paths = glob.glob(os.path.join(subdir, "0", "*_rollout.json"))
            paths = sorted(paths)
            
            # print("Looking for rollouts in:", os.path.join(subdir, "0", "*_rollout.json"))
            # print("Found paths:", paths)
            
            mean, err, scores, sus_rate = load_results(paths)
            if np.isnan(mean):
                continue
            path, name = os.path.split(subdir)
            print(
                f"{dataset.ljust(30)} | {name.ljust(50)} | {path.ljust(50)} | {len(scores)} scores \n    {mean:.1f} +/- {err:.2f}"
                f"\nsus_rate: {sus_rate * 100:.2f}"
            )
            if verbose:
                print(scores)
                print(sus_rate)

            with open("birrt_results.txt", "a") as f:
                f.write(f"{dataset.ljust(30)} | {name.ljust(50)} | {path.ljust(50)} | {len(scores)} scores\n")
                f.write(f"    {mean:.1f} +/- {err:.2f}\n")
                f.write(f"sus_rate: {sus_rate * 100:.2f}\n")



            if len(scores) > 0:
                plt.figure(figsize=(8,6))
                plt.hist(scores, bins=20, color="skyblue", edgecolor="black")
                plt.xlabel("Score")
                plt.ylabel("Frequency")
                plt.title("Performance Distribution for birrt-dataset-v0")
                plt.grid(True)
                plt.savefig(f"birrt_results_{name}.png")
                plt.close()