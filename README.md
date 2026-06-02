# Masters-Thesis-Trajectory-Planning-in-Mobile-Robots-with-Diffusion-Probabilistic-Models

Abstract: This repository contains the implementation, training pipeline, and evaluation framework for my Master’s Thesis: "Trajectory Planning in Mobile Robots with Diffusion Probabilistic Models." The work explores whether diffusion-based generative models can learn from Bi-RRT expert demonstrations to produce smoother, more consistent, and more optimal trajectories for mobile robot navigation.

To achieve this, the project implements a Simple Hierarchical Diffusion (SHD) architecture consisting of:

- A High‑Level Diffusion Model that predicts coarse sub‑goals
- A Low‑Level Diffusion Model that generates short‑horizon action sequences
- A Hybrid SHD + Bi‑RRT execution pipeline
- A Gym-based evaluation environment with LL diffuser actions and PID-controlled tracking

## Installation
This repository contains the full pipeline for hierarchical diffusion training, and evaluation in a custom Gym environment.

### Create environment
```bash
conda env create -f environment.yml
conda activate diffusion_planner
pip install -e .
```
## Dataset Setup (IMPORTANT)

The Bi-RRT expert dataset used for training the diffusion models must be placed inside the `.d4rl` directory in your home folder.
### Steps:
- Download the dataset file `birrt-dataset-v0.hdf5`
- Move the dataset file into it `.d4RL` directory
D4RL-style datasets are automatically loaded from this location during training and evaluation.


## Model Training
The High-Level and Low-Level diffusion models can be trained independently or in parallel.

- Train the High-Level Diffusion Model
```bash
python scripts/train.py --config config.maze2d_hl --dataset birrt-dataset-v0
```
- Train the Low-Level Diffusion Model
```bash
python scripts/train.py --config config.maze2d_ll --dataset birrt-dataset-v0
```

## Model Evaluation
To evaluate the hierarchical diffusion planner in the Gym environment:

```bash
python scripts/hd_plan_maze2d.py --dataset birrt-dataset-v0
```
You may adjust:

- Sub-goal sampling temperature
- Denoising steps
- PID gains
- Action horizon

## Acknowledgements
This work is inspired by and builds upon:

### Simple Hierarchical Planning with Diffusion
Chang Chen, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre, Sungjin Ahn
Paper: https://arxiv.org/pdf/2401.02644  
GitHub: https://github.com/changchencc/Simple-Hierarchical-Planning-with-Diffusion

Their hierarchical diffusion framework served as a conceptual reference for the architecture implemented in this thesis.

Additional thanks to:
- Technical University of Dortmund 
- Ruhr Universität Bochum – Mechanics of Adaptive Systems
- Supervisors: Prof. Dr.-Ing. Tamara Nestorović, M.Sc. Amit Pal
- Special thanks: Jr. Prof. Dr. David Kappel, M.Sc. Neeraj Mohan Sushma

## Citation

If you use this code in academic work, please cite the accompanying thesis:
```bash
@mastersthesis{agarwal2026diffusionplanning,
  title     = {Trajectory Planning in Mobile Robots with Diffusion Probabilistic Models},
  author    = {Agarwal, Prasanna},
  school    = {Technical University of Dortmund},
  year      = {2026},
  address   = {Dortmund, Germany},
}
```
