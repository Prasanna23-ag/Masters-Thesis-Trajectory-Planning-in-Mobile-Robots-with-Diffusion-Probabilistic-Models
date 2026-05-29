# Masters-Thesis-Trajectory-Planning-in-Mobile-Robots-with-Diffusion-Probabilistic-Models

Abstract: This repository contains the implementation, training pipeline, and evaluation framework for my Master’s Thesis: Trajectory Planning in Mobile Robots with Diffusion Probabilistic Models. The work explores whether diffusion-based generative models can learn from Bi-RRT expert demonstrations to produce smoother, more consistent, and more optimal trajectories for mobile robot navigation.
To achieve this, the project implements a Simple Hierarchical Diffusion (SHD) architecture consisting of:

- A High‑Level Diffusion Model that predicts coarse sub‑goals
- A Low‑Level Diffusion Model that generates short‑horizon action sequences
- A Hybrid SHD + Bi‑RRT execution pipeline
- A Gym-based evaluation environment with LL diffuser actions and PID-controlled tracking

## Installation
This repository contains the full pipeline for hierarchical diffusion training, and evaluation in a custom Gym environment.

#Create environment
```bash
conda env create -f environment.yml
conda activate diffusion_planner
pip install -e .
```
