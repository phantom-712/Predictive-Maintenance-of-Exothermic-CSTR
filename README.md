# Predictive Maintenance of Exothermic CSTR

This repository contains the foundational work for generating and visualizing synthetic data for the predictive maintenance of an exothermic Continuous Stirred-Tank Reactor (CSTR).

## Contents

- scripts/task_1.py: A Python script that utilizes scipy.integrate.solve_ivp and multiprocessing to simulate the kinetics and thermodynamics of the reactor. It generates a dataset by solving ordinary differential equations across thousands of parameter variations in parallel.
- dataset/: Contains the generated cstr_21k_dataset_parallel.csv which includes 21,000 samples of reactor states. Features include flow rate (q), inlet concentration (cAi), inlet temperature (Ti), and coolant temperature (Tc), with targets of final concentration (final_cA) and final temperature (final_T).
- plots/: Contains exploratory data analysis (EDA) visualizations generated from the dataset, including:
  - Input parameter distributions
  - Correlation heatmaps of reactor variables
  - Safety and thermal runaway analysis thresholds

## Setup and Usage

To run the simulation and regenerate the dataset and plots, install the required dependencies:

```bash
pip install numpy pandas matplotlib scipy seaborn
```

Execute the script:

```bash
python scripts/task_1.py
```
