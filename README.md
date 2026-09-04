 # EV Charging Optimization using TD3
 ## Project Overview

This project applies Twin Delayed Deep Deterministic Policy Gradient (TD3) to the problem of electric vehicle (EV) charging optimization.

The goal is to learn a continuous charging policy that balances:

charging cost
grid/peak contribution
charging requirement satisfaction
physical charging constraints

The project uses a custom Gymnasium environment and the Stable-Baselines3 TD3 implementation.

## Dataset

The project uses an EV charging dataset containing charging-session information such as:

initial and final SOC
battery capacity
charging power
charging duration
electricity price
station load
charging demand
queue length
waiting time
renewable energy ratio
charging priority

The dataset is provided as:

ev_charging_dataset.csv
Project Structure
.
├── ev_charging_dataset.csv
├── ev_charging_env.py
├── validate_env.py
├── train_td3.py
├── evaluate.py
├── plot_results.py
├── evaluation_results.csv
├── learning_curves.png
├── metrics_comparison.png
├── models/
│   ├── td3_seed0.zip
│   ├── td3_seed1.zip
│   └── td3_seed2.zip
└── logs/
    ├── td3_seed0_episodes.csv
    ├── td3_seed1_episodes.csv
    └── td3_seed2_episodes.csv
## Environment

ev_charging_env.py implements the custom Gymnasium environment.

implements the custom Gymnasium environment.

## Observation

The environment uses six normalized state variables:

1. Current SOC
2. Target SOC
3. Remaining charging-time fraction
4. Electricity price
5. Background charging demand
6. Vehicle power-limit fraction

## Action

The agent produces a continuous normalized charging action:

0.0 → no charging
1.0 → maximum available charging power

The physical charging power is:

actual_power = action × vehicle_power_limit

## Reward

The reward combines:

SOC progress
electricity cost
grid/peak contribution
target fulfilment
shortfall penalties

The purpose is to encourage useful charging while discouraging unnecessarily aggressive charging.

## Validation

Before training, the environment is checked for:

Gymnasium API compliance
Reproducibility under a fixed seed
Physical feasibility of reaching the target at 100% charging power

Run:

python validate_env.py --data ev_charging_dataset.csv

Expected result:

All validation checks passed successfully.

## Training

TD3 is trained using Stable-Baselines3.

Example:

```bash
python train_td3.py \
    --data ev_charging_dataset.csv \
    --seed 0 \
    --timesteps 100000
```

Three independent seeds were used for the final experiment.

The trained models are saved under:

```text
models/
```

and episode-level training logs are saved under:

```text
logs/
```

---

## Evaluation

The trained policies are compared with an **Immediate Charging** baseline.

The baseline always selects:

```text
action = 1.0
```

The TD3 models are evaluated deterministically on the held-out test split.

Example:

```bash
python evaluate.py \
    --data ev_charging_dataset.csv \
    --model models/td3_seed0.zip models/td3_seed1.zip models/td3_seed2.zip \
    --seeds 0 1 2 \
    --episodes 100
```

---

## Final Results

### Immediate Charging

| Metric | Result |
|---|---:|
| Cost | 0.895 ± 0.031 |
| Peak contribution | 14.163 ± 0.967 |
| Target met | 100.0 ± 0.000% |
| Violations | 0.000 ± 0.000 |
| Reward | 7.464 ± 0.712 |
| Mean action | 1.000 ± 0.000 |

### TD3

| Metric | Result |
|---|---:|
| Cost | **0.730 ± 0.027** |
| Peak contribution | **7.264 ± 0.606** |
| Target met | 62.0 ± 4.899% |
| Violations | **0.000 ± 0.000** |
| Reward | 6.412 ± 0.779 |
| Mean action | **0.821 ± 0.017** |

### Main finding

TD3 reduced:

- charging cost by approximately **18.4%**
- peak contribution by approximately **48.7%**

However, target fulfilment decreased from **100% to 62%**.

Therefore, the learned policy demonstrates a meaningful cost/grid-efficiency trade-off but requires further reward tuning to improve charging-service reliability.

---
