# Implement EV charging Gymnasium environment

- Load and preprocess EV charging dataset
- Create normalized 8-dimensional observation space
- Define continuous charging action space [0, 1]
- Implement environment reset and state generation
- Add reproducible environment initialization using random seeds
- Validate environment initialization and observation bounds


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

## Results Visualizations

### Training convergence

`learning_curves.png` shows the TD3 training reward across the three seeds.

### Baseline vs TD3

`metrics_comparison.png` compares:

- cumulative reward
- peak contribution
- charging cost
- target fulfilment

---

## Known Limitations

The current environment is a simplified representation of EV charging.

The main limitations are:

- manually selected reward weights
- incomplete target fulfilment by the learned policy
- simplified station dynamics
- limited number of random seeds
- no full multi-EV charger interaction model

These limitations provide directions for future work.

---

## Team Contributions

Each member contributed to the implementation, experimentation, documentation and final presentation.

The Git history provides a record of individual contributions.

Each team member also made a direct contribution to this README.

---

## Reproducibility

Install the required packages:

```bash
pip install "stable-baselines3[extra]" gymnasium pandas numpy matplotlib
```

Then validate, train, evaluate and generate plots using the scripts provided in this repository.

---

## License

This repository was developed as an academic project for coursework purposes.