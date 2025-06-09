# Adapting to Dissimilar Tasks for Continual Learning via Gradient Norm Regularisation

This repository provides a modular and extensible framework for evaluating continual learning algorithms using [Avalanche](https://avalanche.continualai.org/). It includes baseline implementations such as **GNR (Gradient Norm Regularisation)** in a plugin-style architecture.

## 📂 Project Structure

```
├── main_C_baseline.py         # Main script to run experiments
├── model.py                   # Network architecture (MTAlexNet)
├── plugins/                   # Custom plugins including GNR and SPG
│   ├── gnr.py                 # Gradient-Norm-Aware update
│   ├── spg.py                 # Gradient-Norm-Aware consolidation
│   ├── customise_plugin.py    # Early stopping, LR getter, etc.
│   └── ...                    # Other utility plugins
├── README.md
└── ...
```


## 🔧 Installation

First, install dependencies:

```bash
git clone https://github.com/wang-xulong/GNR.git
cd GNR
pip install -r requirements.txt
```

Make sure you also have:

- Python ≥ 3.8
- PyTorch ≥ 1.12
- [Avalanche](https://github.com/ContinualAI/avalanche) ≥ 0.3
- `wandb` (optional, for logging)

## 🧪 Quick Start

Run GNR on Split CIFAR-100 (10 experiences, 5 seeds):

```bash
python main_C_baseline.py
```

### Custom Hyperparameters

You can override defaults via command line arguments:

```bash
python main_C_baseline.py --cuda 0 --n_experiences 5 --epochs 100 --drop1 0.3 --drop2 0.5 --init_lr 0.05
```

### Key Arguments

| Argument          | Description                       | Default |
| ----------------- | --------------------------------- | ------- |
| `--cuda`          | GPU index (-1 for CPU)            | `0`     |
| `--n_experiences` | Number of experiences             | `10`    |
| `--epochs`        | Training epochs per experience    | `200`   |
| `--batch_size`    | Mini-batch size                   | `64`    |
| `--drop1/drop2`   | Dropout rates                     | `0.2`   |
| `--init_lr`       | Initial learning rate             | `0.1`   |
| `--min_lr`        | Learning rate floor for scheduler | `1e-3`  |
| `--patience`      | Patience for early stopping       | `10`    |
| `--eval_every`    | Evaluation interval in epochs     | `5`     |

## 📈 Evaluation

This project uses Avalanche's `EvaluationPlugin` to track:

- Accuracy (per epoch, experience, stream)
- Forgetting
- Backward transfer
- Loss


## 📜 License

This project is licensed under the MIT License.