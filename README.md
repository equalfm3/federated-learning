# Privacy-Preserving Federated Learning

Federated learning framework with federated averaging (FedAvg), differential privacy (DP-SGD), secure aggregation simulation, and non-IID data handling. Trains across simulated clients without sharing raw data.

## What This Covers

- FedAvg from scratch: local training → weight aggregation → global model
- Differential privacy: DP-SGD with per-sample gradient clipping and noise
- Secure aggregation: simulated secret sharing protocol
- Non-IID data partitioning: Dirichlet allocation across clients
- Privacy budget tracking (epsilon accounting)
- Convergence analysis: IID vs non-IID, with/without DP

## Structure

```
├── src/
│   ├── server.py              # Federated server (aggregation)
│   ├── client.py              # Federated client (local training)
│   ├── aggregation/
│   │   ├── fedavg.py          # Federated averaging
│   │   ├── fedprox.py         # FedProx (heterogeneity)
│   │   └── secure_agg.py     # Secure aggregation simulation
│   ├── privacy/
│   │   ├── dp_sgd.py          # DP-SGD optimizer
│   │   ├── noise.py           # Gaussian/Laplace noise
│   │   └── accountant.py      # Privacy budget accounting
│   ├── data/
│   │   └── partitioner.py     # IID / non-IID data splitting
│   └── models/
│       └── simple_models.py   # CNN, MLP for experiments
├── configs/
│   ├── fedavg_mnist.yaml
│   └── dp_fedavg_cifar.yaml
├── notebooks/
│   └── walkthrough.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.server --config configs/fedavg_mnist.yaml --num_clients 10 --rounds 50
```
