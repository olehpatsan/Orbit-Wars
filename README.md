# Orbit Wars 1v1

Reinforcement learning agent for the [Orbit Wars Kaggle competition](https://www.kaggle.com/competitions/orbit-wars). The agent reached **~top-100** on the final leaderboard.

## Architecture

**Encoder:** A 8-layer Transformer (8 heads, d_model=224, d_ff=896) over three token types - owned planets, all fleets, and neutral planets - pooled into a global embedding.

**Policy:** Per-planet autoregressive decoding in descending ship-count order. For each owned planet the model predicts:
- `act/skip` (binary): whether to send a fleet this tick
- `target` (pointer): which planet/comet to send to
- `frac` (regression, sigmoid): what fraction of ships to send

The AR state vector accumulates information about already-committed moves so later planet decisions are conditioned on earlier ones.

**Value head:** Scalar V(s) from the global embedding, trained with GAE.

## Training pipeline

1. **Behavioral Cloning (BC)** - pretrain on decoded top-player replays:
   ```bash
   python src/train_bc.py /path/to/shards \
       --d-model 224 --n-layers 8 --d-ff 896 \
       --epochs 3 --lr 3e-4 \
       --save-path bc_checkpoint.npz
   ```

2. **PPO self-play** - fine-tune with league-based self-play:
   ```bash
   python scripts/run_train.py
   ```
   Edit `scripts/run_train.py` to set the BC checkpoint path, model config, and PPO hyperparameters before running.

## Run a local match

```bash
python scripts/visualize_match.py \
    --bc bc_checkpoint_5000.npz \
    --ppo orbit_1v1_1500.npz
```

Returns a `kaggle_environments` env object. In a Jupyter notebook, call `env.render(mode="ipython")` to view the animation.

To benchmark BC vs PPO win rate over N games:
```bash
python scripts/test_bc_vs_ppo.py \
    --bc bc_checkpoint_5000.npz \
    --ppo orbit_1v1_1500.npz \
    --n-games 20
```

## Requirements

```
pip install -r requirements.txt
```

- `jax` / `jaxlib` - simulator and training (GPU recommended for PPO)
- `optax` - optimizer
- `numpy` - CPU inference in the Kaggle submission
- `kaggle-environments` - local match runner and competition environment

## Project structure

```
src/               Core library (simulator, features, network, training)
scripts/           Entry points for training and evaluation
kaggle/            Kaggle submission agent and parallel map generation worker
```
