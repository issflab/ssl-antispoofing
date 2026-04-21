
- https://github.com/TakHemlata/SSL_Anti-spoofing/tree/main

# SSL Anti‑Spoofing

This repository implements a self‑supervised learning (SSL) based anti‑spoofing pipeline for audio. It has two main stages:

* **Frontend (SSL feature extractors):** It can extract many ssl features from [s3prl library](https://github.com/s3prl/s3prl). Examples include wavlm\_large, wav2vec 2.0 xlsr (xls_r_300m), npc\_960hr, etc.
* **Backend (Classifier models):** Downstream models include a Linear Head model, [AASIST](https://arxiv.org/pdf/2110.01200), and [XLSR-SLS](https://openreview.net/pdf?id=acJMIXJg2u).

---

<!-- ## Features

* **SSL feature extractors:**

  * `wavlm_large`
  * `mae_ast_frame`
  * `npc_960hr`
* **Classifier models:**

  * `aasist`
  * `sls`
* Switch SSL extractor or model via config file or command‑line
* Simple commands for training and evaluation -->
---

## Installation


## ⚙️ Configuration

This project now uses a structured `.env` experiment config with five logical sections:

1. `experiment`
2. `data`
3. `model`
4. `training`
5. `evaluation`

The environment variables mirror this structure with `SECTION__FIELD` naming.  
For example:

```env
EXPERIMENT__NAME=aasist_codec1
DATA__DATABASE_PATH=/data/Data
MODEL__BACKEND=aasist
TRAINING__EPOCHS=50
EVALUATION__CHECKPOINT=
```

### Main Experiment Config

The main `.env` file contains run-specific settings such as:

- experiment name and output directory
- dataset and protocol paths
- frontend, backend, and loss selection
- optimizer, learning rate, scheduler, and metric
- evaluation checkpoint and score-file options

See:

- `example_configs/aasist_codecfake.env`
- `example_configs/aasist_mlaad.env`

### Per-Model Config

To keep the main experiment config from getting too cluttered, backend-specific defaults live in:

- `configs/models/aasist.json`
- `configs/models/sls.json`

These files are referenced from the main `.env` via:

```env
MODEL__BACKEND_CONFIG=configs/models/aasist.json
```

The model config can define backend-specific architecture defaults and training defaults for that backend.  
The main `.env` remains the primary experiment-level configuration, and values in the `.env` override model defaults.

<!-- **Or via CLI flags (model architecture is set in `config.py`):**

* **wavlm\_large**

  ```bash
  python main2.py --batch_size 14 --num_epochs 50 --lr 1e-6 --weight_decay 1e-4 --ssl_feature wavlm_large --seed 1234 --emb_size 256 --num_encoders 12
  ```
* **mae\_ast\_frame**

  ```bash
  python main2.py --batch_size 14 --num_epochs 50 --lr 1e-6 --weight_decay 1e-4 --ssl_feature mae_ast_frame --seed 1234 --emb_size 256 --num_encoders 12
  ```
* **npc\_960hr**

  ```bash
  python main2.py --batch_size 14 --num_epochs 50 --lr 1e-6 --weight_decay 1e-4 --ssl_feature npc_960hr --seed 1234 --emb_size 256 --num_encoders 12
  ``` -->

## Training

To train with an experiment config:

```bash
python main2.py --env_file example_configs/aasist_codecfake.env
```

## Evaluation

```bash
python main2.py --env_file example_configs/aasist_codecfake.env
```

Set `TRAINING__MODE=eval` and `EVALUATION__CHECKPOINT=/path/to/model.pth` in the `.env` file for evaluation runs.

---

## Switching Components

Change the SSL frontend in the `.env`:

```env
MODEL__FRONTEND=wavlm_large
```

Change the backend and its model config:

```env
MODEL__BACKEND=aasist
MODEL__BACKEND_CONFIG=configs/models/aasist.json
```

Change the loss:

```env
MODEL__LOSS=weighted_cce
MODEL__FOCAL_GAMMA=2.0
```

---

## Logs & Outputs

Outputs are saved under:

```text
<EXPERIMENT__OUTPUT_DIR>/<EXPERIMENT__NAME>/
```

This run directory contains:

- `checkpoints/`
- `logs/`
- `metrics/`

## Extension Layout

The repository now has a small modular layer under `anti_spoofing/`:

* `anti_spoofing/models.py`: backend model registry
* `anti_spoofing/losses.py`: pluggable loss registry
* `anti_spoofing/data.py`: dataset/protocol builders
* `anti_spoofing/optim.py`: optimizer selection
* `anti_spoofing/engine.py`: reusable train/eval loops
* `anti_spoofing/frontends/`: SSL frontend implementations
* `anti_spoofing/backends/`: backend classifier implementations

To add a new backend model, implement the model and register it in `anti_spoofing/models.py`.
To add a new loss, implement it and register it in `anti_spoofing/losses.py`.
