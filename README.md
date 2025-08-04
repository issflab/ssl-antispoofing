
- https://github.com/TakHemlata/SSL_Anti-spoofing/tree/main
- https://github.com/TakHemlata/SSL_Anti-spoofing/tree/main

# SSL Anti‑Spoofing

This repository implements a self‑supervised learning (SSL) based anti‑spoofing pipeline for audio. It has two main stages:

* **Frontend (SSL feature extractors):** wavlm\_large, mae\_ast\_frame, npc\_960hr
* **Backend (Classifier models):** AASIST, SLS, XLSR‑Mamba

---

## Features

* **SSL feature extractors:**

  * `wavlm_large`
  * `mae_ast_frame`
  * `npc_960hr`
* **Classifier models:**

  * `aasist`
  * `sls`
  * `xlsrmamba`
* Switch SSL extractor or model via config file or command‑line
* Simple commands for training and evaluation
---

## Quick Start

### 1. Configure

**YAML file (`config.yaml`):**

```yaml
ssl_feature: wavlm_large    # choose: wavlm_large | mae_ast_frame | npc_960hr
model_arch: aasist          # choose: aasist | sls | xlsrmamba
mode: train                 # train or eval
save_dir: output/models     # where to save models
# other settings: batch size, learning rate, etc.
```

**Or via CLI flags (model architecture is set in `config.yaml`):**

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
  ```

### 2. Training

```bash
python main2.py --config config.yaml
```

### 3. Evaluation

```bash
python main2.py \
  --config config.yaml \
  --mode eval \
  --ckpt output/models/your_model.pth
```

---

## Switching Components

* **Change SSL feature extractor:**

  * In `config.yaml`: set `ssl_feature` to `wavlm_large`, `mae_ast_frame`, or `npc_960hr`.
  * Or add `--ssl_feature <name>` on the CLI.
* **Change classifier model:**

  * In `config.yaml`: set `model_arch` to `aasist`, `sls`, or `xlsrmamba`.
  * Or add `--model_arch <name>` on the CLI.

---

## Logs & Outputs

* **Model checkpoints:** saved under the directory specified by `save_dir`.
