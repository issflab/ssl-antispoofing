
- https://github.com/TakHemlata/SSL_Anti-spoofing/tree/main

# SSL Anti‑Spoofing

This repository implements a self‑supervised learning (SSL) based anti‑spoofing pipeline for audio. It has two main stages:

* **Frontend (SSL feature extractors):** It can extract many ssl features from [s3prl library](https://github.com/s3prl/s3prl). Examples include wavlm\_large, wav2vec 2.0 xlsr (xls_r_300m), npc\_960hr, etc.
* **Backend (Classifier models):** Four downstream models are implemented, including a Linear Head model, [AASIST](https://arxiv.org/pdf/2110.01200), [XLSR-SLS](https://openreview.net/pdf?id=acJMIXJg2u), [XLSR‑Mamba](https://github.com/swagshaw/XLSR-Mamba).

---

<!-- ## Features

* **SSL feature extractors:**

  * `wavlm_large`
  * `mae_ast_frame`
  * `npc_960hr`
* **Classifier models:**

  * `aasist`
  * `sls`
  * `xlsrmamba`
* Switch SSL extractor or model via config file or command‑line
* Simple commands for training and evaluation -->
---

## Quick Start

### 1. Configure

You can configure downstream model, training and evaluation datasets via config file and SSL frontend model via command‑line.

## 🧰 Example Configurations

## ⚙️ Configuration

This project uses a simple `.env`-based configuration system.  
All key training, dataset, and protocol settings are read from environment variables, which allows you to easily switch between different experiments or datasets **without modifying code**.

## 1) Create Your `.env` File

Either use of the provided example config files (aasist_asv19.env, aasist_mlaad, or sls_asv19), or copy the template and edit it:

```bash
cp aasist_asv19.env sample.env
```

Open sample.env and fill in the fields for your dataset and experiment.

## 2) Required Variables

| Variable              | Description                                    | Example                                |
|-----------------------|------------------------------------------------|----------------------------------------|
| `SSL_DATABASE_PATH`   | Absolute path to the dataset root              | `/data/ASV19`                          |
| `SSL_PROTOCOLS_PATH`  | Path to folder containing protocol files       | `/data/ASV19/protocols`                |
| `SSL_TRAIN_PROTOCOL`  | Training protocol filename                     | `ASVspoof2019_train_protocol.txt`      |
| `SSL_DEV_PROTOCOL`    | Development protocol filename                  | `ASVspoof2019_dev_protocol.txt`        |

You must set these before running training or evaluation.

## 3) Protocol Format Variables

Some datasets use different delimiters or column indices. Set these to match your protocol files:

| Variable                    | Description                                              | Typical (ASV19) |
|-----------------------------|----------------------------------------------------------|-----------------|
| `SSL_PROTOCOL_DELIMITER`    | Delimiter in protocol files (" ", ",", or "\t")          | `" "`           |
| `SSL_PROTOCOL_KEY_COL`      | Column index containing utterance IDs                    | `0`             |
| `SSL_PROTOCOL_LABEL_COL`    | Column index containing labels (e.g., bona fide/spoof)   | `4`             |

If your format matches ASVspoof2019, the defaults in `aasist_asv19.env` should work.  
For other datasets (e.g., MLAAD), update these accordingly.

## 4) Model & Run Settings

| Variable            | Description                                    | Example          |
|---------------------|------------------------------------------------|------------------|
| `SSL_MODEL_ARCH`    | Model architecture (aasist, sls, xlsrmamba)    | `aasist`         |
| `SSL_MODE`          | Run mode (train or eval)                       | `train`          |
| `SSL_MODEL_NAME`    | Run name (used for saving checkpoints/logs)    | `aasist_ASV19`   |
| `CUDA_DEVICE`       | GPU device string                              | `cuda:0`         |

## 5) Using the `.env` File

Load your `.env` and run:

```bash
source sample.env
```

You can keep multiple `.env` files for different experiments and switch easily:

```bash
# AASIST on ASV19
source configs/aasist_asv19.env

# AASIST on MLAAD
source configs/aasist_mlaad.env

# SLS on ASV19
source configs/sls_asv19.env
```

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

### 2. Training

To train your model, run the following command,

```bash
python main2.py --batch_size 14 --num_epochs 50 --ssl_model wavlm_large
```

### 3. Evaluation

```bash
python main2.py  --ckpt output/models/your_model.pth
```

---

## Switching Components

* **Change SSL feature extractor:**

  * In `config.py`: set `ssl_feature` to `wavlm_large`, `mae_ast_frame`, or `npc_960hr`.
  * Or add `--ssl_feature <name>` on the CLI.
* **Change classifier model:**

  * In `config.py`: set `model_arch` to `aasist`, `sls`, or `xlsrmamba`.
  * Or add `--model_arch <name>` on the CLI.

---

## Logs & Outputs

* **Model checkpoints:** saved under the directory specified by `save_dir`.
