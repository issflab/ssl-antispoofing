
- https://github.com/TakHemlata/SSL_Anti-spoofing/tree/main
- https://github.com/TakHemlata/SSL_Anti-spoofing/tree/main

## Quick start

### Train
Make sure `config.py` is set with `cfg.mode='train'`, correct `database_path` / `protocols_path`, and `cfg.model_name`. Then run:

```bash
python main2.py --batch_size 14 --num_epochs 50 --lr 1e-6 --weight_decay 1e-4 --algo 5 --ssl_feature wavlm_large --seed 1234 --emb_size 256 --num_encoders 12
```

### Eval
Set `cfg.mode='eval'` in `config.py` (or use `--eval` if your script supports it) and point to a trained checkpoint:

```bash
python main2.py --eval --model_path ./output/models/run1/best.pth --batch_size 14 --ssl_feature wavlm_large --seed 1234 --emb_size 256 --num_encoders 12
```
