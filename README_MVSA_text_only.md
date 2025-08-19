## MVSA (Single) — Text-only Training Guide

This guide explains, step by step, how to run text-only experiments on the MVSA-Single dataset using this repository.

### 1) Environment setup

- Create and activate the environment:
```bash
conda env create -f timrel-env.yml
conda activate timrel-env
```

- Ensure CUDA is available if you intend to train on GPU.

### 2) Data preparation

There are two things needed:
- Images: MVSA-Single images under `data/MVSA-Single/data/*.jpg`
- Split + labels file: `data/data_key_mvsa.csv` with columns `tweet_id,text,label,split`

You have two options:

- Option A: Use the helper script (recommended). Place `MVSA-single.zip` at the repo root (`MVSA_MULTIMODAL/MVSA-single.zip`) and run:
```bash
cd MVSA_MULTIMODAL
python dataunzip.py
```
This extracts into `data/MVSA-Single/data/`. The script also normalizes common zip folder layouts.

- Option B: Manual placement. Ensure the following exist:
- `MVSA_MULTIMODAL/data/MVSA-Single/data/<id>.jpg`
- `MVSA_MULTIMODAL/data/data_key_mvsa.csv`

Notes:
- The path constant used by the code is configured in `models/config.py` (`DATA_PATH = "../data/"`). Default layout already matches `MVSA_MULTIMODAL/data/`.
- For MVSA (task 3), image path format is set to `MVSA-Single/data/{}.jpg` in `models/config.py`.

### 3) Quick sanity check

- Verify the key files are present:
```bash
ls -l MVSA_MULTIMODAL/data/data_key_mvsa.csv
ls -l MVSA_MULTIMODAL/data/MVSA-Single/data | head
```

### 4) Run text-only training

Core script: `models/run_txt.py`

- Common command (BERNICE on MVSA task 3):
```bash
cd MVSA_MULTIMODAL/models
python run_txt.py \
  --model_name bernice \
  --task 3 \
  --epochs 10 \
  --lr 5e-6 \
  --weight_decay 0.001 \
  --dropout 0.1 \
  --warmup_steps 50 \
  --gradient_clip 0.5 \
  --scheduler warmup_linear \
  --seed 30 \
  --save_model \
  --save_preds
```

- Tips:
  - If loss plateaus early, try `--lr 1e-5`.
  - `--testing` runs on a small sample for a quick smoke test.

### 5) What gets saved

Outputs are written under `results/txt_only/`:
- `*_metrics_val.csv`: validation metrics per epoch (F1/Precision/Recall weighted & macro, loss)
- `*_metrics_test.csv`: test metrics per epoch
- `*_preds.csv`: per-example predictions (columns: `data_id,label,prediction`)
- `*_net.pth`, `*_net_best.pth`: model checkpoints

Example folder:
- `MVSA_MULTIMODAL/results/txt_only/bernice_task3_seed30_preds.csv`
- `MVSA_MULTIMODAL/results/txt_only/bernice_task3_seed30_metrics_val.csv`
- `MVSA_MULTIMODAL/results/txt_only/bernice_task3_seed30_metrics_test.csv`

### 6) Evaluate predictions (optional)

Two quick ways:

- Use the built-in epoch CSVs (`*_metrics_val.csv`, `*_metrics_test.csv`).
- Or compute metrics from the saved predictions file. This repo includes a helper script `calc_metrics.py`:
```bash
cd MVSA_MULTIMODAL
python calc_metrics.py
```
It reads `results/txt_only/bernice_task3_seed30_preds.csv` and prints:
- F1 (weighted, macro)
- Precision (weighted, macro)
- Recall (weighted, macro)
- Accuracy
- Per-class report and confusion matrix

### 7) Reproducibility and configuration

- Seed: set via `--seed` (default 30).
- Batch size: configured by task in `models/config.py` (for MVSA task 3 default is 8).
- Max sequence length: configured in `models/config.py` (default 128).
- Paths: `models/config.py` sets `DATA_PATH` and image formats per task.

### 8) Troubleshooting

- FileNotFoundError for images:
  - Re-check extraction under `data/MVSA-Single/data/` and that filenames match tweet IDs in `data_key_mvsa.csv`.
- Loss not decreasing:
  - Try `--lr 1e-5` and keep `--scheduler warmup_linear`.
  - Ensure `data_key_mvsa.csv` has correct labels and `split` values: `train/val/test`.
- GPU out of memory:
  - Lower batch size for task 3 in `models/config.py` (e.g., 8 → 4).

### 9) Example minimal run (smoke test)
```bash
cd MVSA_MULTIMODAL/models
python run_txt.py --model_name bernice --task 3 --epochs 1 --testing --save_preds
```
This quickly verifies the pipeline, producing small outputs in `results/txt_only/testing/`.
