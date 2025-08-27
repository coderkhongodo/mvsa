ViClickbait-2025 integration

This repository now supports a new binary task (task=7) for Vietnamese clickbait detection using the ViClickbait-2025 dataset.

Dataset layout
After running `python dataunzip.py` with `dataset.zip` present at the project root, the dataset is extracted to:

- `data/ViClickbait-2025/`
  - `clickbait_dataset_vietnamese.csv`
  - `clickbait_dataset_vietnamese.jsonl`
  - `images/` (PNG/JPG thumbnails)

The CSV header includes at least:
- `id,url,title,lead_paragraph,category,publish_datetime,source,thumbnail_url,label`
- `label` is `clickbait` or `non-clickbait`.

Create a 20% stratified key
Use the script to create a 20% per-split stratified key file which preserves class balance across splits and resolves image paths:

```bash
python preprocessing/create_viclickbait_20.py --dataset_root "C:\\Users\\huynh\\OneDrive\\Máy tính\\task1\\mvsa\\data\\ViClickbait-2025"
```

This produces:
- `data/data_key_viclickbait_20.csv` with columns: `tweet_id,text,image,label,split`
  - `tweet_id` ← `id`
  - `text` ← `title`
  - `image` ← absolute/normalized path to the thumbnail (local file or http URL)
  - `label` ← integer (non-clickbait=0, clickbait=1)

Configuration (task=7)
We added task 7 (`viclick`) in `models/config.py`:
- `PATH[7] = data/data_key_viclickbait_20.csv`
- `IMG_FMT[7] = None` (dataset provides absolute image paths)
- `CLASSES[7] = ['non-clickbait','clickbait']`

Paths are now resolved dynamically from the repository root, removing hardcoded `/root/...`.

Models
- Text backbone: in addition to existing options, `phobert` is supported (mapped to `vinai/phobert-large`).
- Multimodal late fusion supports reading absolute image paths from the key for task 7.
- Auxiliary tasks available (recommended for ViClickbait):
  - ITC (image–text contrastive)
  - TIM (image–text matching)
  - IADDS is not applicable (no label provided)

Example runs
Text-only (PhoBERT):
```bash
python models/run_txt.py --model_name phobert --task 7 --epochs 5 --save_preds --save_model
```

Image-only (ViT):
```bash
python models/run_img.py --model_name vit --task 7 --epochs 5 --save_preds --save_model
```

Multimodal late fusion (ViT + PhoBERT) with auxiliary losses:
```bash
python models/run_mm_late.py \
  --txt_model_name phobert \
  --img_model_name vit \
  --fusion_name concat \
  --use_clip_loss --beta_itc 0.1 \
  --use_tim_loss  --beta_itm 0.1 \
  --task 7 --epochs 5 --save_model --save_preds
```

Notes
- Ensure `transformers`, `torch`, and `torchvision` are installed; models will be downloaded from HuggingFace.
- The key file contains absolute paths on Windows; training scripts will open images from those paths directly.
- You can adjust `--frac` and `--seed` in `create_viclickbait_20.py` to change subset size and reproducibility.


