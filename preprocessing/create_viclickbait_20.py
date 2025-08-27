import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(df: pd.DataFrame, seed: int = 42):
    # Binary labels assumed: 'clickbait' / 'non-clickbait'
    df = df.copy()
    df['label_int'] = df['label'].map({'non-clickbait': 0, 'clickbait': 1})
    assert df['label_int'].isna().sum() == 0, "Label values must be 'clickbait' or 'non-clickbait'"

    # 80/10/10 stratified
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df['label_int']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df['label_int']
    )

    train_df = train_df.assign(split='train')
    val_df = val_df.assign(split='val')
    test_df = test_df.assign(split='test')
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def stratified_subsample(df_split: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    parts = []
    for label_val, grp in df_split.groupby('label_int'):
        n = max(1, int(len(grp) * frac))
        parts.append(grp.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join('data', 'ViClickbait-2025', 'clickbait_dataset_vietnamese.csv'), help='path to ViClickbait CSV')
    parser.add_argument('--dataset_root', default=os.path.join('data', 'ViClickbait-2025'), help='dataset root to resolve thumbnail paths')
    parser.add_argument('--output', default=os.path.join('data', 'data_key_viclickbait.csv'), help='output key CSV path')
    parser.add_argument('--frac', type=float, default=1.0, help='fraction per split to keep (use 1.0 for 100%)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {'id', 'title', 'thumbnail_url', 'label'}
    missing = required - set(df.columns)
    assert not missing, f'Missing required columns: {missing}'

    # Add split 80/10/10 stratified by label
    df_split = make_splits(df, seed=args.seed)

    # 20% per split (stratified)
    out_parts = []
    for split_name, grp in df_split.groupby('split'):
        sub = stratified_subsample(grp, args.frac, seed=args.seed)
        out_parts.append(sub)
        print(f"Split {split_name}: {len(grp)} -> {len(sub)}")
        print(sub['label_int'].value_counts().sort_index())
    out = pd.concat(out_parts).reset_index(drop=True)

    # Resolve image path: if thumbnail_url is relative (e.g., 'data/images/...'), make it rooted at dataset_root
    def resolve_image(p: str) -> str:
        p = str(p or '')
        if p.startswith('http://') or p.startswith('https://'):
            return p
        # Normalize separators for robust matching on Windows/Linux
        p_norm = p.replace('\\', '/').lstrip('/')
        # 1) If directly exists under dataset_root
        direct = os.path.normpath(os.path.join(args.dataset_root, p_norm))
        if os.path.exists(direct):
            return direct
        # 2) If path contains 'images/...', map to dataset_root/images/...
        if 'images/' in p_norm:
            after = p_norm.split('images/', 1)[1]
            candidate = os.path.normpath(os.path.join(args.dataset_root, 'images', after))
            if os.path.exists(candidate):
                return candidate
        # 3) Fallback: try by filename under dataset_root/images
        fname = os.path.basename(p_norm)
        candidate2 = os.path.normpath(os.path.join(args.dataset_root, 'images', fname))
        if os.path.exists(candidate2):
            return candidate2
        # 4) Last resort: join to dataset_root
        return direct

    out['image'] = out['thumbnail_url'].apply(resolve_image)

    # Build key similar to project: tweet_id, text, label, split (+ keep image column for convenience)
    key = out.rename(columns={'id': 'tweet_id', 'title': 'text', 'label_int': 'label'})[
        ['tweet_id', 'text', 'image', 'label', 'split']
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    key.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()


