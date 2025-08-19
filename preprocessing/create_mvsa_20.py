import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# This script creates a 20% stratified subset of the original MVSA key file,
# preserving class distribution within each split (train/val/test).

def stratified_subsample(df_split: pd.DataFrame, frac: float) -> pd.DataFrame:
    # group by label to preserve class distribution
    parts = []
    for label, df_lab in df_split.groupby('label'):
        # round to at least 1 if class exists
        n = max(1, int(len(df_lab) * frac))
        parts.append(df_lab.sample(n=n, random_state=42))
    return pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../data/data_key_mvsa.csv', help='path to original data_key_mvsa.csv')
    parser.add_argument('--output', default='../data/data_key_mvsa_20.csv', help='path to save the 20% subset')
    parser.add_argument('--frac', type=float, default=0.20, help='fraction per split to keep')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert set(['tweet_id','text','label','split']).issubset(df.columns), 'Missing required columns'

    out_parts = []
    for split_name, df_split in df.groupby('split'):
        sub = stratified_subsample(df_split, args.frac)
        out_parts.append(sub)
        print(f"Split {split_name}: {len(df_split)} -> {len(sub)}")
        print(sub['label'].value_counts().sort_index())
    out = pd.concat(out_parts).reset_index(drop=True)

    # Optional: ensure tweet_id uniqueness
    out = out.drop_duplicates(subset=['tweet_id']).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

if __name__ == '__main__':
    main()
