import os
import shutil
import argparse
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
SRC_IMG_DIR = os.path.join(DATA_DIR, 'MVSA-Single', 'data')
DST_IMG_DIR = os.path.join(DATA_DIR, 'MVSA-Single-20', 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', default=os.path.join(DATA_DIR, 'data_key_mvsa_20.csv'), help='Path to 20% key CSV')
    parser.add_argument('--src', default=SRC_IMG_DIR, help='Source images dir (MVSA-Single/data)')
    parser.add_argument('--dst', default=DST_IMG_DIR, help='Destination images dir (MVSA-Single-20/data)')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    df = pd.read_csv(args.key)
    ids = df['tweet_id'].astype(str).unique().tolist()

    copied, missing = 0, []
    for tid in ids:
        src_jpg = os.path.join(args.src, f'{tid}.jpg')
        src_png = os.path.join(args.src, f'{tid}.png')
        if os.path.exists(src_jpg):
            dst = os.path.join(args.dst, f'{tid}.jpg')
            shutil.copy2(src_jpg, dst)
            copied += 1
        elif os.path.exists(src_png):
            dst = os.path.join(args.dst, f'{tid}.png')
            shutil.copy2(src_png, dst)
            copied += 1
        else:
            missing.append(tid)

    print(f'Copied: {copied} images to {args.dst}')
    print(f'Missing: {len(missing)}')
    if missing[:10]:
        print('Examples:', missing[:10])

if __name__ == '__main__':
    main()
