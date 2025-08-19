import os
import zipfile
import shutil

# Optional: chỉ dùng gdown nếu thiếu file zip
try:
    import gdown
except ImportError:
    gdown = None

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(ROOT, "MVSA-single.zip")
DATA_DIR = os.path.join(ROOT, "data")
DEST_DIR = os.path.join(DATA_DIR, "MVSA-Single")
TMP_DIR = os.path.join(DATA_DIR, "_mvsa_tmp")

FILE_ID = "1zdoXpFXLzqsP_E-WO6JgNJxviInYMdDe"  # nếu cần tải

os.makedirs(DATA_DIR, exist_ok=True)

# 1) Có sẵn zip thì dùng luôn, không thì tải về
if not os.path.exists(ZIP_PATH):
    if gdown is None:
        raise RuntimeError("MVSA-single.zip không tồn tại và gdown chưa được cài. Hãy cài: pip install gdown")
    print("⬇️ Downloading MVSA-single.zip...")
    gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)

# 2) Giải nén vào thư mục tạm
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

print("📦 Extracting...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(TMP_DIR)

# 3) Chuẩn hoá thành data/MVSA-Single
# Các tình huống thường gặp sau khi giải nén:
# - TMP_DIR/MVSA_Single/data/...
# - TMP_DIR/MVSA-Single/data/...
# - TMP_DIR/data/...
candidates = [d for d in os.listdir(TMP_DIR) if os.path.isdir(os.path.join(TMP_DIR, d))]
src_dir = None

# Ưu tiên thư mục bắt đầu bằng 'MVSA'
for name in candidates:
    lower = name.lower()
    if lower.startswith("mvsa"):
        src_dir = os.path.join(TMP_DIR, name)
        break

# Nếu không có, kiểm tra thư mục 'data' trực tiếp bên trong TMP_DIR
if src_dir is None and os.path.isdir(os.path.join(TMP_DIR, "data")):
    # Tạo đích dạng data/MVSA-Single và di chuyển 'data' vào trong
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR, exist_ok=True)
    shutil.move(os.path.join(TMP_DIR, "data"), os.path.join(DEST_DIR, "data"))
else:
    # Có src_dir (MVSA_Single hoặc MVSA-Single)
    if src_dir is None:
        # fallback: di chuyển toàn bộ TMP_DIR vào DEST_DIR
        if os.path.exists(DEST_DIR):
            shutil.rmtree(DEST_DIR)
        shutil.move(TMP_DIR, DEST_DIR)
    else:
        if os.path.exists(DEST_DIR):
            shutil.rmtree(DEST_DIR)
        shutil.move(src_dir, DEST_DIR)

# 4) Dọn dẹp TMP_DIR nếu vẫn còn
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)

print(f"✅ Done. Dataset is at: {DEST_DIR}")
print(f"Images should be at: {os.path.join(DEST_DIR, 'data')}/<id>.jpg")