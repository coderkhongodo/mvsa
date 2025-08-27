import os
import zipfile
import shutil

# Optional: chỉ dùng gdown nếu cần tải MVSA (giữ tương thích cũ)
try:
    import gdown
except ImportError:
    gdown = None

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Hai chế độ hỗ trợ: 1) ViClickbait-2025 (dataset.zip) 2) MVSA-Single (MVSA-single.zip)
DATASET_ZIP = os.path.join(ROOT, "dataset.zip")  # ViClickbait-2025
MVSA_ZIP = os.path.join(ROOT, "MVSA-single.zip")

VICLICK_DEST = os.path.join(DATA_DIR, "ViClickbait-2025")
MVSA_DEST = os.path.join(DATA_DIR, "MVSA-Single")

# Thư mục tạm theo chế độ
TMP_DIR = os.path.join(DATA_DIR, "_extract_tmp")

# ID tải cho MVSA (nếu cần)
FILE_ID_MVSA = "1zdoXpFXLzqsP_E-WO6JgNJxviInYMdDe"

def _clean_tmp(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)

def _ensure_empty_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def _detect_src_root(tmp_dir: str) -> str:
    # Nếu giải nén ra 1 thư mục duy nhất -> chọn thư mục đó; nếu nhiều mục hoặc có file rời -> dùng tmp_dir làm gốc
    entries = os.listdir(tmp_dir)
    dirs = [d for d in entries if os.path.isdir(os.path.join(tmp_dir, d))]
    files = [f for f in entries if os.path.isfile(os.path.join(tmp_dir, f))]
    if len(dirs) == 1 and len(files) == 0:
        return os.path.join(tmp_dir, dirs[0])
    return tmp_dir

def _extract_zip(zip_path: str, tmp_dir: str):
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

def _move_all(src_root: str, dest_dir: str):
    _ensure_empty_dir(dest_dir)
    for name in os.listdir(src_root):
        src_path = os.path.join(src_root, name)
        dst_path = os.path.join(dest_dir, name)
        shutil.move(src_path, dst_path)

def _extract_viclickbait():
    # 1) Kiểm tra zip
    if not os.path.exists(DATASET_ZIP):
        raise RuntimeError("Không tìm thấy dataset.zip ở thư mục gốc dự án. Hãy đặt file vào cùng thư mục với script.")
    # 2) Giải nén
    _clean_tmp(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    _extract_zip(DATASET_ZIP, TMP_DIR)
    # 3) Xác định gốc nội dung và di chuyển toàn bộ vào data/ViClickbait-2025
    src_root = _detect_src_root(TMP_DIR)
    _move_all(src_root, VICLICK_DEST)
    # 4) Dọn dẹp
    _clean_tmp(TMP_DIR)
    # 5) Thông tin
    print(f"✅ Done. Dataset is at: {VICLICK_DEST}")
    print("Contents:")
    try:
        print("- CSV/JSON:", [f for f in os.listdir(VICLICK_DEST) if f.lower().endswith((".csv",".json"))])
        img_dirs = [d for d in os.listdir(VICLICK_DEST) if os.path.isdir(os.path.join(VICLICK_DEST, d))]
        print("- Dirs:", img_dirs)
    except Exception:
        pass

def _extract_mvsa():
    # 1) Có sẵn zip thì dùng, không có thì tải (giữ hành vi cũ)
    if not os.path.exists(MVSA_ZIP):
        if gdown is None:
            raise RuntimeError("MVSA-single.zip không tồn tại và gdown chưa được cài. Hãy cài: pip install gdown")
        print("⬇️ Downloading MVSA-single.zip...")
        gdown.download(id=FILE_ID_MVSA, output=MVSA_ZIP, quiet=False)
    # 2) Giải nén vào thư mục tạm
    _clean_tmp(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    _extract_zip(MVSA_ZIP, TMP_DIR)
    # 3) Chuẩn hoá thành data/MVSA-Single (giữ logic tương thích cũ)
    candidates = [d for d in os.listdir(TMP_DIR) if os.path.isdir(os.path.join(TMP_DIR, d))]
    src_dir = None
    for name in candidates:
        lower = name.lower()
        if lower.startswith("mvsa"):
            src_dir = os.path.join(TMP_DIR, name)
            break
    if src_dir is None and os.path.isdir(os.path.join(TMP_DIR, "data")):
        if os.path.exists(MVSA_DEST):
            shutil.rmtree(MVSA_DEST)
        os.makedirs(MVSA_DEST, exist_ok=True)
        shutil.move(os.path.join(TMP_DIR, "data"), os.path.join(MVSA_DEST, "data"))
    else:
        if src_dir is None:
            if os.path.exists(MVSA_DEST):
                shutil.rmtree(MVSA_DEST)
            shutil.move(TMP_DIR, MVSA_DEST)
        else:
            if os.path.exists(MVSA_DEST):
                shutil.rmtree(MVSA_DEST)
            shutil.move(src_dir, MVSA_DEST)
    # 4) Dọn tạm
    _clean_tmp(TMP_DIR)
    print(f"✅ Done. Dataset is at: {MVSA_DEST}")
    print(f"Images should be at: {os.path.join(MVSA_DEST, 'data')}/<id>.jpg")

if __name__ == "__main__":
    # Ưu tiên giải nén ViClickbait-2025 nếu có dataset.zip; ngược lại rơi về MVSA
    if os.path.exists(DATASET_ZIP):
        _extract_viclickbait()
    else:
        _extract_mvsa()