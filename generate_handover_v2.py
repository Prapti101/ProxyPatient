"""
ProxyPatient — F5: Generate handover_v2/ package with MANIFEST.txt
Run after all v2 parquets are saved.
"""
import os, hashlib, json
import pandas as pd

BASE        = r"C:\Users\PraptiPriya\OneDrive\Desktop\data"
PROC_DIR    = os.path.join(BASE, "processed")
HANDOVER    = os.path.join(BASE, "handover_v2")
os.makedirs(HANDOVER, exist_ok=True)

FILES = [
    # Preprocessed (NOT imputed)
    "train_v2.parquet",
    "val_v2.parquet",
    "test_v2.parquet",
    # DAE imputed
    "dae_imputed_train_v2.parquet",
    "dae_imputed_val_v2.parquet",
    "dae_imputed_test_v2.parquet",
    "dae_imputed_combined_v2.parquet",
    # Full cleaned
    "combined_clean_v2.parquet",
    "women_clean_v2.parquet",
    "men_clean_v2.parquet",
    # Model artifacts
    "preprocess_v2.pkl",
    "dae_weights_v2.pt",
    "dae_fit_stats_v2.pkl",
]

import shutil

manifest = []
for fname in FILES:
    src = os.path.join(PROC_DIR, fname)
    if not os.path.exists(src):
        print(f"  MISSING: {fname}")
        manifest.append({
            "file": fname, "status": "MISSING",
            "shape": None, "size_mb": None,
            "columns": None, "sha256": None
        })
        continue
    # Copy to handover dir
    dst = os.path.join(HANDOVER, fname)
    shutil.copy2(src, dst)
    # Compute SHA-256
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    sha = h.hexdigest()
    size_mb = os.path.getsize(dst) / 1e6
    # Shape and columns (for parquet only)
    shape = None
    cols  = None
    if fname.endswith(".parquet"):
        df = pd.read_parquet(dst)
        shape = list(df.shape)
        cols  = list(df.columns)
    entry = {
        "file": fname,
        "status": "OK",
        "shape": shape,
        "size_mb": round(size_mb, 2),
        "columns": cols,
        "sha256": sha,
    }
    manifest.append(entry)
    print(f"  {fname}: shape={shape} {size_mb:.1f}MB sha={sha[:16]}...")

# Save MANIFEST.txt (human-readable + JSON)
manifest_path = os.path.join(HANDOVER, "MANIFEST.txt")
with open(manifest_path, "w") as f:
    f.write("ProxyPatient — handover_v2 MANIFEST\n")
    f.write("="*60 + "\n")
    f.write("Share this directory PRIVATELY with registered team members only.\n")
    f.write("NEVER upload to GitHub, AI chats, or public drives.\n\n")
    for e in manifest:
        f.write(f"\nFile: {e['file']}\n")
        f.write(f"  Status:   {e['status']}\n")
        f.write(f"  Size:     {e['size_mb']} MB\n")
        f.write(f"  Shape:    {e['shape']}\n")
        f.write(f"  SHA-256:  {e['sha256']}\n")
        if e['columns']:
            f.write(f"  Columns:  {', '.join(e['columns'])}\n")

# Also save as JSON for programmatic verification
manifest_json = os.path.join(HANDOVER, "MANIFEST.json")
with open(manifest_json, "w") as f:
    json.dump({"manifest": manifest}, f, indent=2)

print(f"\nMANIFEST.txt saved to {manifest_path}")
print(f"MANIFEST.json saved to {manifest_json}")
print("\nF5 DONE.")
