"""Re-pack best.pt directory and re-save via torch with weights_only=False, 
then verify YOLO can load it."""
import os, zipfile, sys, torch

src = r"D:\surya\mini_project\end-end\marine_ai\backend\models\best.pt\best"
dst = r"D:\surya\mini_project\end-end\marine_ai\backend\models\best_repacked.pt"

# Step 1: zip the extracted directory back
print("Step 1: Creating ZIP...", flush=True)
with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zf:
    count = 0
    for dirpath, dirnames, filenames in os.walk(src):
        for f in filenames:
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, src).replace(os.sep, "/")
            arcname = "best/" + rel
            zf.write(full, arcname)
            count += 1
    print(f"  Added {count} files", flush=True)

# Step 2: load with weights_only=False and re-save
print("Step 2: Loading + re-saving...", flush=True)
ckpt = torch.load(dst, map_location="cpu", weights_only=False)
torch.save(ckpt, dst)
print(f"  Re-saved OK ({os.path.getsize(dst)} bytes)", flush=True)

# Step 3: monkey-patch torch.load to allow unsafe weights, then load YOLO
print("Step 3: Loading with YOLO (patched torch.load)...", flush=True)
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO
model = YOLO(dst)
print(f"  YOLO loaded OK!", flush=True)
print(f"  Class names: {model.names}", flush=True)

torch.load = _orig_torch_load  # restore
print("\nSUCCESS!", flush=True)
