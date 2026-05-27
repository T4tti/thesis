# [AUTO] EXPORT OOF PROBABILITIES
# Lưu xác suất val và test để dùng trong Ensemble (KB7/KB8/KB9)
# Chạy cell này SAU KHI notebook đã huấn luyện xong.

import numpy as np
import torch

_MODEL_KEY = 'patchtst'

# ─ Bước 1: Forward pass để khôi phục y_val/val_proba nếu thiếu ───────────────
def _extract_and_predict(model_obj, loader_obj, device_obj):
    model_obj.eval()
    all_y, all_prob = [], []
    with torch.no_grad():
        for batch in loader_obj:
            if len(batch) == 7:
                xb, lyb, sb, _, _, yb, _ = batch
            elif len(batch) == 4:
                xb, lyb, sb, yb = batch
            elif len(batch) == 2:
                xb, yb = batch
                lyb, sb = None, None
            else:
                xb = batch[0]; yb = batch[-1]
                lyb = batch[1] if len(batch) > 2 else None
                sb  = batch[2] if len(batch) > 3 else None
            xb = xb.to(device_obj)
            if lyb is not None: lyb = lyb.to(device_obj)
            if sb  is not None: sb  = sb.to(device_obj)
            if lyb is not None and sb is not None:
                logits = model_obj(xb, lyb, sb)
            else:
                logits = model_obj(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            all_prob.append(prob)
            all_y.append(yb.cpu().numpy() if isinstance(yb, torch.Tensor) else np.array(yb))
    return np.concatenate(all_prob), np.concatenate(all_y)

if ('y_val' not in globals() or 'y_test' not in globals()
        or 'val_proba' not in globals() or 'test_proba' not in globals()):
    _device = device if 'device' in globals() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[AUTO] Trích xuất dữ liệu trên thiết bị: {_device}...')
    if 'model' in globals() and 'val_loader' in globals() and 'test_loader' in globals():
        try:
            if 'val_proba' not in globals() or 'y_val' not in globals():
                val_proba, y_val = _extract_and_predict(model, val_loader, _device)
            if 'test_proba' not in globals() or 'y_test' not in globals():
                test_proba, y_test = _extract_and_predict(model, test_loader, _device)
            print('[AUTO] Trích xuất thành công val_proba, y_val, test_proba, y_test!')
        except Exception as _e:
            print(f'[AUTO] Cảnh báo: Lỗi khi tự động trích xuất: {_e}')
    else:
        print('[AUTO] Cảnh báo: Không tìm thấy model, val_loader hoặc test_loader trong globals.')

# ─ Bước 2: Guard cứng ARTIFACT_DIR + y_val + y_test ─────────────────────────
if 'ARTIFACT_DIR' not in globals():
    raise RuntimeError("ARTIFACT_DIR chưa được định nghĩa. Hãy chạy các cell setup trước.")
if 'y_val' not in globals():
    raise RuntimeError("y_val chưa được định nghĩa và không thể tự động khôi phục.")
if 'y_test' not in globals():
    raise RuntimeError("y_test chưa được định nghĩa và không thể tự động khôi phục.")
if 'val_proba' not in globals():
    raise RuntimeError("val_proba chưa được định nghĩa và không thể tự động khôi phục.")
if 'test_proba' not in globals():
    raise RuntimeError("test_proba chưa được định nghĩa và không thể tự động khôi phục.")

val_proba  = np.asarray(val_proba,  dtype=np.float32)
test_proba = np.asarray(test_proba, dtype=np.float32)

# ─ Validate shape ────────────────────────────────────────────────────────────
assert val_proba.shape[0] == len(y_val), (
    f"Shape mismatch: val_proba {val_proba.shape} vs y_val {len(y_val)}"
)
assert test_proba.shape[0] == len(y_test), (
    f"Shape mismatch: test_proba {test_proba.shape} vs y_test {len(y_test)}"
)

# ─ Lưu file ──────────────────────────────────────────────────────────────────
_save_val    = ARTIFACT_DIR / f'{_MODEL_KEY}_val_proba.npy'
_save_test   = ARTIFACT_DIR / f'{_MODEL_KEY}_test_proba.npy'
_save_y_val  = ARTIFACT_DIR / f'{_MODEL_KEY}_y_val.npy'
_save_y_test = ARTIFACT_DIR / f'{_MODEL_KEY}_y_test.npy'

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
np.save(_save_val,    val_proba)
np.save(_save_test,   test_proba)
np.save(_save_y_val,  y_val)
np.save(_save_y_test, y_test)

print(f'[OK] Saved val_proba  → {_save_val}')
print(f'[OK] Saved test_proba → {_save_test}')
print(f'[OK] Saved y_val      → {_save_y_val}')
print(f'[OK] Saved y_test     → {_save_y_test}')