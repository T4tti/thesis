import json
from pathlib import Path

notebooks = {
    "notebooks/lstm-baseline.ipynb": {
        "type": "sequence",
        "input_size": 8,
    },
    "notebooks/tcn-baseline.ipynb": {
        "type": "sequence",
        "input_size": 12,
    },
    "notebooks/patchtst-baseline.ipynb": {
        "type": "sequence",
        "input_size": 12,
    },
    "notebooks/xgboost-baseline.ipynb": {
        "type": "tabular",
        "input_size": 8,
    },
    "notebooks/lightgbm-baseline.ipynb": {
        "type": "tabular",
        "input_size": 8,
    }
}

def get_sequence_target(input_size):
    return f"""def build_sequences(frame, input_size={input_size}, horizon=1):
    out = {{'train': [], 'val': [], 'test': []}}
    for _, g in frame.groupby('ticker'):
        g = g.sort_values('rating_date').reset_index(drop=True)
        vals = g[MODEL_FEATURES].values.astype(np.float32)
        ys = g[TARGET_COL].values.astype(int)
        sec = g['sector_id'].values.astype(int)
        sp = g['__split__'].astype(str).str.lower().values
        n = len(g)
        if n >= input_size + horizon:
            for i in range(n - input_size - horizon + 1):
                t = i + input_size
                out[sp[t]].append((vals[i:i+input_size], int(ys[i+input_size-1]), int(sec[t]), int(ys[t])))
        else:
            t = n - 1
            x = build_padded_window(vals, t, input_size)
            out[sp[t]].append((x, int(ys[max(0, t-1)]), int(sec[t]), int(ys[t])))
    return out"""

def get_sequence_replacement(input_size):
    return f"""def build_sequences(frame, input_size={input_size}, horizon=1):
    out = {{'train': [], 'val': [], 'test': []}}
    for _, g in frame.groupby('ticker'):
        g = g.sort_values('rating_date').reset_index(drop=True)
        vals = g[MODEL_FEATURES].values.astype(np.float32)
        ys = g[TARGET_COL].values.astype(int)
        sec = g['sector_id'].values.astype(int)
        sp = g['__split__'].astype(str).str.lower().values
        n = len(g)
        for t in range(n):
            x = build_padded_window(vals, t, input_size)
            last_y_idx = max(0, t - 1)
            out[sp[t]].append((x, int(ys[last_y_idx]), int(sec[t]), int(ys[t])))
    return out"""

def get_tabular_target(input_size):
    return f"""def build_tabular_samples(frame, input_size={input_size}, horizon=1):
    out = {{'train': [], 'val': [], 'test': []}}
    for _, g in frame.groupby('ticker'):
        g = g.sort_values('rating_date').reset_index(drop=True)
        vals = g[MODEL_FEATURES].values.astype(np.float32)
        ys = g[TARGET_COL].values.astype(int)
        sec = g['sector_id'].values.astype(int)
        sp = g['__split__'].astype(str).str.lower().values
        n = len(g)
        if n >= input_size + horizon:
            for i in range(n - input_size - horizon + 1):
                t = i + input_size
                x_window = vals[i:i+input_size]
                feat = x_window.reshape(-1)
                row = np.concatenate([feat, np.array([ys[i+input_size-1], sec[t]], dtype=np.float32)])
                out[sp[t]].append((row, int(ys[t])))
        else:
            t = n - 1
            x_window = build_padded_window(vals, t, input_size)
            feat = x_window.reshape(-1)
            row = np.concatenate([feat, np.array([ys[max(0, t-1)], sec[t]], dtype=np.float32)])
            out[sp[t]].append((row, int(ys[t])))
    return out"""

def get_tabular_replacement(input_size):
    return f"""def build_tabular_samples(frame, input_size={input_size}, horizon=1):
    out = {{'train': [], 'val': [], 'test': []}}
    for _, g in frame.groupby('ticker'):
        g = g.sort_values('rating_date').reset_index(drop=True)
        vals = g[MODEL_FEATURES].values.astype(np.float32)
        ys = g[TARGET_COL].values.astype(int)
        sec = g['sector_id'].values.astype(int)
        sp = g['__split__'].astype(str).str.lower().values
        n = len(g)
        for t in range(n):
            x_window = build_padded_window(vals, t, input_size)
            feat = x_window.reshape(-1)
            last_y_idx = max(0, t - 1)
            row = np.concatenate([feat, np.array([ys[last_y_idx], sec[t]], dtype=np.float32)])
            out[sp[t]].append((row, int(ys[t])))
    return out"""

for nb_path, cfg in notebooks.items():
    path = Path("e:/thesis") / nb_path
    print(f"Processing: {nb_path}...")
    
    with open(path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    input_size = cfg["input_size"]
    if cfg["type"] == "sequence":
        target = get_sequence_target(input_size)
        replacement = get_sequence_replacement(input_size)
    else:
        target = get_tabular_target(input_size)
        replacement = get_tabular_replacement(input_size)
        
    replaced = False
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            src_str = "".join(cell.get("source", []))
            
            # Normalize whitespace/newlines for robust comparison
            norm_src = "\n".join(line.rstrip() for line in src_str.splitlines())
            norm_target = "\n".join(line.rstrip() for line in target.splitlines())
            norm_replacement = "\n".join(line.rstrip() for line in replacement.splitlines())
            
            if norm_target in norm_src:
                norm_src = norm_src.replace(norm_target, norm_replacement)
                cell["source"] = [line + "\n" for line in norm_src.splitlines()]
                replaced = True
                print("  Successfully updated cell!")
                break
                
    if not replaced:
        print(f"  Warning: Target function not found in {nb_path}! Attempting fallback matching.")
        # Let's do a more robust substring matching if line ending differences exist
        for cell in nb_data.get("cells", []):
            if cell.get("cell_type") == "code":
                src_str = "".join(cell.get("source", []))
                # Let's check if the function signature is in the cell source
                sig = "def build_sequences" if cfg["type"] == "sequence" else "def build_tabular_samples"
                if sig in src_str:
                    # Let's print out the cell source to see what matched
                    print(f"  Found function signature '{sig}' in cell. Let's do a replace of the cell content using normalized line comparison.")
                    lines = src_str.splitlines()
                    start_fn = -1
                    end_fn = -1
                    for i, line in enumerate(lines):
                        if sig in line:
                            start_fn = i
                        if start_fn != -1 and line.startswith("    return out"):
                            end_fn = i
                            break
                    if start_fn != -1 and end_fn != -1:
                        # Replace lines from start_fn to end_fn with replacement
                        new_lines = lines[:start_fn] + replacement.splitlines() + lines[end_fn+1:]
                        cell["source"] = [l + "\n" for l in new_lines]
                        replaced = True
                        print("  Successfully updated cell using fallback line-range replacement!")
                        break
                        
    if replaced:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, ensure_ascii=False, indent=1)
        print(f"  Saved {nb_path}.")
    else:
        print(f"  Error: Failed to update {nb_path}!")
