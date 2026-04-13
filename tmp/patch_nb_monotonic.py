import json

filepath = r"e:\thesis\notebooks\timegan_data_preparation_kaggle.ipynb"

with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed = False

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        if "source" not in cell: continue
        source = "".join(cell["source"])
        if "def materialize_rows_as_sequences(" in source:
            print("Found materialize_rows_as_sequences function.")
            
            # Target 1: Add headroom logic after current_dt is clamped
            target1 = "        current_dt = clamp_to_train_date_range(start_dt)\n"
            replace1 = """        current_dt = clamp_to_train_date_range(start_dt)
        
        min_required_days = size
        if (train_date_max - current_dt).days < min_required_days:
            current_dt = train_date_max - pd.Timedelta(days=min_required_days)
            if current_dt < train_date_min: 
                current_dt = train_date_min
"""
            if target1 in source and "min_required_days" not in source:
                source = source.replace(target1, replace1)
                print("Applied headroom logic.")
                changed = True

            # Target 2: Cap step_days inside the loop
            target2 =  "                step_days = int(step_pool[int(rng_main.integers(0, len(step_pool)))]) if len(step_pool) > 0 else 90\n                current_dt = bounded_dt + pd.Timedelta(days=step_days)\n"
            replace2 = """                step_days = int(step_pool[int(rng_main.integers(0, len(step_pool)))]) if len(step_pool) > 0 else 90
                
                remaining_steps = size - local_idx - 1
                if remaining_steps > 0:
                    max_step_allowed = (train_date_max - bounded_dt).days - remaining_steps
                    if step_days > max_step_allowed:
                        step_days = max(1, max_step_allowed)
                else:
                    step_days = 1
                    
                current_dt = bounded_dt + pd.Timedelta(days=step_days)
"""
            if target2 in source and "remaining_steps = size - local_idx - 1" not in source:
                source = source.replace(target2, replace2)
                print("Applied step_days constraint.")
                changed = True
            
            # Repack
            new_source_lines = []
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    new_source_lines.append(line + '\n')
                elif line:
                    new_source_lines.append(line)
            
            cell["source"] = new_source_lines

if changed:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Notebook changes saved successfully.")
else:
    print("Targets not found or already patched.")
