# Calculate TimeGAN sampling plan (row-level target -> window-level generation)
full_class_range = list(range(int(CONFIG['target_min_label']), int(CONFIG['target_max_label']) + 1))
class_counts = (
    train[CONFIG['target_column']]
    .value_counts()
    .reindex(full_class_range, fill_value=0)
    .sort_index()
    .astype(int)
)

nonzero_counts = class_counts[class_counts > 0]
median_count = float(nonzero_counts.median()) if len(nonzero_counts) > 0 else 1.0

print("=== TimeGAN Sampling Plan ===")
print("\nOriginal class counts:")
print(class_counts)
print(f"\nMedian class count (non-zero classes): {median_count:.0f}")

priority_classes = [int(c) for c in CONFIG.get('priority_class_labels', [])]
priority_classes_present = [c for c in priority_classes if c in class_counts.index]
print(f"\nPriority classes: {priority_classes_present}")

def allocate_with_budget(need_map, budget):
    """Scale class-wise needs to fit total synthetic budget."""
    if budget <= 0 or sum(need_map.values()) <= 0:
        return {k: 0 for k in need_map}

    total_need = sum(need_map.values())
    if total_need <= budget:
        return need_map.copy()

    scaled = {}
    remainders = []
    allocated = 0

    for cls, need in need_map.items():
        exact = budget * (need / total_need)
        flo = int(np.floor(exact))
        scaled[cls] = flo
        allocated += flo
        remainders.append((exact - flo, cls))

    remaining = budget - allocated
    for _, cls in sorted(remainders, reverse=True)[:remaining]:
        scaled[cls] += 1

    return scaled

base_target = int(round(median_count))
priority_boost_multiplier = float(CONFIG.get('priority_boost_multiplier', 1.5))
priority_min_count = int(CONFIG.get('priority_min_count', 200))
priority_max_factor = float(CONFIG.get('priority_max_oversample_factor', 8.0))

raw_need = {}
for class_label, count in class_counts.items():
    count = int(count)
    target_count = int(base_target)

    if int(class_label) in priority_classes_present:
        boosted_target = int(round(base_target * priority_boost_multiplier))
        boosted_target = max(boosted_target, priority_min_count)

        if count > 0:
            cap = int(np.ceil(count * priority_max_factor))
            target_count = min(boosted_target, cap)
        else:
            target_count = boosted_target
    else:
        if count == 0:
            target_count = 0

    raw_need[int(class_label)] = max(0, target_count - count)

max_total_synthetic = int(len(train) * float(CONFIG['max_synthetic_ratio']))
samples_per_class = allocate_with_budget(raw_need, max_total_synthetic)
total_synthetic_samples = int(sum(samples_per_class.values()))

# Convert row-level demand to sequence-window demand
windows_per_class = {
    int(cls): int(np.ceil(n_rows / max(1, int(CONFIG['seq_len']))))
    for cls, n_rows in samples_per_class.items()
    if int(n_rows) > 0
}

expected_counts = class_counts.add(pd.Series(samples_per_class), fill_value=0).astype(int)

print(f"\nBase target: {base_target}")
print(f"Synthetic budget (rows): {max_total_synthetic}")
print(f"Synthetic planned (rows): {total_synthetic_samples}")
print(f"Synthetic planned ratio: {total_synthetic_samples / len(train):.1%}")

print("\nPlanned synthetic rows per class:")
for cls in sorted(samples_per_class.keys()):
    if samples_per_class[cls] > 0:
        model_key = f'class_{cls}'
        availability = 'class-model' if model_key in TIMEGAN_MODELS else 'global-fallback'
        print(f"  Class {cls}: {samples_per_class[cls]} rows ({windows_per_class.get(cls, 0)} windows) -> {availability}")

print("\nExpected class counts after augmentation:")
print(expected_counts.sort_index())
if len(nonzero_counts) > 0 and nonzero_counts.max() > 0:
    before_ratio = nonzero_counts.min() / nonzero_counts.max()
else:
    before_ratio = np.nan
nonzero_ec = expected_counts[expected_counts > 0]
after_ratio = nonzero_ec.min() / nonzero_ec.max() if len(nonzero_ec) > 0 and nonzero_ec.max() > 0 else np.nan
print(f"\nImbalance ratio before: {before_ratio:.4f}" if np.isfinite(before_ratio) else "\nImbalance ratio before: N/A")
print(f"Imbalance ratio after:  {after_ratio:.4f}" if np.isfinite(after_ratio) else "Imbalance ratio after:  N/A")