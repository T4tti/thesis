# Sampling Plan v2 — Log-Ratio Balancing
# Changes vs v1:
# - target_i = geometric_mean(count_i, P75_count) for non-priority
# - priority classes still boosted but capped at priority_max_oversample_factor
# - budget allocation unchanged (Hamilton method)

full_class_range = list(range(int(CONFIG['target_min_label']),
                               int(CONFIG['target_max_label']) + 1))
class_counts = (
    train[CONFIG['target_column']].value_counts()
    .reindex(full_class_range, fill_value=0).sort_index().astype(int)
)
nonzero_counts = class_counts[class_counts > 0]
median_count = float(nonzero_counts.median()) if len(nonzero_counts) > 0 else 1.0
p75_count    = float(nonzero_counts.quantile(0.75)) if len(nonzero_counts) > 0 else 1.0

print("=== Sampling Plan v2 (Log-Ratio Balancing) ===")
print(class_counts)
print(f"\nMedian={median_count:.0f}  P75={p75_count:.0f}")

priority_cls_list = [int(c) for c in CONFIG.get('priority_class_labels', [])]
priority_cls_present = [c for c in priority_cls_list
                        if c in class_counts.index and class_counts[c] > 0]

strategy    = CONFIG.get('balance_strategy', 'log_ratio')
prio_boost  = float(CONFIG.get('priority_boost_multiplier', 1.5))
prio_min    = int(CONFIG.get('priority_min_count', 300))
prio_maxf   = float(CONFIG.get('priority_max_oversample_factor', 150.0))

print(f"\nBalance strategy: '{strategy}'")
print(f"Priority classes present: {priority_cls_present}")


def compute_target(cls_label: int, real_cnt: int) -> int:
    """
    Compute augmentation target for a class.

    Strategies:
    - log_ratio : target = geometric_mean(count, P75), capped at P75 for majority
    - sqrt_flat : target = sqrt(count * median)
    - median_flat: target = median (v1 behavior)
    """
    is_p = int(cls_label) in priority_cls_list
    cnt  = max(real_cnt, 0)

    if strategy == 'log_ratio':
        # Geometric mean between count and P75
        base = int(np.sqrt(cnt * p75_count)) if cnt > 0 else int(p75_count)
        if not is_p:
            base = min(base, int(p75_count))  # cap majority at P75
    elif strategy == 'sqrt_flat':
        base = int(np.sqrt(cnt * median_count)) if cnt > 0 else int(median_count)
    else:  # median_flat (v1 default)
        base = int(median_count)

    if is_p:
        boosted = max(int(base * prio_boost), prio_min)
        cap = int(np.ceil(cnt * prio_maxf)) if cnt > 0 else boosted
        base = min(boosted, cap)
    elif cnt == 0:
        base = 0   # absent class: skip

    return max(0, base)


def allocate_budget(need_map: dict, budget: int) -> dict:
    """Allocate synthetic budget proportionally (Hamilton method)."""
    if budget <= 0 or sum(need_map.values()) <= 0:
        return {k: 0 for k in need_map}
    total = sum(need_map.values())
    if total <= budget:
        return need_map.copy()
    scaled = {}; rems = []; alloc = 0
    for k, v in need_map.items():
        ex = budget * (v / total)
        fl = int(np.floor(ex))
        scaled[k] = fl; alloc += fl
        rems.append((ex - fl, k))
    for _, k in sorted(rems, reverse=True)[:budget - alloc]:
        scaled[k] += 1
    return scaled


raw_need = {
    int(c): max(0, compute_target(int(c), int(v)) - int(v))
    for c, v in class_counts.items()
}

max_synth = int(len(train) * float(CONFIG['max_synthetic_ratio']))
samples_per_class = allocate_budget(raw_need, max_synth)
total_synth = int(sum(samples_per_class.values()))

windows_per_class = {
    int(c): int(np.ceil(n / max(1, CONFIG['seq_len'])))
    for c, n in samples_per_class.items() if int(n) > 0
}

expected_counts = class_counts.add(pd.Series(samples_per_class), fill_value=0).astype(int)

print(f"\nBudget: {max_synth} | Planned: {total_synth} ({total_synth/len(train):.1%} of train)")
print(f"\n{'Class':>6} {'Real':>7} {'Target':>8} {'Need':>8} {'Windows':>9}  Model-Source")
print("-" * 65)
for cls in sorted(samples_per_class):
    nd = samples_per_class[cls]
    if nd <= 0:
        continue
    rl  = int(class_counts.get(cls, 0))
    tg  = compute_target(int(cls), rl)
    wn  = windows_per_class.get(cls, 0)
    mk  = f'class_{cls}'
    sk  = SECTOR_FALLBACK_MAP.get(int(cls), '')
    if mk in TIMEGAN_MODELS:
        src = 'per-class'
    elif sk and sk in TIMEGAN_MODELS:
        src = f'sector({sk})'
    elif 'global' in TIMEGAN_MODELS:
        src = 'global'
    else:
        src = 'NONE !'
    print(f"{cls:>6} {rl:>7} {tg:>8} {nd:>8} {wn:>9}  {src}")

print(f"\nExpected class counts after augmentation:")
print(expected_counts.sort_index())

nec = expected_counts[expected_counts > 0]
br = (nonzero_counts.min() / nonzero_counts.max()
      if len(nonzero_counts) > 0 and nonzero_counts.max() > 0 else float('nan'))
ar = (nec.min() / nec.max()
      if len(nec) > 0 and nec.max() > 0 else float('nan'))
print(f"\nImbalance ratio before: {br:.4f}" if np.isfinite(br) else "\nImbalance before: N/A")
print(f"Expected ratio after:   {ar:.4f}" if np.isfinite(ar) else "Expected after:  N/A")
