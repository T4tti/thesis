# Hybrid Rebalancing v2 (ENABLED by default)
# Changes vs v1:
# - ENABLE_HYBRID_REBALANCING: True (was False)
# - majority_cap_percentile: 80 (was 75) — less aggressive capping
# - minority_floor: 150 (was 200) — realistic for rare classes
# - When downsampling majority: prioritize keeping real data over synthetic
# - After augmentation: also cap MixUp rows to not exceed cap_value

ENABLE_HYBRID_REBALANCING = bool(CONFIG.get('enable_hybrid_rebalancing', True))
CAP_PCT   = int(CONFIG.get('majority_cap_percentile', 80))
MIN_FLOOR = int(CONFIG.get('minority_floor', 150))

if ENABLE_HYBRID_REBALANCING and len(train_augmented) > 0:
    print("=" * 65)
    print("HYBRID REBALANCING v2  (enabled by default)")
    print("=" * 65)

    tc    = CONFIG['target_column']
    pcls  = set(int(c) for c in CONFIG.get('priority_class_labels', []))
    aug_c = train_augmented[tc].value_counts().sort_index()

    # Determine the cap from non-priority class distribution only
    np_cnt = aug_c[[i for i in aug_c.index if i not in pcls]]
    nz_np  = np_cnt[np_cnt > 0]
    if len(nz_np) > 0:
        cap = int(np.percentile(nz_np.values, CAP_PCT))
    else:
        nz_all = aug_c[aug_c > 0]
        cap = int(nz_all.quantile(0.80)) if len(nz_all) > 0 else MIN_FLOOR * 4
    cap = max(cap, MIN_FLOOR * 2)

    print(f"  Non-priority P{CAP_PCT} cap: {cap}")
    print(f"  Minority floor: {MIN_FLOOR}")
    print(f"  Priority classes (not capped): {sorted(pcls)}")

    chunks = []
    for lbl in sorted(train_augmented[tc].unique()):
        cd   = train_augmented[train_augmented[tc] == lbl].copy()
        n    = len(cd)
        is_p = int(lbl) in pcls

        if not is_p and n > cap:
            # Downsample majority: keep real data, trim synthetic first
            if 'is_synthetic' in cd.columns:
                real_d = cd[cd['is_synthetic'] == 0]
                syn_d  = cd[cd['is_synthetic'] == 1]
            else:
                real_d = cd
                syn_d  = pd.DataFrame()

            n_real = len(real_d)
            n_need = cap - n_real
            if n_need >= 0 and len(syn_d) > 0:
                syn_kept = syn_d.sample(
                    n=min(n_need, len(syn_d)), random_state=RANDOM_SEED
                )
                cd = pd.concat([real_d, syn_kept], ignore_index=True)
            elif n_need < 0:
                # Even real data exceeds cap — sample real data
                cd = real_d.sample(n=cap, random_state=RANDOM_SEED)
            print(f"  Cls {lbl} (majority): {n} -> {len(cd)} (capped at {cap})")

        elif n < MIN_FLOOR and n > 0:
            # Minority floor via duplication (last resort)
            extra = cd.sample(
                n=MIN_FLOOR - n, random_state=RANDOM_SEED, replace=True
            )
            if 'is_synthetic' in extra.columns:
                extra['is_synthetic'] = 1
            cd = pd.concat([cd, extra], ignore_index=True)
            print(f"  Cls {lbl} (minority): {n} -> {len(cd)} (floor padded)")

        else:
            status = "priority-kept" if is_p else "kept"
            print(f"  Cls {lbl}: {n} ({status})")

        chunks.append(cd)

    train_augmented = pd.concat(chunks, ignore_index=True)
    train_augmented = train_augmented.sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    fc   = train_augmented[tc].value_counts().sort_index()
    nzf  = fc[fc > 0]
    rat  = nzf.min() / nzf.max() if len(nzf) > 0 and nzf.max() > 0 else float('nan')

    print(f"\n  After rebalancing: {len(train_augmented)} samples")
    print(f"  Class range: [{nzf.min()}, {nzf.max()}]")
    if np.isfinite(rat):
        ql = ('EXCELLENT' if rat > 0.5 else
              'GOOD'      if rat > 0.3 else
              'MODERATE'  if rat > 0.1 else 'STILL IMBALANCED')
        print(f"  Imbalance ratio: {rat:.3f}  ({ql})")
    else:
        print("  Imbalance ratio: N/A")

    print("\n  Full distribution after rebalancing:")
    print(fc)

else:
    print("\nHybrid rebalancing: SKIPPED (disabled in CONFIG or no data)")
