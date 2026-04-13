def build_timegan_model():
    """Create tsgm TimeGAN from CONFIG."""
    return TimeGAN(
        seq_len=int(CONFIG['seq_len']),
        module=str(CONFIG.get('timegan_module', 'gru')),
        hidden_dim=int(CONFIG.get('timegan_latent_dim', 24)),
        n_features=len(TIMEGAN_FEATURE_COLUMNS),
        n_layers=int(CONFIG.get('timegan_n_layers', 3)),
        batch_size=int(CONFIG['timegan_batch_size']),
        gamma=float(CONFIG['timegan_gamma'])
    )


def train_timegan_on_windows(window_data):
    """Train TimeGAN on sequence windows."""
    if len(window_data) == 0:
        raise ValueError("Empty window data")
    m = build_timegan_model()
    m.compile()
    m.fit(data=np.asarray(window_data, dtype=np.float32),
          epochs=int(CONFIG['timegan_train_steps']))
    return m


print("=== Training TimeGAN Models v2 (Sector-Conditional + Min-Window Guard) ===")
min_w = int(CONFIG.get('min_windows_per_class_for_timegan', 10))
priority_cls_set = set(int(c) for c in CONFIG.get('priority_class_labels', []))

TIMEGAN_MODELS = {}
TIMEGAN_TRAINING_REGISTRY = {
    'global': {}, 'per_class': {}, 'per_sector': {},
    'train_steps': int(CONFIG['timegan_train_steps']),
    'seq_len': int(CONFIG['seq_len']),
    'feature_columns': TIMEGAN_FEATURE_COLUMNS,
    'window_count': int(TIMEGAN_SEQUENCE_WINDOWS.shape[0])
}
SECTOR_FALLBACK_MAP = {}  # {class_label: model_key}

# ── Global fallback: all priority class windows ──
prio_mask = TIMEGAN_SEQUENCE_META['window_label'].isin(priority_cls_set)
prio_wins = TIMEGAN_SEQUENCE_WINDOWS[prio_mask]
print(f"\n[Global] Training fallback on {len(prio_wins)} priority windows...")
try:
    TIMEGAN_MODELS['global'] = train_timegan_on_windows(prio_wins)
    TIMEGAN_TRAINING_REGISTRY['global'] = {'status': 'trained', 'n': int(len(prio_wins))}
    print("[Global] Training complete")
except Exception as e:
    TIMEGAN_TRAINING_REGISTRY['global'] = {'status': 'failed', 'error': str(e)}
    print(f"[Global] FAILED: {e}")

# ── Sector-conditional fallback (v2 new) ──────────────────────────────────
sec_col = CONFIG.get('sector_column', 'sector')
if CONFIG.get('enable_sector_conditional', True) and sec_col in train.columns:
    print("\n[Sector-Conditional] Building sector-level fallback models...")
    ent_col = CONFIG['entity_column']
    if ent_col in train.columns:
        ent_sec = (
            train.groupby(ent_col)[sec_col]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')
        )
        TIMEGAN_SEQUENCE_META['sector'] = (
            TIMEGAN_SEQUENCE_META['entity'].map(ent_sec).fillna('Unknown')
        )
    else:
        TIMEGAN_SEQUENCE_META['sector'] = 'Unknown'

    for sec in TIMEGAN_SEQUENCE_META['sector'].unique():
        sec_mask = TIMEGAN_SEQUENCE_META['sector'] == sec
        sec_wins = TIMEGAN_SEQUENCE_WINDOWS[sec_mask.to_numpy()]
        if len(sec_wins) < max(10, min_w):
            TIMEGAN_TRAINING_REGISTRY['per_sector'][sec] = {
                'status': 'skipped', 'n': len(sec_wins)
            }
            continue
        safe_key = 'sector_' + re.sub(r'[^a-zA-Z0-9_]', '_', str(sec))
        try:
            TIMEGAN_MODELS[safe_key] = train_timegan_on_windows(sec_wins)
            TIMEGAN_TRAINING_REGISTRY['per_sector'][sec] = {
                'status': 'trained', 'n': len(sec_wins)
            }
            print(f"  [Sector={sec}] OK ({len(sec_wins)} windows)")
        except Exception as e:
            TIMEGAN_TRAINING_REGISTRY['per_sector'][sec] = {
                'status': 'failed', 'error': str(e)
            }
            print(f"  [Sector={sec}] FAILED: {e}")

    # Map class -> dominant sector's model
    if ent_col in train.columns:
        cls_sec = (
            train.groupby([CONFIG['target_column'], sec_col])
            .size().reset_index(name='n')
            .sort_values('n', ascending=False)
            .groupby(CONFIG['target_column']).first()[sec_col]
        )
        for clabel, sname in cls_sec.items():
            sk = 'sector_' + re.sub(r'[^a-zA-Z0-9_]', '_', str(sname))
            if sk in TIMEGAN_MODELS:
                SECTOR_FALLBACK_MAP[int(clabel)] = sk
    print(f"  Sector fallback map: {SECTOR_FALLBACK_MAP}")
else:
    print("[Sector-Conditional] Disabled or sector column absent.")

# ── Per-class TimeGAN (with min_windows guard) ──────────────────────────
print(f"\n[Per-Class] Dedicated models (min_windows={min_w})...")
for cls in sorted(priority_cls_set):
    cls_mask = TIMEGAN_SEQUENCE_META['window_label'].to_numpy() == int(cls)
    cls_wins = TIMEGAN_SEQUENCE_WINDOWS[cls_mask]
    n_ow = len(cls_wins)
    entry = {'n': n_ow, 'status': 'not_started'}

    if n_ow < min_w:
        # Insufficient data — use sector/global fallback
        fb = 'sector' if cls in SECTOR_FALLBACK_MAP else 'global'
        entry['status'] = 'skipped_insufficient'
        entry['reason'] = f'{n_ow} < {min_w}'
        print(f"  [Cls {cls}] SKIP ({n_ow} wins < {min_w}) -> {fb} fallback")
    else:
        try:
            print(f"  [Cls {cls}] Training on {n_ow} windows...")
            TIMEGAN_MODELS[f'class_{cls}'] = train_timegan_on_windows(cls_wins)
            try:
                TIMEGAN_MODELS[f'class_{cls}'].save(
                    str(MODELS_DIR / f'timegan_class_{cls}.pkl')
                )
                entry['save'] = 'ok'
            except Exception as se:
                entry['save'] = str(se)
            entry['status'] = 'trained'
            print(f"  [Cls {cls}] OK")
        except Exception as e:
            entry.update({'status': 'failed', 'error': str(e)})
            print(f"  [Cls {cls}] FAILED: {e}")

    TIMEGAN_TRAINING_REGISTRY['per_class'][str(cls)] = entry

trained_cls = [k for k, v in TIMEGAN_TRAINING_REGISTRY['per_class'].items()
               if v.get('status') == 'trained']
skipped_cls = [k for k, v in TIMEGAN_TRAINING_REGISTRY['per_class'].items()
               if 'skipped' in v.get('status', '')]
print(f"\nPer-class trained: {trained_cls}")
print(f"Per-class skipped: {skipped_cls}")
print(f"Sector fallbacks:  {list(SECTOR_FALLBACK_MAP.items())}")
