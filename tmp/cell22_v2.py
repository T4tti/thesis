# Generation v2: Tiered model selection + JS Divergence + Ordinal MixUp fallback
#
# Thứ tự ưu tiên chọn model:
#   1. per-class TimeGAN (nếu class đủ windows để train)
#   2. sector-conditional TimeGAN (cùng ngành)
#   3. global TimeGAN (tất cả priority classes)
#   4. Ordinal-aware MixUp (last resort, không cần TimeGAN)
#
# Bộ lọc chất lượng:
#   - JS Divergence < 0.20: loại synthetic sequence quá xa phân phối thực
#   - RF Accuracy >= 0.50: giữ sequence được classifier nhận đúng label

from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import jensenshannon

# ── [Step 1] RF Quality Classifier ────────────────────────────────────────
print("=== [1] RF Quality Classifier ===")
quality_clf = RandomForestClassifier(
    n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1,
    class_weight='balanced', max_depth=15, min_samples_leaf=2
)
quality_clf.fit(
    train[TIMEGAN_FEATURE_COLUMNS].fillna(0),
    train[CONFIG['target_column']]
)
print(f"RF trained on {len(train)} samples")

# ── [Step 2] Per-class reference distributions for JS divergence ──────────
print("\n[2] Building reference distributions for JS filter...")
bins_cfg = {}
for col in TIMEGAN_NUMERIC_COLUMNS:
    lo = float(train[col].quantile(0.01))
    hi = float(train[col].quantile(0.99))
    bins_cfg[col] = np.linspace(lo - 1e-9, hi + 1e-9, 31)

REF_DISTS = {}
for lbl in sorted(train[CONFIG['target_column']].unique()):
    cd = train[train[CONFIG['target_column']] == lbl]
    ch = {}
    for col, bins in bins_cfg.items():
        if col in cd.columns:
            h, _ = np.histogram(cd[col].dropna(), bins=bins, density=True)
            h = h + 1e-8
            h /= h.sum()
            ch[col] = h
    REF_DISTS[int(lbl)] = ch
print(f"Reference distributions built for {len(REF_DISTS)} classes")


def js_score_window(win_df: pd.DataFrame, cls: int) -> float:
    """Average JS divergence between a synthetic window and class reference."""
    if int(cls) not in REF_DISTS:
        return 0.0
    ref = REF_DISTS[int(cls)]
    vals = []
    for col, rh in ref.items():
        if col not in win_df.columns:
            continue
        h, _ = np.histogram(win_df[col].dropna(), bins=bins_cfg[col], density=True)
        h = h + 1e-8
        h /= h.sum()
        vals.append(float(jensenshannon(rh, h)))
    return float(np.mean(vals)) if vals else 0.0


def ordinal_mixup(
    train_df: pd.DataFrame,
    cls: int,
    n_needed: int,
    feat_cols: list,
    tgt_col: str,
    alpha: float = 0.3,
    adjacent: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Ordinal-aware MixUp: interpolate between class `cls` and adjacent classes.
    Only mix class i with i+/-1 to preserve ordinal semantics.
    """
    rng = np.random.default_rng(seed)
    cls_a = train_df[train_df[tgt_col] == cls][feat_cols].dropna()
    if len(cls_a) == 0:
        return pd.DataFrame()

    partners = []
    if adjacent:
        for adj in [cls - 1, cls + 1]:
            p = train_df[train_df[tgt_col] == adj][feat_cols].dropna()
            if len(p) > 0:
                partners.append(p)
    if not partners:
        partners.append(cls_a)  # same-class fallback with noise

    rows = []
    for _ in range(n_needed):
        a = cls_a.sample(1, replace=True, random_state=None).values[0]
        b = partners[int(rng.integers(0, len(partners)))].sample(
            1, replace=True, random_state=None
        ).values[0]
        lam = float(rng.beta(alpha, alpha))
        rows.append(lam * a + (1.0 - lam) * b)

    out = pd.DataFrame(rows, columns=feat_cols)
    out[tgt_col] = cls
    return out


# ── [Step 3] Main Generation Loop ─────────────────────────────────────────
print("\n[3] Generating synthetic samples (TimeGAN + MixUp fallback)...")

tgt      = CONFIG['target_column']
dtc      = CONFIG['date_column']
ent      = CONFIG['entity_column']
sl       = int(CONFIG['seq_len'])
js_t     = float(CONFIG.get('js_divergence_threshold', 0.20))
rf_t     = float(CONFIG.get('rf_quality_threshold', 0.50))
use_mix  = bool(CONFIG.get('enable_ordinal_mixup_fallback', True))
mix_a    = float(CONFIG.get('mixup_alpha', 0.3))
mix_adj  = bool(CONFIG.get('mixup_adjacent_only', True))
rng_main = np.random.default_rng(int(CONFIG['random_seed']))


def _refdt(series):
    d = pd.to_datetime(series, errors='coerce').dropna()
    return d.max() if len(d) > 0 else pd.Timestamp(datetime.now().date())


base_dt = _refdt(train[dtc]) if dtc in train.columns else pd.Timestamp(datetime.now().date())


def oversample_factor(rc: int) -> float:
    """Extra generation multiplier to compensate for JS+RF filter losses."""
    if rc < 5:  return 30.0
    if rc < 15: return 20.0
    if rc < 30: return 12.0
    if rc < 60: return 6.0
    return 3.5


gen_rows = []     # rows from TimeGAN (accepted after filters)
mix_rows = []     # rows from Ordinal MixUp
gen_log  = []
wcnt     = 0      # global window counter for unique ticker generation

for cls, nw in sorted(windows_per_class.items()):
    if int(nw) <= 0:
        continue

    real_c  = int(class_counts.get(int(cls), 0))
    need_r  = int(samples_per_class.get(int(cls), 0))

    # ── Model selection: per-class > sector > global ──────────────────────
    mk  = f'class_{int(cls)}'
    sk  = SECTOR_FALLBACK_MAP.get(int(cls), '')
    if mk in TIMEGAN_MODELS:
        mdl = TIMEGAN_MODELS[mk]; msrc = mk
    elif sk and sk in TIMEGAN_MODELS:
        mdl = TIMEGAN_MODELS[sk]; msrc = sk
    elif 'global' in TIMEGAN_MODELS:
        mdl = TIMEGAN_MODELS['global']; msrc = 'global'
    else:
        mdl = None; msrc = None

    cls_gen = []   # rows accepted for this class from TimeGAN

    if mdl is not None:
        try:
            ovf      = oversample_factor(real_c)
            over_nw  = int(np.ceil(nw * ovf))
            print(f"\nCls {cls} (real={real_c}, need={need_r}): "
                  f"request {over_nw} windows via {msrc} ({ovf}x)")

            sn = safe_timegan_sample(mdl, over_nw)
            if sn.shape[0] == 0:
                print("  WARNING: model returned 0 windows")
            else:
                sn = sn[:, :sl, :len(TIMEGAN_FEATURE_COLUMNS)]
                sr = inverse_timegan_scale(sn, TIMEGAN_SCALER, TIMEGAN_FEATURE_COLUMNS)

                cls_meta = TIMEGAN_SEQUENCE_META[
                    TIMEGAN_SEQUENCE_META['window_label'] == int(cls)
                ]
                sdts = pd.to_datetime(
                    cls_meta.get('start_date', pd.Series([], dtype='datetime64[ns]')),
                    errors='coerce'
                ).dropna().tolist()
                if len(sdts) == 0:
                    sdts = [base_dt]

                ok = 0; rj = 0; rrf = 0
                for wi in range(sr.shape[0]):
                    blk = sr[wi]
                    rd  = sdts[wi % len(sdts)]
                    if pd.isna(rd):
                        rd = base_dt + pd.Timedelta(days=30 * (wcnt + 1))
                    stk = f"SYN_{int(cls)}_{wcnt:07d}"

                    # Build window rows tentatively
                    wr = []
                    for si in range(sl):
                        rd2 = {c: float(blk[si, ci])
                               for ci, c in enumerate(TIMEGAN_FEATURE_COLUMNS)}
                        rd2[tgt] = int(cls)
                        if ent in train.columns:
                            rd2[ent] = stk
                        if dtc in train.columns:
                            rd2[dtc] = pd.to_datetime(rd) + pd.Timedelta(days=90 * si)
                        wr.append(rd2)

                    wdf = pd.DataFrame(wr)

                    # JS Divergence filter
                    js_val = js_score_window(wdf, int(cls))
                    if js_val > js_t:
                        rj += 1; wcnt += 1
                        continue

                    # RF quality filter
                    preds = quality_clf.predict(wdf[TIMEGAN_FEATURE_COLUMNS].fillna(0))
                    if float(np.mean(preds == int(cls))) < rf_t:
                        rrf += 1; wcnt += 1
                        continue

                    cls_gen.extend(wr)
                    ok += 1; wcnt += 1
                    if ok >= nw:
                        break

                gen_log.append({
                    'cls': int(cls), 'req': nw, 'acc': ok,
                    'rj_js': rj, 'rj_rf': rrf, 'src': msrc
                })
                print(f"  Accepted={ok}  JS-rejected={rj}  RF-rejected={rrf}")
        except Exception as e:
            print(f"  TimeGAN FAILED: {e}")

    gen_rows.extend(cls_gen)

    # ── Ordinal MixUp fallback: fill remaining gap ─────────────────────────
    gap = need_r - len(cls_gen)
    if gap > 0 and use_mix:
        print(f"  -> MixUp fallback: {gap} more rows needed for cls {cls}")
        mdf = ordinal_mixup(
            train_df=train,
            cls=int(cls),
            n_needed=gap,
            feat_cols=TIMEGAN_FEATURE_COLUMNS,
            tgt_col=tgt,
            alpha=mix_a,
            adjacent=mix_adj,
            seed=int(CONFIG['random_seed'])
        )
        if len(mdf) > 0:
            for i, row in mdf.iterrows():
                r = row.to_dict()
                r['is_mixup'] = 1
                if ent in train.columns:
                    r[ent] = f"MIX_{int(cls)}_{i:07d}"
                if dtc in train.columns:
                    r[dtc] = base_dt + pd.Timedelta(days=90 * i)
                mix_rows.append(r)
            print(f"  MixUp: created {len(mdf)} rows")
    elif gap > 0:
        print(f"  WARNING: gap={gap} rows for cls {cls}, MixUp disabled")

# ── Combine all synthetic rows ─────────────────────────────────────────────
all_rows = gen_rows + mix_rows
if len(all_rows) == 0:
    synthetic_df = pd.DataFrame()
    print("\nWARNING: 0 synthetic rows generated")
else:
    synthetic_df = pd.DataFrame(all_rows)
    if 'is_mixup' not in synthetic_df.columns:
        synthetic_df['is_mixup'] = 0
    synthetic_df['is_mixup'] = synthetic_df['is_mixup'].fillna(0).astype(int)

    print(f"\nTotal synthetic: {len(synthetic_df)}")
    print(f"  TimeGAN rows: {len(gen_rows)}")
    print(f"  MixUp rows:   {len(mix_rows)}")
    print("\nClass distribution (synthetic):")
    print(synthetic_df[tgt].value_counts().sort_index())

TIMEGAN_GENERATION_LOG = gen_log
