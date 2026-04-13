# Analyze class distribution
print("\n=== Class Distribution ===")
full_class_index = list(range(int(CONFIG['target_min_label']), int(CONFIG['target_max_label']) + 1))

train_class_dist = (
    train[CONFIG['target_column']]
    .value_counts()
    .sort_index()
    .reindex(full_class_index, fill_value=0)
    .astype(int)
)
train_class_pct = (train_class_dist / max(train_class_dist.sum(), 1) * 100).round(2)

print("\nTrain set (count):")
print(train_class_dist)
nonzero_classes = train_class_dist[train_class_dist > 0]
if len(nonzero_classes) > 0:
    print(f"\nClass balance ratio (non-zero classes): {nonzero_classes.min() / nonzero_classes.max():.3f}")
else:
    print("\nClass balance ratio: N/A (all classes are zero)")

missing_classes = train_class_dist[train_class_dist == 0].index.tolist()
if missing_classes:
    print(f"Missing classes in train (shown as zero on chart): {missing_classes}")

# Subplot visualization: count + percentage
fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

# Left: absolute counts
train_class_dist.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black', linewidth=0.5)
axes[0].set_title('Class Distribution - Count (Train)')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
axes[0].grid(axis='y', alpha=0.25)

# Right: percentage
train_class_pct.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black', linewidth=0.5)
axes[1].set_title('Class Distribution - Percentage (Train)')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Percentage (%)')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(axis='y', alpha=0.25)

# Save + show
plt.savefig(REPORTS_DIR / 'class_distribution_before_timegan_subplot.png', dpi=150, bbox_inches='tight')
plt.show()