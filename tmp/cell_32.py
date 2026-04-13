# Visualize before/after augmentation
full_class_index = list(range(int(CONFIG['target_min_label']), int(CONFIG['target_max_label']) + 1))

before_counts = (
    train[CONFIG['target_column']]
    .value_counts()
    .sort_index()
    .reindex(full_class_index, fill_value=0)
    .astype(int)
)
after_counts = (
    train_augmented[CONFIG['target_column']]
    .value_counts()
    .sort_index()
    .reindex(full_class_index, fill_value=0)
    .astype(int)
)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Before
before_counts.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Before TimeGAN Augmentation')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')

# After
after_counts.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('After TimeGAN Augmentation')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'class_distribution_timegan_comparison.png', dpi=150, bbox_inches='tight')
plt.show()