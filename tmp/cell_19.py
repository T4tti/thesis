# Save TimeGAN training registry
registry_path = MODELS_DIR / 'timegan_training_registry.json'
with open(registry_path, 'w') as f:
    json.dump(TIMEGAN_TRAINING_REGISTRY, f, indent=2, default=str)

print(f"âœ“ Training registry saved to {registry_path}")