$src = "notebooks/timegan_data_preparation_kaggle.ipynb"
$dst = "notebooks/timegan_data_preparation_3groups.ipynb"
Copy-Item -Path $src -Destination $dst -Force
$content = Get-Content -Raw -Path $dst

$oldPathBlock = @"
                ""# Configure paths for Kaggle environment"",
                ""INPUT_PATH = Path('/kaggle/input/datasets/tailength/corporate-credit-rating')"",
                ""OUTPUT_PATH = Path('/kaggle/working')"",
"@
$newPathBlock = @"
                ""# Configure paths with Kaggle/local fallback"",
                ""if Path('/kaggle/input').exists():"",
                ""    INPUT_PATH = Path('/kaggle/input/datasets/tailength/corporate-credit-rating')"",
                ""    OUTPUT_PATH = Path('/kaggle/working')"",
                ""else:"",
                ""    _cwd = Path.cwd()"",
                ""    _root = _cwd if (_cwd / 'data' / 'processed').exists() else _cwd.parent"",
                ""    INPUT_PATH = _root / 'data' / 'processed'"",
                ""    OUTPUT_PATH = _root / 'archive' / 'timegan' / '__results___files'"",
"@
$content = $content.Replace($oldPathBlock, $newPathBlock)

$content = $content.Replace("'data_file': 'merged_credit_rating_common.csv'", "'data_file': 'merged_credit_rating_common_3groups.csv'")
$content = $content.Replace("'target_is_encoded_label': True, 'target_min_label': 0, 'target_max_label': 21", "'target_is_encoded_label': True, 'target_min_label': 0, 'target_max_label': 2")
$content = $content.Replace("'priority_class_labels': [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21]", "'priority_class_labels': [0, 1]")

$content = $content.Replace("# Normalize target column for encoded labels (expected range 0-21)", "# Normalize target column for encoded labels (expected range 0-2)")
$content = $content.Replace("Applying fallback mapping to labels 0-21", "Applying fallback mapping to labels 0-2")

$oldMapBlock = @"
                ""        # Full 22-class mapping observed in merged_credit_rating_common.csv (worst -> best)."",
                ""        # Labels are contiguous in [0, 21]."",
                ""        ordered_ratings = ["",
                ""            'D', 'C', 'CC', 'CCC-', 'CCC', 'CCC+',"",
                ""            'B-', 'B', 'B+',"",
                ""            'BB-', 'BB', 'BB+',"",
                ""            'BBB-', 'BBB', 'BBB+',"",
                ""            'A-', 'A', 'A+',"",
                ""            'AA-', 'AA', 'AA+', 'AAA'"",
                ""        ]"",
"@
$newMapBlock = @"
                ""        # 3-group mapping for merged_credit_rating_common_3groups.csv (worst -> best)."",
                ""        # Labels are contiguous in [0, 2]."",
                ""        ordered_ratings = ["",
                ""            'DISTRESSED', 'HY', 'IG'"",
                ""        ]"",
"@
$content = $content.Replace($oldMapBlock, $newMapBlock)

$content = $content.Replace("max_label = int(CONFIG.get('target_max_label', 21))", "max_label = int(CONFIG.get('target_max_label', 2))")

Set-Content -Path $dst -Value $content -Encoding UTF8
Write-Output "CREATED=$dst"
Write-Output "SIZE_BYTES=$((Get-Item $dst).Length)"
