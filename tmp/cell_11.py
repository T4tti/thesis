if not preprocessed:

    print("\n" + "=" * 70)

    print("PERFORMING PREPROCESSING PIPELINE")

    print("=" * 70)



    # Identify column types

    NUMERIC_COLS = train.select_dtypes(include=[np.number]).columns.tolist()

    CATEGORICAL_COLS = train.select_dtypes(include=['object', 'category']).columns.tolist()



    # Remove identifier and target columns

    IDENTIFIER_COLS = []

    if 'company_name' in train.columns:

        IDENTIFIER_COLS.append('company_name')

    if 'ticker' in train.columns:

        IDENTIFIER_COLS.append('ticker')

    if CONFIG['date_column'] in train.columns:

        IDENTIFIER_COLS.append(CONFIG['date_column'])



    FEATURE_NUMERIC = [c for c in NUMERIC_COLS if c not in [CONFIG['target_column']] + IDENTIFIER_COLS]

    FEATURE_CATEGORICAL = [c for c in CATEGORICAL_COLS if c not in [CONFIG['target_column']] + IDENTIFIER_COLS]



    print(f"\nâœ“ Feature identification:")

    print(f"  Numeric features: {len(FEATURE_NUMERIC)}")

    print(f"  Categorical features: {len(FEATURE_CATEGORICAL)}")



    PREPROCESSING_META = {

        'numeric_features': FEATURE_NUMERIC,

        'categorical_features': FEATURE_CATEGORICAL,

        'transformations': {}

    }



    # 1. Missing value imputation

    print(f"\n=== Missing Value Imputation ===")

    numeric_imputer = SimpleImputer(strategy='median')

    train[FEATURE_NUMERIC] = numeric_imputer.fit_transform(train[FEATURE_NUMERIC])

    val[FEATURE_NUMERIC] = numeric_imputer.transform(val[FEATURE_NUMERIC])

    test[FEATURE_NUMERIC] = numeric_imputer.transform(test[FEATURE_NUMERIC])

    print("âœ“ Numeric imputation (median)")



    if FEATURE_CATEGORICAL:

        categorical_imputer = SimpleImputer(strategy='constant', fill_value='__MISSING__')

        train[FEATURE_CATEGORICAL] = categorical_imputer.fit_transform(train[FEATURE_CATEGORICAL])

        val[FEATURE_CATEGORICAL] = categorical_imputer.transform(val[FEATURE_CATEGORICAL])

        test[FEATURE_CATEGORICAL] = categorical_imputer.transform(test[FEATURE_CATEGORICAL])

        print("âœ“ Categorical imputation (__MISSING__)")



    # 2. Log transformation

    print(f"\n=== Log Transformation for Skewness Reduction ===")

    SKEWNESS_THRESHOLD = 1.0

    LOG_TRANSFORM_META = {}



    for col in FEATURE_NUMERIC:

        pre_skew = stats.skew(train[col].dropna())



        if abs(pre_skew) < SKEWNESS_THRESHOLD:

            transform_type = 'none'

        elif (train[col] >= 0).all():

            transform_type = 'log1p'

            train[col] = np.log1p(train[col])

            val[col] = np.log1p(val[col])

            test[col] = np.log1p(test[col])

        else:

            transform_type = 'signed_log1p'

            train[col] = np.sign(train[col]) * np.log1p(np.abs(train[col]))

            val[col] = np.sign(val[col]) * np.log1p(np.abs(val[col]))

            test[col] = np.sign(test[col]) * np.log1p(np.abs(test[col]))



        post_skew = stats.skew(train[col].dropna())

        LOG_TRANSFORM_META[col] = {

            'transform_type': transform_type,

            'pre_skewness': float(pre_skew),

            'post_skewness': float(post_skew)

        }



    transformed_cols = [k for k, v in LOG_TRANSFORM_META.items() if v['transform_type'] != 'none']

    print(f"âœ“ Log transformation applied ({len(transformed_cols)} columns)")

    PREPROCESSING_META['transformations']['log_transform'] = LOG_TRANSFORM_META



    # 3. Scaling

    print(f"\n=== Scaling ===")

    scaler = StandardScaler()

    train[FEATURE_NUMERIC] = scaler.fit_transform(train[FEATURE_NUMERIC])

    val[FEATURE_NUMERIC] = scaler.transform(val[FEATURE_NUMERIC])

    test[FEATURE_NUMERIC] = scaler.transform(test[FEATURE_NUMERIC])

    print("âœ“ StandardScaler applied")



    # 4. Categorical encoding

    if FEATURE_CATEGORICAL:

        print(f"\n=== Categorical Encoding ===")

        CATEGORICAL_MAPPINGS = {}



        for col in FEATURE_CATEGORICAL:

            le = LabelEncoder()

            train[col] = train[col].astype(str)

            val[col] = val[col].astype(str)

            test[col] = test[col].astype(str)



            # Fit on train plus fallback token to avoid unseen-category crashes

            train_with_missing = pd.concat([train[col], pd.Series(['__MISSING__'])], ignore_index=True)

            le.fit(train_with_missing)



            train[col] = le.transform(train[col])

            val[col] = val[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')

            test[col] = test[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')

            val[col] = le.transform(val[col])

            test[col] = le.transform(test[col])



            CATEGORICAL_MAPPINGS[col] = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))



        PREPROCESSING_META['transformations']['categorical_encoding'] = CATEGORICAL_MAPPINGS

        print(f"âœ“ Label encoding applied ({len(FEATURE_CATEGORICAL)} columns)")



    # Save preprocessing metadata

    with open(MODELS_DIR / 'preprocessing_metadata.json', 'w') as f:

        json.dump(PREPROCESSING_META, f, indent=2, default=str)

    print(f"\nâœ“ Preprocessing metadata saved")



    # Save preprocessed splits

    train.to_csv(SPLITS_DIR / 'train.csv', index=False)

    val.to_csv(SPLITS_DIR / 'val.csv', index=False)

    test.to_csv(SPLITS_DIR / 'test.csv', index=False)

    print(f"âœ“ Preprocessed splits saved to {SPLITS_DIR}")



    print(f"\n{'='*70}")

    print("PREPROCESSING COMPLETE")

    print(f"{'='*70}")

else:

    print("\nâœ“ Data already preprocessed, skipping preprocessing steps")



# Identify features for later use

FEATURE_NUMERIC = PREPROCESSING_META['numeric_features']

FEATURE_CATEGORICAL = PREPROCESSING_META['categorical_features']
