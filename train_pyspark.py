import sys
import os


def build_and_train(input_csv, model_out):
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            StringIndexer,
            OneHotEncoder,
            VectorAssembler,
            Imputer,
        )
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PySpark is not installed. Install optional dependencies with: "
            "pip install -r requirements-pyspark.txt"
        ) from exc

    spark = SparkSession.builder.appName("credit_card_approval").getOrCreate()

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(input_csv)
    )

    # Drop identifiers / date
    df = df.drop("application_id", "application_date")

    # Target
    label_indexer = StringIndexer(inputCol="approval_status", outputCol="label")

    # Columns to treat as categorical
    cat_cols = [
        "employment_type",
        "marital_status",
        "education_level",
        "residence_type",
        "city_tier",
        "card_type_requested",
    ]

    # Numeric columns
    numeric_cols = [c for c, t in df.dtypes if t in ("int", "double", "bigint") and c != "label"]
    # but ensure we exclude approval_status
    numeric_cols = [c for c in numeric_cols if c != "approval_status"]

    # Impute numeric missing values
    if numeric_cols:
        imputer = Imputer(inputCols=numeric_cols, outputCols=[c + "_imputed" for c in numeric_cols])
        assembled_numeric = [c + "_imputed" for c in numeric_cols]
    else:
        imputer = None
        assembled_numeric = []

    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_enc") for c in cat_cols]

    feature_inputs = assembled_numeric + [c + "_enc" for c in cat_cols]

    assembler = VectorAssembler(inputCols=feature_inputs, outputCol="features", handleInvalid="keep")

    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)

    stages = []
    stages.append(label_indexer)
    if imputer:
        stages.append(imputer)
    stages += indexers
    stages += encoders
    stages.append(assembler)
    stages.append(rf)

    pipeline = Pipeline(stages=stages)

    # Train/test split
    train, test = df.randomSplit([0.7, 0.3], seed=42)

    model = pipeline.fit(train)

    preds = model.transform(test)

    acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    auc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC")

    accuracy = acc_evaluator.evaluate(preds)
    try:
        auc = auc_evaluator.evaluate(preds)
    except Exception:
        auc = None

    print(f"Test accuracy: {accuracy:.4f}")
    if auc is not None:
        print(f"Test AUC: {auc:.4f}")

    # Save model
    os.makedirs(model_out, exist_ok=True)
    model.write().overwrite().save(os.path.join(model_out, "pipeline"))

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        streamlit_context = (
            "streamlit" in " ".join(sys.argv).lower()
            or os.environ.get("STREAMLIT_SERVER_PORT") is not None
        )
        if streamlit_context:
            # If Streamlit is pointed at this file by mistake, forward to the app entrypoint.
            from app import main as run_streamlit_app

            run_streamlit_app()
            sys.exit(0)

        print("Usage: python train_pyspark.py <input_csv> [model_out_dir]")
        sys.exit(1)

    input_csv = sys.argv[1]
    model_out = sys.argv[2] if len(sys.argv) > 2 else "models/pyspark_rf_model"

    build_and_train(input_csv, model_out)
