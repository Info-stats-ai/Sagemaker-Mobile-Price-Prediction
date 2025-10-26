from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sklearn
import joblib
import argparse
import os
import pandas as pd

def model_fn(model_dir):
    """Load model for inference"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == "__main__":
    print("....[INFO] Starting training.....")

    # Create parser and add ALL arguments BEFORE parsing
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)

    # SageMaker paths
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", ""))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", ""))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", ""))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR", ""))

    # File names (match your uploaded files!)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")

    # NOW parse arguments
    args, _ = parser.parse_known_args()

    print(f"Received arguments {args}")
    print("SKLEARN VERSION: ", sklearn.__version__)
    print("JOBLIB VERSION: ", joblib.__version__)

    # Read data
    train_data = pd.read_csv(os.path.join(args.train, args.train_file))
    test_data = pd.read_csv(os.path.join(args.test, args.test_file))

    # Prepare features
    feature_names = train_data.columns.drop("price_range")
    X_train = train_data[feature_names]
    y_train = train_data["price_range"]
    X_test = test_data[feature_names]
    y_test = test_data["price_range"]

    print("....Training model.....")
    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        verbose=2,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Save model (only once!)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"....Model saved at {model_path}")

    # After saving , we can do prediction
    print("....Starting predictions.....")
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"....Test accuracy: {test_accuracy}")
    test_report = classification_report(y_test, y_pred)
    print(f"....Test report: {test_report}")

