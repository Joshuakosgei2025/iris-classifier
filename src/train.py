#%% Imports
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

#%% Functions
def parse_args():
    parser = argparse.ArgumentParser(description="Train Iris Decision Tree")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of test set")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_data():
    """Load Iris dataset."""
    data = load_iris()
    X = data.data
    y = data.target
    class_names = data.target_names
    return X, y, class_names

def split_data(X, y, test_size, random_state):
    """Split data into training and testing sets."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def train_model(X_train, y_train, random_state):
    """Train a Decision Tree classifier."""
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model using multiple metrics."""
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1_score": f1_score(y_test, predictions, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }

    return metrics

def save_confusion_matrix(cm, class_names, filename="outputs/confusion_matrix.png"):
    """Save a labeled confusion matrix as a PNG file."""
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

#%% Main
def main():
    args = parse_args()

    # Load data
    X, y, class_names = load_data()
    print("Data loaded. Shape:", X.shape)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print("Data split completed.")

    # Train model
    model = train_model(X_train, y_train, random_state=args.random_state)
    print("Model training completed.")
    
    # Save trained model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.joblib")
    print("Trained model saved as 'outputs/model.joblib'.")

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy:  {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1-score:  {metrics['f1_score']:.2f}")

    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Save confusion matrix
    save_confusion_matrix(metrics["confusion_matrix"], class_names)
    print("Confusion matrix saved as 'outputs/confusion_matrix.png'.")

#%% Entry point
if __name__ == "__main__":
    main()
