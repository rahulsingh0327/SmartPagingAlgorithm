import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info("CSV file loaded.")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise

    df = df.iloc[10:]
    df.drop(columns=['Timestamp', 'OS', 'Machine_ID', 'Hostname'], inplace=True)

    # Map the action labels
    def map_page_fault_action(pf_delta):
        if pf_delta <= 3000:
            return "0"
        elif 3000 < pf_delta <= 5000:
            return "1"
        elif 5000 < pf_delta <= 100000:
            return "2"
        else:
            return "3"

    df['Suggested_Action'] = df['Page_Faults_Delta'].apply(map_page_fault_action)
    df.drop(columns=['Page_Faults_Delta'], inplace=True)

    return df

def train_model(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)
    logging.info("Model training complete.")
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    logging.info("Evaluation complete.")
    logging.info("\n" + report)
    logging.info(f"Confusion Matrix:\n{matrix}")
    return y_pred

def save_model(clf, output_path: str = "memory_action_classifier.pkl"):
    joblib.dump(clf, output_path)
    logging.info(f"Model saved to {output_path}")

def save_predictions(X_test, y_test, y_pred, output_path: str = "test_data_with_predictions.csv"):
    test_data = X_test.copy()
    test_data['True_Label'] = y_test.values
    test_data['Predicted_Label'] = y_pred
    test_data.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

def main():
    data_file = "data/windows.csv"
    model_file = "app_utils/memory_action_classifier.pkl"
    predictions_file = "app_utils/test_data_with_predictions.csv"

    df = load_and_preprocess_data(data_file)

    X = df[['RAM_Usage_MB', 'Swap_Usage_MB', 'CPU_Usage']]
    y = df['Suggested_Action']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = train_model(X_train, y_train)
    y_pred = evaluate_model(clf, X_test, y_test)

    save_model(clf, model_file)
    save_predictions(X_test, y_test, y_pred, predictions_file)

if __name__ == "__main__":
    main()
