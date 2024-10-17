import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import sys

# Redirect stdout to a file
original_stdout = sys.stdout
with open('output_thic.txt', 'w') as f:
    sys.stdout = f

    # Load the data
    data = pd.read_csv('df_thic_adj.csv')  # Replace with your actual file name

    # Separate features and target
    X = data.drop(['subject', 'site', 'pid', 'outcome'], axis=1)
    y = data['outcome']

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Initialize LabelEncoder for the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Print the encoding mapping
    print("Class encoding:")
    for cls, encoded_value in zip(le.classes_, le.transform(le.classes_)):
        print(f"{cls}: {encoded_value}")

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])

    # Create a pipeline
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', class_weight=class_weight_dict, random_state=42))
    ])

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize lists to store performance metrics
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    # Perform stratified k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded), 1):
        print(f"Fold {fold}")
        
        # Split the data
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]
        
        # Fit the pipeline and make predictions
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Store the metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print()

    # Print average performance across all folds
    print("Average Performance:")
    print(f"Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
    print(f"Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
    print(f"F1-score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

    # Print classification report for the last fold
    print("\nDetailed Classification Report (Last Fold):")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

# Restore the original stdout
sys.stdout = original_stdout

print("Classification completed. Results have been written to 'output.txt'.")