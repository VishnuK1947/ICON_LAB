import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import uniform, randint
import sys

def custom_scorer(y_true, y_pred):
    pte_recall = recall_score(y_true, y_pred, pos_label=pte_class)
    tbi_recall = recall_score(y_true, y_pred, pos_label=tbi_class)
    return (pte_recall + tbi_recall) / 2  # Average recall for both classes

# Redirect stdout to a file
original_stdout = sys.stdout
with open('pte_tbi_classification_output.txt', 'w') as f:
    sys.stdout = f

    # Load the data
    data = pd.read_csv('df_vol_adj.csv')  # Replace with your actual file name

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

    # Identify PTE and TBI classes
    pte_class = le.transform(['PTE'])[0]
    tbi_class = le.transform(['TBI'])[0]

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', StandardScaler(), categorical_features)  # Treat categorical as numeric for simplicity
        ])

    # Create a pipeline with SMOTE, feature selection, and classifier
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('feature_selection', SelectKBest(f_classif, k='all')),
        ('classifier', SVC(class_weight=class_weight_dict, random_state=42, probability=True))
    ])

    # Parameter distribution for randomized search
    param_dist = {
        'feature_selection__k': randint(1, len(X.columns)),
        'classifier__C': uniform(0.1, 100),
        'classifier__gamma': uniform(0.001, 1),
        'classifier__kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Custom scorer that balances sensitivity for both classes
    balanced_recall_scorer = make_scorer(custom_scorer)

    # Initialize RandomizedSearchCV with custom scorer
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=100, cv=5, 
        scoring=balanced_recall_scorer, n_jobs=-1, random_state=42, verbose=1
    )

    # Fit RandomizedSearchCV
    random_search.fit(X, y_encoded)

    # Print best parameters and score
    print("\nBest parameters found:")
    print(random_search.best_params_)
    print(f"\nBest cross-validation score (Balanced Recall): {random_search.best_score_:.4f}")

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize lists to store performance metrics
    accuracies, precisions, recalls, f1_scores, pte_sensitivities, tbi_sensitivities, auc_scores = [], [], [], [], [], [], []

    # Perform stratified k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\nFold {fold}")
        
        # Split the data
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]
        
        # Fit the model and make predictions
        best_model = random_search.best_estimator_
        best_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        pte_sensitivity = recall_score(y_val, y_pred, pos_label=pte_class)
        tbi_sensitivity = recall_score(y_val, y_pred, pos_label=tbi_class)
        auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        
        # Store the metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        pte_sensitivities.append(pte_sensitivity)
        tbi_sensitivities.append(tbi_sensitivity)
        auc_scores.append(auc)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"PTE Sensitivity: {pte_sensitivity:.4f}")
        print(f"TBI Sensitivity: {tbi_sensitivity:.4f}")
        print(f"AUC: {auc:.4f}")

    # Print average performance across all folds
    print("\nAverage Performance:")
    print(f"Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
    print(f"Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
    print(f"F1-score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    print(f"PTE Sensitivity: {np.mean(pte_sensitivities):.4f} (+/- {np.std(pte_sensitivities):.4f})")
    print(f"TBI Sensitivity: {np.mean(tbi_sensitivities):.4f} (+/- {np.std(tbi_sensitivities):.4f})")
    print(f"AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

    # Print classification report for the last fold
    print("\nDetailed Classification Report (Last Fold):")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

# Restore the original stdout
sys.stdout = original_stdout

print("PTE/TBI classification completed. Results have been written to 'pte_tbi_classification_output.txt'.")