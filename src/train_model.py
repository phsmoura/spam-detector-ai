# src/train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
import numpy as np
from data_processing import prepare_dataset, create_tfidf_features

def train_spam_detector(spam_dir, ham_dir, model_type='naive_bayes', optimize=False):
    """Train spam detection model with enhanced NLP features"""
    
    # Prepare data
    print("Preparing dataset with NLP preprocessing...")
    emails, labels = prepare_dataset(spam_dir, ham_dir)
    
    # Create features
    print("Creating enhanced TF-IDF features...")
    X, vectorizer = create_tfidf_features(emails)
    y = np.array(labels)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Choose and configure model
    if model_type == 'naive_bayes':
        model = MultinomialNB(alpha=0.1)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42,
            max_depth=20,
            class_weight='balanced'
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            C=1.0
        )
    elif model_type == 'svm':
        model = SVC(
            kernel='linear', 
            probability=True, 
            random_state=42,
            class_weight='balanced',
            C=0.1
        )
    else:
        raise ValueError("Model type not supported")
    
    # Hyperparameter optimization (optional)
    if optimize and model_type == 'logistic_regression':
        param_grid = {'C': [0.1, 1, 10]}
        model = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    
    # Train model
    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # If optimized, get best estimator
    if optimize and hasattr(model, 'best_estimator_'):
        model = model.best_estimator_
        print(f"Best parameters: {model.get_params()}")
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Model Performance - {model_type.upper()}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Test set size: {len(y_test)} emails")
    print(f"Spam samples in test: {sum(y_test)}")
    print(f"Ham samples in test: {len(y_test) - sum(y_test)}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save model and vectorizer
    os.makedirs('data/model', exist_ok=True)
    
    with open('data/model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('data/model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save performance metrics
    performance = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    with open('data/model/performance.pkl', 'wb') as f:
        pickle.dump(performance, f)
    
    print("\nModel saved in data/model/")
    print("Files created: model.pkl, vectorizer.pkl, performance.pkl")
    
    return model, vectorizer, accuracy

def compare_models(spam_dir, ham_dir):
    """Compare different model performances"""
    models = ['naive_bayes', 'logistic_regression', 'random_forest', 'svm']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*30}")
        print(f"Training {model_type}...")
        print(f"{'='*30}")
        
        model, vectorizer, accuracy = train_spam_detector(
            spam_dir, ham_dir, model_type=model_type
        )
        results[model_type] = accuracy
    
    # Display comparison
    print(f"\n{'='*50}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*50}")
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:20}: {acc:.4f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    spam_dir = "data/raw/en/spam"
    ham_dir = "data/raw/en/ham"
    
    # Train single model
    train_spam_detector(spam_dir, ham_dir, model_type='naive_bayes')
    
    # Or compare all models
    # compare_models(spam_dir, ham_dir)