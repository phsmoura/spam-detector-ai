# src/predict.py
import pickle
import os
import numpy as np
from data_processing import advanced_preprocess_text

class SpamDetector:
    def __init__(self, model_path, vectorizer_path):
        """Initialize spam detector with trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("Spam detector model loaded successfully!")
            
        except FileNotFoundError:
            print("Model files not found. Please train the model first.")
            raise
    
    def predict_email(self, email_text, threshold=0.5):
        """Predict if an email is spam with confidence scores"""
        # Preprocess with NLP techniques
        processed_text = advanced_preprocess_text(email_text)
        
        # Transform to features
        features = self.vectorizer.transform([processed_text])
        
        # Prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Apply threshold
        spam_prob = probability[1]
        is_spam = spam_prob > threshold
        
        return {
            'is_spam': bool(is_spam),
            'spam_probability': spam_prob,
            'ham_probability': probability[0],
            'confidence': max(probability),
            'threshold': threshold,
            'processed_text': processed_text[:200] + '...' if len(processed_text) > 200 else processed_text
        }
    
    def batch_predict(self, email_texts, threshold=0.5):
        """Predict multiple emails at once"""
        results = []
        for text in email_texts:
            results.append(self.predict_email(text, threshold))
        return results

def predict_single_email(email_path, model_dir='data/model', threshold=0.5):
    """Predict if a single email is spam"""
    try:
        detector = SpamDetector(
            os.path.join(model_dir, 'spam_model.pkl'),
            os.path.join(model_dir, 'vectorizer.pkl')
        )
        
        with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
            email_content = f.read()
        
        result = detector.predict_email(email_content, threshold)
        
        print(f"\n{'='*50}")
        print("SPAM DETECTION RESULTS")
        print(f"{'='*50}")
        print(f"File: {email_path}")
        print(f"Spam: {'YES' if result['is_spam'] else 'NO'}")
        print(f"Spam probability: {result['spam_probability']:.4f}")
        print(f"Ham probability: {result['ham_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Threshold: {result['threshold']}")
        print(f"\nProcessed text preview:")
        print(result['processed_text'])
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def predict_directory(directory_path, model_dir='data/model', threshold=0.5):
    """Predict all emails in a directory"""
    try:
        detector = SpamDetector(
            os.path.join(model_dir, 'spam_model.pkl'),
            os.path.join(model_dir, 'vectorizer.pkl')
        )
        
        results = []
        spam_count = 0
        total_count = 0
        
        print(f"Processing directory: {directory_path}")
        
        for filename in os.listdir(directory_path):
            # if filename.endswith('.txt') or filename.endswith('.eml'):
            email_path = os.path.join(directory_path, filename)
            total_count += 1
            
            try:
                with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
                    email_content = f.read()
                
                result = detector.predict_email(email_content, threshold)
                result['filename'] = filename
                results.append(result)
                
                if result['is_spam']:
                    spam_count += 1
                    status = "SPAM"
                else:
                    status = "HAM"
                
                print(f"{filename:30}: {status} ({result['spam_probability']:.3f})")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Summary
        print(f"\n{'='*50}")
        print("BATCH PREDICTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total emails processed: {total_count}")
        print(f"Spam emails detected: {spam_count}")
        print(f"Ham emails detected: {total_count - spam_count}")
        print(f"Spam percentage: {(spam_count/total_count*100):.1f}%")
        
        return results
        
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return []

def load_performance_metrics(model_dir='data/models'):
    """Load and display model performance metrics"""
    try:
        with open(os.path.join(model_dir, 'performance.pkl'), 'rb') as f:
            performance = pickle.load(f)
        
        print(f"\n{'='*50}")
        print("MODEL PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Accuracy: {performance['accuracy']:.4f}")
        print("\nClassification Report:")
        for key, value in performance['classification_report'].items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"{key}: {value:.4f}")
        
        return performance
        
    except FileNotFoundError:
        print("Performance metrics not found.")
        return None

if __name__ == "__main__":
    # Example usage
    # email_to_check = "data/predict/mail.txt"
    
    # Predict single email
    # predict_single_email(email_to_check, threshold=0.5)
    
    # Or predict directory
    predict_directory("data/predict/hams/users@lists.fedoraproject.org")
    
    # Or load performance metrics
    # load_performance_metrics()