# src/data_processing.py
import os
import email
import email.policy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

def load_emails(directory, label):
    """Load emails from directory and assign labels"""
    emails = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    content = f.read()
                    emails.append(content)
                    labels.append(label)
                except:
                    continue
    
    return emails, labels

def advanced_preprocess_text(text):
    """Advanced text preprocessing with NLP techniques"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove email headers
    text = re.sub(r'^.*?:.*?$', '', text, flags=re.MULTILINE)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers, keep letters and basic punctuation
    text = re.sub(r'[^a-záéíóúâêîôûãõç\s\.!?]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back to text
    processed_text = ' '.join(tokens)
    
    # Remove extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

def prepare_dataset(spam_dir, ham_dir):
    """Prepare complete dataset with advanced preprocessing"""
    spam_emails, spam_labels = load_emails(spam_dir, 1)  # 1 = spam
    ham_emails, ham_labels = load_emails(ham_dir, 0)     # 0 = ham
    
    all_emails = spam_emails + ham_emails
    all_labels = spam_labels + ham_labels
    
    # Advanced preprocessing with NLP techniques
    print("Applying NLP preprocessing...")
    processed_emails = [advanced_preprocess_text(email) for email in all_emails]
    
    return processed_emails, all_labels

def create_tfidf_features(emails, max_features=5000):
    """Create TF-IDF features with enhanced parameters"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
        min_df=2,           # Ignore terms that appear in less than 2 documents
        max_df=0.8,         # Ignore terms that appear in more than 80% of documents
        sublinear_tf=True   # Apply sublinear TF scaling
    )
    
    X = vectorizer.fit_transform(emails)
    return X, vectorizer
