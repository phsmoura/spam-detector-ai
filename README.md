# Spam Detector AI

This project implements a spam classifier that combines natural language processing (NLP) with machine learning algorithms. It can process raw emails, extract relevant features, and make predictions to classify emails as spam or non-spam in the language it was trained on.

## Dataset

The dataset used in this project for spam emails was downloaded from [Enron-Spam](https://www2.aueb.gr/users/ion/data/enron-spam/) and [SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/), public collections used for spam detection research.

For non-spam samples, emails were downloaded directly from the Fedora mail server, as the objective of this project is to add a spam blocking layer for email lists.

To test predictions, emails not included in the training base were used. 

For non-spam emails, new emails downloaded from the Fedora mail server were used, achieving 100% accuracy with no false positives.

For spam emails, the [phishing_pot](https://github.com/rf-peixoto/phishing_pot/tree/main/email) dataset was used, which contains email samples in various languages and also base64-encoded emails. In this case, encoded emails and emails in languages other than English were classified as legitimate, since the dataset and code were designed to detect spam exclusively in the English language.

## Directory Structure

The project is organized in a structure that separates data, source code, and resources:

```
spam-detector-ai/
├── data/
│   ├── model/                          # Stores trained models and vectorizers
│   │   ├── performance.pkl
│   │   ├── model.pkl
│   │   └── vectorizer.pkl
│   ├── predict/                        # Contains new emails for classification
│   ├── raw/                            # Organized data for training (spam/ham)
│   │   └── en                          # Separated by language
│   │       ├── ham
│   │       └── spam
│   └── raw_downloaded/                 # Originally downloaded raw data
├── src/
│   ├── data_processing.py              # NLP preprocessing module
│   ├── organize_raw_downloaded.py      # Script to organize raw data
│   ├── train_model.py                  # Model training script
│   └── predict.py                      # Prediction script
├── requirements.txt                    # Project dependencies list
├── LICENSE                             # Usage license
└── README.md                           # Main documentation
```

## Main Features

### Natural Language Processing

The system performs text preprocessing including tokenization, stopword removal, stemming, email header cleaning, URL removal, and text normalization. Features are extracted using TF-IDF with n-grams to capture complex patterns.

### Machine Learning Models

Four different algorithms were implemented: Naive Bayes (fast and efficient for text), Logistic Regression (good interpretability), Random Forest (robust against overfitting), and SVM (high performance in text classification). Each model can be trained and compared.

### Prediction System

The `SpamDetector` class allows predictions, with adjustable threshold to balance between false positives and false negatives. The code returns detailed probabilities and confidence levels.

## Getting Started

First, install the dependencies: 

`pip install -r requirements.txt`

Then, download the NLTK data by running: 

`python -m nltk.downloader -d .venv/nltk_data all`

To organize the raw data, execute: 

`python src/organize_raw_downloaded.py`

Model training can be done with: 

`python src/train_model.py`

To make predictions: 

`python src/predict.py`

### Workflow:

1. **Data Preparation**: Raw emails are automatically organized into spam and ham directories
2. **Preprocessing**: Text is cleaned and transformed using NLP techniques
3. **Feature Extraction**: TF-IDF converts text into numerical representation
4. **Training**: Models are trained and evaluated with cross-validation
5. **Prediction**: New emails are classified with confidence probabilities

## Configuration and Customization

There is flexibility to adjust parameters such as detection threshold, model type, and TF-IDF settings. The modular structure facilitates the incorporation of new algorithms or processing techniques.

## Performance

The typical models mentioned above achieve accuracy between 70% and 100%. However, if the dataset is unbalanced, stratified validation and hyperparameter optimization help prevent overfitting. In the case of Naive Bayes, accuracy can reach ~90% or higher for email classification, according to academic literature, and large datasets are not required to train this algorithm.

Therefore, as default, this project uses Naive Bayes with stratified validation, since the dataset was unbalanced, training is faster and recommended for this use case, and the trained model, with the dataset explained above, achieved 99.5% accuracy.

Articles about Naive Bayes and spam detection:
- [Spam Email Detection using Naive Bayes classifier](https://www.researchgate.net/publication/388324015_Spam_Email_Detection_using_Naive_Bayes_classifier)
- [A Modified Naive Bayes Classifier for Detecting Spam E-mails based on Feature Selection](https://ieeexplore.ieee.org/document/9788340)

## Technical Requirements

The project requires Python 3.13+ and the libraries listed in [requirements.txt](). The directory structure must be maintained as described to ensure proper script functionality.

## License

Distributed under the MIT license. See the [LICENSE]() file for more details.
