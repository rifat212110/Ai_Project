import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import nltk
import string
import re
from textblob import TextBlob
import warnings
import threading
import time
warnings.filterwarnings('ignore')

PLACEHOLDER_TEXT = "Type your message here to check for cyberbullying..."

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# ---------------------- Enhanced Preprocessing Function ----------------------

def preprocess(text):
    """Enhanced preprocessing with better text cleaning"""
    text = str(text).lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Handle contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    
    # Remove excessive punctuation but keep some for context
    text = re.sub(r'[!]{2,}', ' EXCLAMATION ', text)
    text = re.sub(r'[?]{2,}', ' QUESTION ', text)
    text = re.sub(r'[.]{3,}', ' DOTS ', text)
    
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords but keep negation words
    negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
    words = text.split()
    words = [word for word in words if word not in stop_words or word in negation_words]
    
    return ' '.join(words)

def extract_features(text):
    """Extract additional features from text"""
    features = {}
    
    # Basic text statistics
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # Sentiment features using VADER
    try:
        sentiment_scores = sia.polarity_scores(text)
        features['sentiment_pos'] = sentiment_scores['pos']
        features['sentiment_neg'] = sentiment_scores['neg']
        features['sentiment_neu'] = sentiment_scores['neu']
        features['sentiment_compound'] = sentiment_scores['compound']
    except:
        features['sentiment_pos'] = 0
        features['sentiment_neg'] = 0
        features['sentiment_neu'] = 0
        features['sentiment_compound'] = 0
    
    # Offensive word patterns (basic indicators)
    offensive_patterns = ['hate', 'stupid', 'idiot', 'kill', 'die', 'ugly', 'fat', 'loser']
    features['offensive_word_count'] = sum(1 for word in offensive_patterns if word in text.lower())
    
    return features

# ---------------------- Load and Prepare Dataset ----------------------

df = pd.read_csv("cyberbullying_tweets.csv")
df.columns = ['text', 'label']

# Enhanced preprocessing
df['cleaned_text'] = df['text'].apply(preprocess)
df = df[df['cleaned_text'].str.strip() != '']

# Extract additional features
feature_data = []
for text in df['text']:
    feature_data.append(extract_features(text))

feature_df = pd.DataFrame(feature_data)
df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

# Convert to binary labels
df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)

# Enhanced balancing with stratification
from sklearn.utils import resample

# Separate classes
cyber_df = df[df['binary_label'] == 1]
not_cyber_df = df[df['binary_label'] == 0]

# Balance dataset with better sampling
min_len = min(len(cyber_df), len(not_cyber_df))
max_samples = min(min_len * 2, 10000)  # Limit to prevent memory issues

if len(cyber_df) > max_samples:
    cyber_df = resample(cyber_df, n_samples=max_samples, random_state=42)
if len(not_cyber_df) > max_samples:
    not_cyber_df = resample(not_cyber_df, n_samples=max_samples, random_state=42)

df_balanced = pd.concat([cyber_df, not_cyber_df])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset size after balancing: {len(df_balanced)}")
print(f"Cyberbullying samples: {sum(df_balanced['binary_label'])}")
print(f"Non-cyberbullying samples: {len(df_balanced) - sum(df_balanced['binary_label'])}")
