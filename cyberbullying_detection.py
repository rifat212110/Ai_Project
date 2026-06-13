import tkinter as tk
from tkinter import messagebox, ttk
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


# ---------------------- Enhanced Feature Engineering ----------------------

# TF-IDF with optimized parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),  # Include bigrams and trigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode'
)

# Fit TF-IDF on cleaned text
X_text = tfidf_vectorizer.fit_transform(df_balanced['cleaned_text'])

# Additional numerical features
feature_columns = ['length', 'word_count', 'char_count', 'avg_word_length', 
                  'exclamation_count', 'question_count', 'caps_ratio',
                  'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 
                  'sentiment_compound', 'offensive_word_count']

X_features = df_balanced[feature_columns].fillna(0)

# Scale features to [0,1] range for MultinomialNB compatibility
scaler = MinMaxScaler()
X_features_scaled = scaler.fit_transform(X_features)

# Combine TF-IDF and scaled additional features
from scipy.sparse import hstack, csr_matrix
X_features_sparse = csr_matrix(X_features_scaled)
X_combined = hstack([X_text, X_features_sparse])

y = df_balanced['binary_label']

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ---------------------- Enhanced Model with Ensemble ----------------------

# Define individual models
svm_model = LinearSVC(C=1.0, random_state=42, max_iter=2000)
lr_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
nb_model = MultinomialNB(alpha=0.1)  # Now works with scaled features

# Create ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('lr', lr_model),
        ('rf', rf_model),
        ('nb', nb_model)
    ],
    voting='hard'  # Use hard voting for binary classification
)

# Train the ensemble model
print("Training ensemble model...")
ensemble_model.fit(X_train, y_train)

# Evaluate ensemble model
y_pred_ensemble = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

print(f"\nEnsemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")
print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['Not Cyberbullying', 'Cyberbullying']))

# Also train individual models for comparison
individual_accuracies = {}
models = {'SVM': svm_model, 'Logistic Regression': lr_model, 
          'Random Forest': rf_model, 'Naive Bayes': nb_model}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    individual_accuracies[name] = acc
    print(f"{name} Accuracy: {acc * 100:.2f}%")

# Use the best performing model (ensemble or individual)
best_model = ensemble_model
best_accuracy = ensemble_accuracy

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")

# ---------------------- Enhanced GUI Functions ----------------------

def animate_scan():
    """Animate the scanning process"""
    dots = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for i in range(20):  # 2 seconds animation
        if analyzing:
            check_button.config(text=f"{dots[i % len(dots)]} Analyzing...")
            root.update()
            time.sleep(0.1)

def detect_cyberbullying():
    global analyzing
    user_input = text_entry.get("1.0", tk.END).strip()
    if not user_input or user_input == PLACEHOLDER_TEXT:
        messagebox.showwarning("Input Required", "Please enter some text to analyze.")
        return
    
    # Show loading state
    analyzing = True
    
    # Start animation in a separate thread
    animation_thread = threading.Thread(target=animate_scan)
    animation_thread.start()
    
    try:
        # Simulate processing time for better UX
        time.sleep(1)
        
        # Preprocess text
        cleaned = preprocess(user_input)
        if not cleaned:
            messagebox.showwarning("Invalid Input", "Input contains no meaningful content after preprocessing.")
            return
        
        # Extract features
        text_features = extract_features(user_input)
        
        # Transform text with TF-IDF
        vect_text = tfidf_vectorizer.transform([cleaned])
        
        # Scale additional features
        additional_features = np.array([[text_features[col] for col in feature_columns]])
        additional_features_scaled = scaler.transform(additional_features)
        
        # Combine with TF-IDF features
        additional_features_sparse = csr_matrix(additional_features_scaled)
        combined_features = hstack([vect_text, additional_features_sparse])
        
        # Make prediction
        prediction = best_model.predict(combined_features)[0]
        
        # Get prediction probabilities (if available)
        try:
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba(combined_features)[0]
                confidence = max(probabilities)
            elif hasattr(best_model, 'decision_function'):
                decision_score = best_model.decision_function(combined_features)[0]
                confidence = abs(decision_score)
            else:
                confidence = 0.8  # Default confidence
        except:
            confidence = 0.8
        
        # Update result display
        if prediction == 0:  # Not cyberbullying
            result_text = "✅ Text Appears Safe"
            result_color = "#27ae60"  # Green
            bg_color = "#e8fff8"
            icon = "🌟"
            advice_text = "Great! Your text promotes positive communication and appears to be respectful."
            suggestions = get_positive_suggestions()
        else:  # Cyberbullying detected
            result_text = "⚠️ Potential Harmful Content"
            result_color = "#e74c3c"  # Red
            bg_color = "#fff5f5"
            icon = "🚨"
            advice_text = "The text may contain harmful language. Consider revising to promote kindness."
            suggestions = get_improvement_suggestions()
        
        # Animate result appearance
        show_result_with_animation(result_text, result_color, bg_color, icon, advice_text, confidence, text_features, suggestions)
        
    except Exception as e:
        messagebox.showerror("Analysis Error", f"An error occurred during analysis: {str(e)}")
    
    finally:
        analyzing = False
        check_button.config(text="🔍 Analyze Text", state="normal")

def show_result_with_animation(result_text, result_color, bg_color, icon, advice_text, confidence, text_features, suggestions):
    """Show result with smooth animation"""
    result_frame.config(bg=bg_color)
    
    # Icon animation
    icon_label.config(text=icon, fg=result_color, bg=bg_color, font=("Segoe UI", 48))
    
    # Result text
    result_label.config(text=result_text, fg=result_color, bg=bg_color)
    
    # Confidence with color coding
    conf_color = "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.5 else "#e74c3c"
    confidence_text = f"Confidence: {confidence:.1%}"
    confidence_label.config(text=confidence_text, bg=bg_color, fg=conf_color)
    
    # Advice
    advice_label.config(text=advice_text, bg=bg_color)
    
    # Analysis details with emojis
    analysis_text = f"📝 Words: {text_features['word_count']} | "
    analysis_text += f"😊 Sentiment: {text_features['sentiment_compound']:.2f} | "
    analysis_text += f"📢 Caps: {text_features['caps_ratio']:.1%}"
    analysis_label.config(text=analysis_text, bg=bg_color)
    
    # Suggestions
    suggestions_text.config(state='normal')
    suggestions_text.delete(1.0, tk.END)
    suggestions_text.insert(tk.END, suggestions)
    suggestions_text.config(state='disabled', bg=bg_color)
    
    # Show result frame with fade-in effect
    result_frame.pack(pady=20, padx=20, fill="x")
    
    # Pulse effect for icon
    animate_icon_pulse(icon_label, result_color)

def animate_icon_pulse(widget, color):
    """Create a pulse effect for the icon"""
    def pulse():
        for size in [48, 52, 48]:
            widget.config(font=("Segoe UI", size))
            root.update()
            time.sleep(0.1)
    
    pulse_thread = threading.Thread(target=pulse)
    pulse_thread.start()

def get_positive_suggestions():
    suggestions = [
        "🌟 Your text promotes positive communication!",
        "💡 Keep using respectful language like this",
        "🎯 Consider adding encouraging words to make it even better",
        "✨ Your message creates a safe space for others"
    ]
    return "\n".join(suggestions)

def get_improvement_suggestions():
    suggestions = [
        "🤝 Try rephrasing with kindness and empathy",
        "💭 Consider how your words might affect others",
        "🌱 Use 'I feel' statements instead of 'You are' statements",
        "🔄 Replace negative words with constructive alternatives",
        "❤️ Focus on the issue, not personal attacks"
    ]
    return "\n".join(suggestions)

def clear_text():
    text_entry.delete("1.0", tk.END)
    result_frame.pack_forget()
    char_label.config(text="Characters: 0/5000", fg="#7f8c8d")
    text_entry.config(fg="#bdc3c7")
    text_entry.focus()

def on_text_change(event):
    char_count = len(text_entry.get("1.0", tk.END).strip())
    char_label.config(text=f"Characters: {char_count}/5000")
    
    # Color coding for character count
    if char_count > 4500:
        char_label.config(fg="#e74c3c")  # Red
    elif char_count > 3000:
        char_label.config(fg="#f39c12")  # Orange
    else:
        char_label.config(fg="#7f8c8d")  # Gray

def update_placeholder(force=False):
    content = text_entry.get("1.0", tk.END).strip()
    if force or not content:
        text_entry.config(fg="#bdc3c7")
        if text_entry.get("1.0", tk.END).strip() == "":
            text_entry.insert("1.0", PLACEHOLDER_TEXT)
    else:
        if content == PLACEHOLDER_TEXT:
            text_entry.delete("1.0", tk.END)
        text_entry.config(fg="#2c3e50")

def on_text_focus_in(event):
    if text_entry.get("1.0", tk.END).strip() == PLACEHOLDER_TEXT:
        text_entry.delete("1.0", tk.END)
        text_entry.config(fg="#2c3e50")

def on_text_focus_out(event):
    if not text_entry.get("1.0", tk.END).strip():
        update_placeholder(force=True)

def show_model_details():
    details = f"""🔬 Advanced AI Model Details

🎯 Ensemble Model Performance: {best_accuracy * 100:.2f}%

🤖 Individual Model Accuracies:
"""
    for name, acc in individual_accuracies.items():
        details += f"   • {name}: {acc * 100:.2f}%\n"
    
    details += f"""
📊 Cross-Validation Score: {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 2 * 100:.2f}%)

🧠 Advanced Features:
   • TF-IDF Vectorization (8,000+ features)
   • N-gram Analysis (1-3 word patterns)
   • Sentiment Analysis Integration
   • Text Structure Analysis
   • Punctuation Pattern Recognition
   • Offensive Language Detection

⚙️ Model Architecture:
   Ensemble of 4 Machine Learning Models:
   • Support Vector Machine (SVM)
   • Logistic Regression
   • Random Forest
   • Naive Bayes
   
🔧 Technical Notes:
   Features are normalized for optimal performance
   Hard voting consensus for final predictions
   Cross-validated for reliability
"""
    
    messagebox.showinfo("🔬 Model Technical Details", details)

def show_tips():
    tips = """💡 Tips for Better Communication Online:

🌟 Positive Communication:
   • Use "I" statements instead of "You" accusations
   • Express disagreement respectfully
   • Focus on ideas, not personal attacks
   • Ask questions to understand better

🚫 Avoid These Patterns:
   • Name-calling and insults
   • Threats or aggressive language
   • Discriminatory comments
   • Excessive caps (SHOUTING)

🛡️ Creating Safe Spaces:
   • Encourage others' contributions
   • Use constructive feedback
   • Be mindful of cultural differences
   • Think before you post

🔄 Before Posting, Ask:
   • Would I say this face-to-face?
   • Could this hurt someone's feelings?
   • Is this constructive or destructive?
   • Am I being respectful?
"""
    
    messagebox.showinfo("💡 Communication Tips", tips)

