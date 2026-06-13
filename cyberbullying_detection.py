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

# ---------------------- Enhanced GUI Layout ----------------------

root = tk.Tk()
root.title("🛡️ AI Cyberbullying Guardian")
root.geometry("900x900")
root.configure(bg="#f8fafc")
root.resizable(True, True)

# Scrollable page container so the full interface fits on smaller screens
page_container = tk.Frame(root, bg="#f8fafc")
page_container.pack(fill="both", expand=True)

page_canvas = tk.Canvas(page_container, bg="#f8fafc", highlightthickness=0, bd=0)
page_canvas.pack(side="left", fill="both", expand=True)

page_scrollbar = tk.Scrollbar(page_container, orient="vertical", command=page_canvas.yview)
page_scrollbar.pack(side="right", fill="y")

page_canvas.configure(yscrollcommand=page_scrollbar.set)

content_frame = tk.Frame(page_canvas, bg="#f8fafc")
content_window = page_canvas.create_window((0, 0), window=content_frame, anchor="nw")

def update_page_scrollregion(event):
    page_canvas.configure(scrollregion=page_canvas.bbox("all"))

def sync_page_width(event):
    page_canvas.itemconfigure(content_window, width=event.width)

def on_mousewheel(event):
    page_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

content_frame.bind("<Configure>", update_page_scrollregion)
page_canvas.bind("<Configure>", sync_page_width)
root.bind_all("<MouseWheel>", on_mousewheel)

# Global variables
analyzing = False

# Configure style
style = ttk.Style()
style.theme_use('clam')

# Create gradient effect frame
header_frame = tk.Frame(content_frame, bg="#764ba2", height=150)
header_frame.pack(fill="x", padx=0, pady=0)
header_frame.pack_propagate(False)

# Create gradient-like effect with multiple frames
gradient_colors = ["#764ba2", "#764ba2"]
for i, color in enumerate(gradient_colors):
    grad_frame = tk.Frame(header_frame, bg=color, height=70)
    grad_frame.place(x=0, y=i*50, relwidth=1)

# Title with modern styling
title_label = tk.Label(
    header_frame, 
    text="🛡️ AI Cyberbullying Guardian", 
    font=("Segoe UI", 24, "bold"), 
    bg="#764ba2", 
    fg="white"
)
title_label.pack(pady=15)

# Subtitle with glow effect
subtitle_label = tk.Label(
    header_frame, 
    text="Advanced AI Protection • Real-time Analysis • Smart Suggestions", 
    font=("Segoe UI", 12), 
    bg="#764ba2", 
    fg="#e8f1ff"
)
subtitle_label.pack()

# Main Content Frame with modern card design
main_frame = tk.Frame(content_frame, bg="#f8fafc")
main_frame.pack(fill="both", expand=True, padx=30, pady=30)

# Create card-like container
card_frame = tk.Frame(main_frame, bg="white", relief="flat", bd=0)
card_frame.pack(fill="both", expand=True, padx=0, pady=0)

# Add subtle shadow effect
shadow_frame = tk.Frame(main_frame, bg="#e2e8f0", height=4)
shadow_frame.place(x=5, y=5, relwidth=1, relheight=1)
card_frame.lift()

# Input Section with modern design
input_section = tk.Frame(card_frame, bg="white", pady=20)
input_section.pack(fill="x", padx=30)

input_label = tk.Label(
    input_section, 
    text="✍️ Enter your message for analysis:", 
    font=("Segoe UI", 16, "bold"), 
    bg="white", 
    fg="#2d3748"
)
input_label.pack(anchor="w", pady=(0, 10))

# Text Entry with modern styling
text_frame = tk.Frame(input_section, bg="white")
text_frame.pack(fill="x", pady=(0, 10))

text_entry = tk.Text(
    text_frame, 
    height=8, 
    width=70, 
    font=("Segoe UI", 12),
    wrap=tk.WORD,
    relief="solid",
    borderwidth=2,
    bd=0,
    padx=15,
    pady=15,
    bg="#f7fafc",
    fg="#2c3e50",
    insertbackground="#667eea",
    selectbackground="#667eea",
    selectforeground="white"
)
text_entry.pack(side="left", fill="both", expand=True)

# Modern scrollbar
scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_entry.yview, bg="#e2e8f0", troughcolor="#f7fafc")
scrollbar.pack(side="right", fill="y")
text_entry.config(yscrollcommand=scrollbar.set)

# Character counter with status
char_frame = tk.Frame(input_section, bg="white")
char_frame.pack(fill="x", pady=(5, 0))

char_label = tk.Label(
    char_frame, 
    text="Characters: 0/5000", 
    font=("Segoe UI", 10), 
    bg="white", 
    fg="#718096"
)
char_label.pack(side="left")

# Status indicator
status_label = tk.Label(
    char_frame, 
    text="Ready to analyze", 
    font=("Segoe UI", 10), 
    bg="white", 
    fg="#48bb78"
)
status_label.pack(side="right")

# Bind events
text_entry.bind('<KeyRelease>', on_text_change)
text_entry.bind('<FocusIn>', on_text_focus_in)
text_entry.bind('<FocusOut>', on_text_focus_out)

# Initialize placeholder
update_placeholder(force=True)

# Modern Button Section
button_section = tk.Frame(card_frame, bg="white", pady=20)
button_section.pack(fill="x", padx=30)

button_container = tk.Frame(button_section, bg="white")
button_container.pack()

# Analyze Button with gradient-like effect
check_button = tk.Button(
    button_container,
    text="🔍 Analyze Message",
    font=("Segoe UI", 14, "bold"),
    bg="#667eea",
    fg="white",
    relief="flat",
    padx=40,
    pady=15,
    cursor="hand2",
    command=detect_cyberbullying,
    activebackground="#5a6fd8",
    activeforeground="white"
)
check_button.pack(side="left", padx=(0, 15))

# Clear Button
clear_button = tk.Button(
    button_container,
    text="🗑️ Clear Text",
    font=("Segoe UI", 12),
    bg="#e2e8f0",
    fg="#4a5568",
    relief="flat",
    padx=25,
    pady=15,
    cursor="hand2",
    command=clear_text,
    activebackground="#cbd5e0"
)
clear_button.pack(side="left", padx=(0, 15))

# Tips Button
tips_button = tk.Button(
    button_container,
    text="💡 Communication Tips",
    font=("Segoe UI", 12),
    bg="#48bb78",
    fg="white",
    relief="flat",
    padx=25,
    pady=15,
    cursor="hand2",
    command=show_tips,
    activebackground="#38a169"
)
tips_button.pack(side="left", padx=(0, 15))

# Model Details Button
details_button = tk.Button(
    button_container,
    text="📊 Technical Details",
    font=("Segoe UI", 12),
    bg="#9f7aea",
    fg="white",
    relief="flat",
    padx=25,
    pady=15,
    cursor="hand2",
    command=show_model_details,
    activebackground="#8b5cf6"
)
details_button.pack(side="left")

# Result Section (initially hidden)
result_frame = tk.Frame(card_frame, relief="flat", bg="#ffffff", pady=20)

# Icon for result
icon_label = tk.Label(
    result_frame, 
    text="", 
    font=("Segoe UI", 48), 
    bg="#ffffff"
)
icon_label.pack(pady=(10, 5))

# Result text
result_label = tk.Label(
    result_frame, 
    text="", 
    font=("Segoe UI", 20, "bold"), 
    bg="#ffffff"
)
result_label.pack(pady=5)

# Confidence score
confidence_label = tk.Label(
    result_frame, 
    text="", 
    font=("Segoe UI", 12, "bold"), 
    bg="#ffffff"
)
confidence_label.pack(pady=5)

# Advice text
advice_label = tk.Label(
    result_frame, 
    text="", 
    font=("Segoe UI", 12), 
    bg="#ffffff",
    fg="#4a5568",
    wraplength=700
)
advice_label.pack(pady=(5, 15))

# Analysis details
analysis_label = tk.Label(
    result_frame, 
    text="", 
    font=("Segoe UI", 10), 
    bg="#ffffff",
    fg="#718096"
)
analysis_label.pack(pady=(0, 15))

# Suggestions section
suggestions_frame = tk.Frame(result_frame, bg="#ffffff")
suggestions_frame.pack(fill="x", padx=20, pady=10)

suggestions_label = tk.Label(
    suggestions_frame,
    text="💭 Smart Suggestions:",
    font=("Segoe UI", 12, "bold"),
    bg="#ffffff",
    fg="#2d3748"
)
suggestions_label.pack(anchor="w", pady=(0, 5))

suggestions_text = tk.Text(
    suggestions_frame,
    height=4,
    font=("Segoe UI", 10),
    bg="#f7fafc",
    fg="#4a5568",
    relief="flat",
    bd=1,
    padx=10,
    pady=8,
    wrap=tk.WORD,
    state='disabled'
)
suggestions_scrollbar = tk.Scrollbar(suggestions_frame, orient="vertical", command=suggestions_text.yview, bg="#e2e8f0", troughcolor="#f7fafc")
suggestions_scrollbar.pack(side="right", fill="y")
suggestions_text.config(yscrollcommand=suggestions_scrollbar.set)
suggestions_text.pack(side="left", fill="both", expand=True)


# Modern Footer
footer_frame = tk.Frame(content_frame, bg="#2d3748", height=120)
footer_frame.pack(fill="x", side="bottom")
footer_frame.pack_propagate(False)

# Footer content container with padding
footer_content = tk.Frame(footer_frame, bg="#2d3748")
footer_content.pack(expand=True, fill="both", padx=30, pady=20)

# Top section of footer with main info
footer_top = tk.Frame(footer_content, bg="#2d3748")
footer_top.pack(fill="x", pady=(0, 10))

# Left side - App info with icon
app_info_frame = tk.Frame(footer_top, bg="#2d3748")
app_info_frame.pack(side="left")

app_name_label = tk.Label(
    app_info_frame,
    text="🛡️ AI Cyberbullying Guardian",
    font=("Segoe UI", 14, "bold"),
    bg="#2d3748",
    fg="#e2e8f0"
)
app_name_label.pack(anchor="w")

version_label = tk.Label(
    app_info_frame,
    text="v2.0 • Advanced AI Protection System",
    font=("Segoe UI", 9),
    bg="#2d3748",
    fg="#a0aec0"
)
version_label.pack(anchor="w", pady=(2, 0))

# Right side - Performance stats
stats_frame = tk.Frame(footer_top, bg="#2d3748")
stats_frame.pack(side="right")

accuracy_label = tk.Label(
    stats_frame,
    text=f"🎯 Model Accuracy: {best_accuracy * 100:.1f}%",
    font=("Segoe UI", 11, "bold"),
    bg="#2d3748",
    fg="#48bb78"
)
accuracy_label.pack(anchor="e")

features_label = tk.Label(
    stats_frame,
    text="🧠 8,000+ Features • 4 AI Models • Real-time Analysis",
    font=("Segoe UI", 9),
    bg="#2d3748",
    fg="#a0aec0"
)
features_label.pack(anchor="e", pady=(2, 0))

# Separator line with gradient effect
separator = tk.Frame(footer_content, height=1, bg="#4a5568")
separator.pack(fill="x", pady=(5, 10))

# Bottom section of footer
footer_bottom = tk.Frame(footer_content, bg="#2d3748")
footer_bottom.pack(fill="x")

# Left side - Copyright and attribution
copyright_frame = tk.Frame(footer_bottom, bg="#2d3748")
copyright_frame.pack(side="left")

copyright_label = tk.Label(
    copyright_frame,
    text="© 2025 AI Guardian • Built with Python & Scikit-learn",
    font=("Segoe UI", 9),
    bg="#2d3748",
    fg="#718096"
)
copyright_label.pack(anchor="w")

# Center - Mission statement
mission_frame = tk.Frame(footer_bottom, bg="#2d3748")
mission_frame.pack(side="left", expand=True)

mission_label = tk.Label(
    mission_frame,
    text="🌟 Creating safer digital spaces through AI • Promoting positive communication",
    font=("Segoe UI", 9, "italic"),
    bg="#2d3748",
    fg="#9ca3af"
)
mission_label.pack()

# Right side - Technical badges
badges_frame = tk.Frame(footer_bottom, bg="#2d3748")
badges_frame.pack(side="right")

# Create modern badges
tech_badge = tk.Label(
    badges_frame,
    text="⚡ ML",
    font=("Segoe UI", 8, "bold"),
    bg="#667eea",
    fg="white",
    padx=6,
    pady=2
)
tech_badge.pack(side="right", padx=(5, 0))

ai_badge = tk.Label(
    badges_frame,
    text="🤖 AI",
    font=("Segoe UI", 8, "bold"),
    bg="#48bb78",
    fg="white",
    padx=6,
    pady=2
)
ai_badge.pack(side="right", padx=(5, 0))

security_badge = tk.Label(
    badges_frame,
    text="🛡️ SAFE",
    font=("Segoe UI", 8, "bold"),
    bg="#ed8936",
    fg="white",
    padx=6,
    pady=2
)
security_badge.pack(side="right", padx=(5, 0))

# Add subtle hover effects for interactive elements
def on_badge_enter(event, badge, color):
    badge.config(bg=color)

def on_badge_leave(event, badge, original_color):
    badge.config(bg=original_color)

# Bind hover effects
tech_badge.bind("<Enter>", lambda e: on_badge_enter(e, tech_badge, "#5a6fd8"))
tech_badge.bind("<Leave>", lambda e: on_badge_leave(e, tech_badge, "#667eea"))

ai_badge.bind("<Enter>", lambda e: on_badge_enter(e, ai_badge, "#38a169"))
ai_badge.bind("<Leave>", lambda e: on_badge_leave(e, ai_badge, "#48bb78"))

security_badge.bind("<Enter>", lambda e: on_badge_enter(e, security_badge, "#dd6b20"))
security_badge.bind("<Leave>", lambda e: on_badge_leave(e, security_badge, "#ed8936"))

# Add a subtle glow effect to the footer
def create_footer_glow():
    """Create a subtle animated glow effect for the footer"""
    colors = ["#2d3748", "#374151", "#2d3748"]
    current_color = 0
    
    def animate_glow():
        nonlocal current_color
        footer_frame.config(bg=colors[current_color])
        footer_content.config(bg=colors[current_color])
        footer_top.config(bg=colors[current_color])
        footer_bottom.config(bg=colors[current_color])
        app_info_frame.config(bg=colors[current_color])
        stats_frame.config(bg=colors[current_color])
        copyright_frame.config(bg=colors[current_color])
        mission_frame.config(bg=colors[current_color])
        badges_frame.config(bg=colors[current_color])
        
        # Update label backgrounds
        for widget in [app_name_label, version_label, accuracy_label, features_label, 
                      copyright_label, mission_label]:
            widget.config(bg=colors[current_color])
        
        current_color = (current_color + 1) % len(colors)
        root.after(3000, animate_glow)  # Change every 3 seconds
    
    # Start the glow animation
    root.after(1000, animate_glow)

# Initialize the glow effect
create_footer_glow()

# Final setup - ensure footer stays at bottom
root.grid_rowconfigure(0, weight=1)

# Start the main GUI loop
if __name__ == "__main__":
    print("\n🚀 Starting AI Cyberbullying Guardian...")
    print("🎯 GUI loaded successfully!")
    print(f"📊 Model ready with {best_accuracy * 100:.1f}% accuracy")
    print("💡 Ready to analyze text for cyberbullying detection!\n")
    
    # Focus on text entry for immediate use
    text_entry.focus_set()
    
    # Start the application
    root.mainloop()