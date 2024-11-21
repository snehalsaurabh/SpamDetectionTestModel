import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Title of the Streamlit app
st.title("Email Spam Classifier")
st.write("This app classifies emails as **Spam** or **Ham** using Random Forest, Naive Bayes, and Logistic Regression.")

# Load and preprocess dataset
@st.cache_data
def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('mail_data.csv', encoding='latin-1')

    # Keep only relevant columns and rename for consistency
    df = df[['Category', 'Message']]  # Adjust based on dataset columns
    df.columns = ['label', 'text']

    # Map labels to binary values: spam = 1, ham = 0
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    return df

# Train all models
@st.cache_resource
def train_models(df):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42
    )

    # Convert text data into numerical format using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)

    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)

    # Evaluate models
    models = {"Random Forest": rf_model, "Naive Bayes": nb_model, "Logistic Regression": lr_model}
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        metrics[name] = {"accuracy": accuracy, "report": report, "matrix": matrix}

    return rf_model, nb_model, lr_model, vectorizer, metrics

# Load and preprocess data
st.write("### Step 1: Train the Models")
df = load_and_preprocess_data()
st.write("Dataset loaded successfully!")
st.write(df.head())

# Train all models
rf_model, nb_model, lr_model, vectorizer, metrics = train_models(df)

# Display training results
st.write("### Model Performance")
for name, metric in metrics.items():
    st.write(f"**{name}**")
    st.write(f"Accuracy: {metric['accuracy']:.2f}")
    st.write("Classification Report:")
    st.text(metric['report'])
    st.write("Confusion Matrix:")
    st.write(metric['matrix'])

# Real-time email classification
st.write("### Step 2: Test with Custom Input")
input_email = st.text_area("Enter the email text here:")

# Predict button
if st.button("Classify"):
    if input_email.strip():
        # Preprocess and vectorize the input
        input_tfidf = vectorizer.transform([input_email])

        # Make predictions with all models
        rf_prediction = rf_model.predict(input_tfidf)[0]
        rf_prob = rf_model.predict_proba(input_tfidf)[0]

        nb_prediction = nb_model.predict(input_tfidf)[0]
        nb_prob = nb_model.predict_proba(input_tfidf)[0]

        lr_prediction = lr_model.predict(input_tfidf)[0]
        lr_prob = lr_model.predict_proba(input_tfidf)[0]

        # Display results for each model
        st.write("### Classification Results")
        # Random Forest
        if rf_prediction == 1:
            st.error(f"**Random Forest**: Spam (Confidence: {rf_prob[1]:.2f})")
        else:
            st.success(f"**Random Forest**: Ham (Confidence: {rf_prob[0]:.2f})")

        # Naive Bayes
        if nb_prediction == 1:
            st.error(f"**Naive Bayes**: Spam (Confidence: {nb_prob[1]:.2f})")
        else:
            st.success(f"**Naive Bayes**: Ham (Confidence: {nb_prob[0]:.2f})")

        # Logistic Regression
        if lr_prediction == 1:
            st.error(f"**Logistic Regression**: Spam (Confidence: {lr_prob[1]:.2f})")
        else:
            st.success(f"**Logistic Regression**: Ham (Confidence: {lr_prob[0]:.2f})")
    else:
        st.warning("Please enter an email to classify.")
