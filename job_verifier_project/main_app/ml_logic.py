# authentijob_app/ml_logic.py
import re
import pandas as pd
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
import nltk
from nltk.corpus import stopwords

# --- Download NLTK data (same as your Streamlit code) ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

# --- Copy your `is_valid_job_post` function here ---
def is_valid_job_post(text):
    text_lower = text.lower()
    keywords = [
        'responsibilities', 'requirements', 'qualifications', 'experience',
        'skills', 'hiring', 'apply', 'job description', 'benefits'
    ]
    found_keywords = sum(1 for keyword in keywords if keyword in text_lower)
    return found_keywords >= 2

# --- Copy and adapt your `predict` function here ---
# It will now take the loaded artifacts as arguments
def predict(input_data, model, vocab, train_df_cols, device, threshold=0.3):
    tokenizer = get_tokenizer('basic_english')
    text_pipeline = lambda x: vocab(tokenizer(x))
    
    # --- All your helper functions from the original predict function go here ---
    def clean_text_pred(text):
        # ... (copy from your streamlit code)
        text = str(text).lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def create_features_for_prediction(df):
        # ... (copy from your streamlit code)
        df['text'] = (df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['requirements'].fillna(''))
        df['text'] = df['text'].apply(clean_text_pred)
        for col in ['telecommuting', 'has_company_logo', 'has_questions']:
            if col in df.columns: df[col] = df[col].astype(float)
        categorical_cols = ['employment_type', 'required_experience', 'required_education']
        cols_to_encode = [col for col in categorical_cols if col in df.columns]
        return pd.get_dummies(df, columns=cols_to_encode, dummy_na=True, drop_first=True)

    def extract_structured_data(raw_text):
        # ... (copy from your streamlit code)
        data = { 'title': '', 'description': '', 'requirements': '', 'employment_type': 'Unspecified' }
        lines = raw_text.strip().split('\n')
        data['title'] = lines[0].strip() if lines else ''
        data['description'] = raw_text
        return data

    # --- Main prediction logic ---
    # ... (copy the rest of your predict function logic here)
    if isinstance(input_data, str):
        structured_data = extract_structured_data(input_data)
        df_pred = pd.DataFrame([structured_data])
    else:
        raise ValueError("Input data must be a raw string.")

    df_pred_processed = create_features_for_prediction(df_pred)
    
    current_cols = df_pred_processed.columns.tolist()
    missing_cols = set(train_df_cols) - set(current_cols)
    for c in missing_cols:
        df_pred_processed[c] = 0
    df_pred_processed = df_pred_processed[train_df_cols]

    text_to_predict = df_pred_processed['text'].iloc[0]
    tabular_cols = [c for c in train_df_cols if c not in ['text', 'fraudulent']]
    tabular_to_predict = df_pred_processed[tabular_cols].values.astype(np.float32)

    processed_text = torch.tensor(text_pipeline(text_to_predict), dtype=torch.long).to(device)
    tabular_tensor = torch.tensor(tabular_to_predict, dtype=torch.float32).to(device)
    text_tensor_unsqueezed = processed_text.unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(text_tensor_unsqueezed, tabular_tensor)
        probability = torch.sigmoid(prediction).item()

    result_label = "Fake Job" if probability >= threshold else "Real Job"
    return result_label, probability