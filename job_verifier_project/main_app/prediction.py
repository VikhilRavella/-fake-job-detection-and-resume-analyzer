# prediction.py
# ===============================
# ❌ ALL ML LOGIC DISABLED
# ===============================

# import re
# import pandas as pd
# import numpy as np
# import torch
# from torchtext.data.utils import get_tokenizer
# from main_app.apps import MainAppConfig


# -------------------------------
# ✅ BASIC VALIDATION (SAFE)
# -------------------------------
def is_valid_job_post(text):
    text_lower = text.lower()
    keywords = [
        'responsibilities', 'requirements', 'qualifications', 'experience',
        'skills', 'hiring', 'apply', 'job description', 'benefits'
    ]
    found_keywords = sum(1 for keyword in keywords if keyword in text_lower)
    return found_keywords >= 2


# -------------------------------
# ❌ ML PREDICTION DISABLED
# -------------------------------
def predict(input_data, threshold=0.3):
    """
    ML prediction is disabled for Render free tier.
    This function exists only to prevent import errors.
    """

    # model = MainAppConfig.model
    # vocab = MainAppConfig.vocab
    # train_df_cols = MainAppConfig.train_df_cols
    # device = MainAppConfig.device
    # stop_words = MainAppConfig.stop_words

    # tokenizer = get_tokenizer('basic_english')
    # text_pipeline = lambda x: vocab(tokenizer(x))

    # def clean_text_pred(text):
    #     text = str(text).lower()
    #     text = re.sub(r'<.*?>', ' ', text)
    #     text = re.sub(r'[^a-z\s]', '', text)
    #     text = ' '.join([word for word in text.split() if word not in stop_words])
    #     return text

    # def create_features_for_prediction(df):
    #     df['text'] = (
    #         df['title'].fillna('') + ' ' +
    #         df['description'].fillna('') + ' ' +
    #         df['requirements'].fillna('')
    #     )
    #     df['text'] = df['text'].apply(clean_text_pred)
    #     return df

    # def extract_structured_data(raw_text):
    #     return {
    #         'title': raw_text.split('\n')[0] if raw_text else '',
    #         'description': raw_text,
    #         'requirements': '',
    #         'employment_type': 'Unspecified'
    #     }

    # ---------------------------
    # TEMPORARY DUMMY RESPONSE
    # ---------------------------
    return "ML Disabled", 0.0
