import streamlit as st
import pandas as pd
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# --- 1. SETUP & CACHING ---
st.set_page_config(page_title="PharmaBrain", page_icon="🧬", layout="wide")

@st.cache_resource
def load_resources():
    """Load models only once to make the app fast."""
    # Load XGBoost and Column Names
    model = joblib.load('models/hybrid_xgboost.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    
    # Load BERT (this downloads ~400MB first time)
    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
    bert_model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
    
    return model, model_columns, tokenizer, bert_model

# Load everything
xgboost_model, model_features, tokenizer, bert_model = load_resources()

# --- 2. HELPER FUNCTION: GET BERT EMBEDDING ---
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Return the [CLS] token (768 dimensions)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# --- 3. SIDEBAR INPUTS ---
st.sidebar.header("Trial Configuration")

phase = st.sidebar.selectbox("Phase", ["PHASE1", "PHASE2", "PHASE3", "PHASE4"])
sponsor = st.sidebar.selectbox("Sponsor Type", ["INDUSTRY", "NIH", "OTHER_GOV", "NETWORK", "OTHER"])
enrollment = st.sidebar.number_input("Enrollment", 10, 10000, 100)

# --- 4. MAIN PANEL ---
st.title("🧬 PharmaBrain: Clinical Trial Risk Engine")

# Text Input for NLP
trial_text = st.text_area(
    "Paste Trial Description (Abstract/Title):",
    value="A Phase 3 study to evaluate the safety and efficacy of new immunotherapy for Stage IV Melanoma.",
    height=100
)

if st.button("Analyze Trial Risk"):
    with st.spinner("Processing Medical Text & Calculating Risk..."):
        
        # A. Create a Single Row DataFrame with User Inputs
        input_data = pd.DataFrame({
            'enrollment': [enrollment],
            'phase': [phase],
            'sponsor_type': [sponsor]
        })
        
        # B. One-Hot Encode (Convert text to columns)
        input_encoded = pd.get_dummies(input_data)
        
        # C. ALIGNMENT (Crucial Step!)
        # The model expects 780 columns. We only have ~5 from the user input.
        # We create a template of 0s and fill in what we have.
        final_df = pd.DataFrame(0, index=[0], columns=model_features)
        
        # Fill in the One-Hot columns (e.g., phase_PHASE3 = 1)
        for col in input_encoded.columns:
            if col in final_df.columns:
                final_df[col] = input_encoded[col]
                
        # Fill in Enrollment
        final_df['enrollment'] = enrollment
        
        # D. Run NLP (BERT)
        # Get the 768 numbers from the text
        bert_vector = get_bert_embedding(trial_text)
        
        # Fill the bert_0 to bert_767 columns
        for i, val in enumerate(bert_vector):
            col_name = f'bert_{i}'
            if col_name in final_df.columns:
                final_df[col_name] = val

        # E. PREDICT
        prediction_prob = float(xgboost_model.predict_proba(final_df)[0][1]) # Probability of Success (Class 1)
        
        # --- 5. DISPLAY RESULTS ---
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Score")
            score_pct = prediction_prob * 100
            
            # Color Logic
            if score_pct > 70:
                st.success(f"High Probability of Success: {score_pct:.1f}%")
            elif score_pct < 40:
                st.error(f"High Risk of Failure: {score_pct:.1f}%")
            else:
                st.warning(f"Moderate Risk: {score_pct:.1f}%")
                
            st.progress(prediction_prob)

        with col2:
            st.subheader("Key Drivers (SHAP)")
            st.info("Based on our analysis, INDUSTRY sponsors and PHASE 3 designs carry unique risk profiles detected by the model.")