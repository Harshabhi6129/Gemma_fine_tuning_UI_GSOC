import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Set page config
st.set_page_config(page_title="Gemma Fine-tuning UI", layout="wide")

# --- Hugging Face Token Login (REQUIRED for Streamlit Cloud) ---
hf_token = os.environ.get("HF_TOKEN")

if hf_token:
    login(token=hf_token)
else:
    st.warning("⚠️ Hugging Face token not found. Please add it in Streamlit Cloud Secrets as 'HF_TOKEN'.")

# --- Title ---
st.title("Gemma Fine-tuning UI")
st.markdown("A professional demonstration app to simulate Gemma model fine-tuning and visualization for GSoC proposal.")

# --- Model Selection ---
st.sidebar.header(" Model Selection")
model_name = st.sidebar.selectbox("Select a base model", ["sshleifer/tiny-gpt2", "distilgpt2", "gpt2"])

with st.spinner("Loading model..."):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
st.sidebar.success(f" Loaded `{model_name}`")

# --- Section 1: Upload Dataset ---
st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a dataset (CSV/JSONL with 'input' and 'output' columns)", type=["csv", "jsonl"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_json(uploaded_file, lines=True)

    st.subheader("Preview of Dataset:")
    st.dataframe(df.head())

    if 'input' in df.columns and 'output' in df.columns:
        st.success(" Dataset format is valid.")
    else:
        st.error(" Dataset must include 'input' and 'output' columns.")

# --- Section 2: Prompt Testing ---
st.header(" 2. Prompt the Base Model")
prompt = st.text_input("Enter a prompt", value="Explain AI in simple terms.")

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        result = generator(prompt, max_length=50, do_sample=True)
        st.success(" Generated Text:")
        st.write(result[0]['generated_text'])

# --- Section 3: Hyperparameter Config ---
st.header(" 3. Configure Hyperparameters")
col1, col2, col3 = st.columns(3)
with col1:
    lr = st.number_input("Learning Rate", value=0.001, format="%.5f")
with col2:
    batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=2)
with col3:
    epochs = st.slider("Epochs", 1, 10, 3)

# --- Section 4: Simulated Fine-tuning ---
st.header("4. Simulate Training and Visualize Loss")
if st.button("Start Simulated Training"):
    if uploaded_file and 'input' in df.columns and 'output' in df.columns:
        st.info(" Simulating training...")
        progress = st.progress(0)
        losses = []

        for epoch in range(epochs):
            time.sleep(0.5)
            simulated_loss = round(1.0 / (epoch + 1) + 0.02 * (epoch % 2), 4)
            losses.append(simulated_loss)
            progress.progress((epoch + 1) / epochs)

        st.success("Training simulation complete!")

        st.subheader(" Loss Curve")
        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), losses, marker='o')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Simulated Loss")
        ax.set_title("Training Loss Visualization")
        st.pyplot(fig)

        st.subheader("Tokenized Input Preview")
        tokens = tokenizer(df['input'][0], return_tensors="pt")
        token_ids = tokens.input_ids.tolist()
        st.json(token_ids)

# Decode and show the readable version
        decoded_text = tokenizer.decode(token_ids[0])
        st.subheader("Decoded Text from Tokens")
        st.code(decoded_text)

    else:
        st.warning("⚠️ Please upload a valid dataset first.")

# --- Section 5: Model Export (Mock) ---
st.header("5. Export Model (Mock)")
if st.button("Download Mock Model"):
    with open("dummy_model.txt", "w") as f:
        f.write("This is a placeholder for your fine-tuned model.")
    with open("dummy_model.txt", "rb") as f:
        st.download_button(" Download File", f, file_name="gemma_finetuned_model.txt")
