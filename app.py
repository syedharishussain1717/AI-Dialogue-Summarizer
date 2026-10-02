import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Hugging Face model
MODEL_NAME = "syedharis1717/distilgpt2-samsum-finetuned"


# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    return tokenizer, model


tokenizer, model = load_model()


# Page settings
st.set_page_config(
    page_title="Dialogue Summarizer",
    page_icon="📝",
    layout="centered"
)


# Title
st.title("📝 Dialogue Summarizer")

st.write(
    "Enter a conversation below and let the fine-tuned AI model "
    "generate a short summary."
)


# User input
dialogue = st.text_area(
    "Enter Dialogue",
    height=200,
    placeholder="Example:\nJohn: Are you coming tomorrow?\nSarah: Yes, I will come in the morning."
)


# Generate button
if st.button("Generate Summary"):

    if dialogue.strip() == "":
        st.warning("Please enter a dialogue first.")

    else:

        prompt = f"""Dialogue:
{dialogue}

Summary:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=60,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                do_sample=False
            )

        result = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        # Extract summary
        summary = result.split("Summary:")[-1].strip()

        st.subheader("Generated Summary")

        st.success(summary)