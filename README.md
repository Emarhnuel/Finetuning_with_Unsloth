# Finetuning_with_Unsloth

## Overview

Problem:  Large language models (LLMs) demonstrate incredible capabilities, but they're often too general-purpose for specific tasks. Finetuning helps adapt these models to particular domains, such as sentiment analysis, by training them on task-relevant data.
Unsloth's Role: Unsloth is a specialized tool that aims to make the finetuning of large language models significantly faster and more memory-efficient than traditional methods. I chose Unsloth because its optimizations promise to streamline the finetuning process and make it more accessible, even with limited computing resources.
Dataset: The IMDB dataset is a widely used benchmark for sentiment analysis. It contains a collection of movie reviews labeled as positive or negative.  This dataset is well-suited for this project because it provides a large and diverse set of text examples for finetuning a model to understand the nuances of sentiment in language.

## Key Features
1. Model: Mistral-7B-BNB-4bit
2. Finetuning Technique: LoRA with Unsloth's optimizations.
3. Performance Improvements: If you observed speed or memory gains, quantify them here (e.g., "Achieved 2x faster fine-tuning compared to baseline LORA").

   
## Installation
```
pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
pip install "git+https://github.com/huggingface/transformers.git"
```

## Usage
1. **Data Preparation:**
Download the IMDB dataset using Hugging Face's load_dataset function.

2. **Finetuning:**
Load the Mistral-7B-BNB-4bit model.
Apply Unsloth's LoRA adaptations.
Use the `SFTTrainer` for the finetuning process.

## Inference
Here's how to load the fine-tuned model, tokenizer, and use it for sentiment prediction:
```
Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "lora_model"  # Update with the path where your finetuned model is saved
tokenizer = AutoTokenizer.from_pretrained(model_path) 
model = AutoModelForSequenceClassification.from_pretrained(model_path)

review = "This movie was truly inspiring and beautifully acted."
inputs = tokenizer(review, return_tensors="pt")
outputs = model(**inputs)

predicted_label = outputs.logits.argmax().item()
print(f"Predicted Sentiment: {tokenizer.decode(predicted_label)}")
```

### Explanation:
1. Loading: The fine-tuned model and its associated tokenizer are loaded from your saved directory.
2. Preprocessing: The user's review is tokenized (converted into numerical representations).
3. Prediction: The model processes the input and generates sentiment scores.
4. Interpretation: The score with the highest value is selected, and the tokenizer decodes it back into a human-readable sentiment label.
