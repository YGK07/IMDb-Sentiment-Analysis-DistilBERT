# IMDb Sentiment Analysis using DistilBERT

## Project Overview

This project performs Sentiment Analysis on IMDb Movie Reviews using the DistilBERT Transformer model.

The model classifies movie reviews into:

- Positive Review
- Negative Review

The project was implemented using:

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-Learn
- Google Colab

---

## Dataset

IMDb Movie Review Dataset

Dataset Structure:

aclImdb/
├── train/
│ ├── pos/
│ └── neg/
│
├── test/
│ ├── pos/
│ └── neg/

Training Samples Used: 10,000

Testing Samples Used: 2,000

---

## Model Architecture

DistilBERT Transformer

Pipeline:

Text Review
↓
DistilBERT Tokenizer
↓
DistilBERT Encoder
↓
Classification Head
↓
Positive / Negative Prediction

---

## Features

- Transformer-based NLP model
- Pretrained DistilBERT
- Fine-tuned on IMDb reviews
- Early Stopping
- Accuracy Tracking
- Precision Tracking
- Recall Tracking
- F1 Score Tracking
- Confusion Matrix
- Sentiment Prediction on Custom Reviews
- Training and Validation Curves

---

## Libraries Used

```python
torch
transformers
datasets
numpy
pandas
scikit-learn
matplotlib
```

## Training Results

| Metric | Score |
|----------|----------|
| Accuracy | 89.2% |
| Precision | 89% |
| Recall | 89% |
| F1 Score | 89% |

---

## Confusion Matrix

| | Predicted Negative | Predicted Positive |
|----|----|----|
| Actual Negative | 916 | 84 |
| Actual Positive | 132 | 868 |

---

## Example Predictions

Review:

```text
This movie was absolutely amazing and brilliant
```

Prediction:

```text
Positive
Confidence: 99.14%
```

Review:

```text
Worst movie I have ever watched
```

Prediction:

```text
Negative
Confidence: 99.30%
```

---

## Learning Outcomes

Through this project I learned:

- Natural Language Processing (NLP)
- Tokenization
- Transformers
- Self-Attention Mechanism
- DistilBERT Architecture
- Fine-Tuning Pretrained Models
- Evaluation Metrics
- Deep Learning using PyTorch
- Hugging Face Transformers

---

## Future Improvements

- Train on full IMDb dataset
- Hyperparameter tuning
- Compare BERT vs DistilBERT
- Deploy as a web application
- Create API endpoint for predictions

---

## Author

Yohan George

B.Tech Computer Science with AI ML

VIT Vellore

Specialization: Artificial Intelligence and Machine Learning
