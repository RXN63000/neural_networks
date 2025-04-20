# Repository Overview
This repository contains the implementation of four tasks related to NLP and deep learning:

NLP Preprocessing Pipeline (Tokenization, Stopword Removal, Stemming)

Named Entity Recognition (NER) with SpaCy

Scaled Dot-Product Attention Implementation

Sentiment Analysis using HuggingFace Transformers

Each task includes Python code and answers to short-answer questions.

# Short Answer Questions
## Q1: NLP Preprocessing Pipeline
### 1. Difference between Stemming and Lemmatization
Stemming reduces words to their root form by chopping off suffixes (e.g., "running" → "run").

Lemmatization uses vocabulary and morphological analysis to return the base or dictionary form (lemma) of a word (e.g., "running" → "run").

Example:

Stemming: "running" → "runn" (Porter Stemmer)

Lemmatization: "running" → "run"

### 2. Stopword Removal: Usefulness vs. Harm
Useful:

Reduces noise in text data.

Improves computational efficiency.

Focuses on meaningful words (e.g., in search engines or topic modeling).

Harmful:

Can remove important context (e.g., "not" is a stop word but crucial for sentiment analysis).

May degrade performance in tasks like machine translation or question answering.

## Q2: Named Entity Recognition (NER) with SpaCy
### 1. NER vs. POS Tagging
NER identifies and classifies named entities (e.g., people, organizations, locations).

POS Tagging identifies grammatical roles (e.g., noun, verb, adjective).

### 2. Real-World NER Applications
Search Engines: Improves search results by recognizing entities in queries.

Customer Support: Automatically extracts product names, locations, or dates from user queries.

## Q3: Scaled Dot-Product Attention
### 1. Why Divide by √d?
Prevents dot products from becoming too large when dimensionality (d) is high.

Ensures stable gradients in softmax for better training.

### 2. How Self-Attention Helps Models
Weighs the importance of different words in a sentence.

Captures long-range dependencies between words regardless of position.

## Q4: Sentiment Analysis with HuggingFace Transformers
### 1. BERT vs. GPT Architecture
BERT: Uses a bidirectional encoder (processes entire input at once).

GPT: Uses a unidirectional decoder (processes tokens left-to-right).

### 2. Benefits of Pre-trained Models
Transfer Learning: Leverages knowledge from large datasets.

Efficiency: Saves time/resources compared to training from scratch.

Performance: Works well even with limited task-specific data.

# Ravi Teja Reddy Nomula
## 700756300
