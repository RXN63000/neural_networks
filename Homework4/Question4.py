# -*- coding: utf-8 -*-
"""Welcome To Colab

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/notebooks/intro.ipynb
"""

from transformers import pipeline

def analyze_sentiment(sentence):
    classifier = pipeline("sentiment-analysis")
    result = classifier(sentence)[0]

    print(f"Sentiment: {result['label']}")
    print(f"Confidence Score: {result['score']:.4f}")

# Test the function
sentence = "Despite the high price, the performance of the new MacBook is outstanding."
analyze_sentiment(sentence)