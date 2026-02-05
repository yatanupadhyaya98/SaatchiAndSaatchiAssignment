# Social Listening Insight Pipeline (Synthetic Data)

## The complete technical documentation for this project is generated using **Doxygen** and is available online:

**[View the Documentation](https://yatanupadhyaya98.github.io/SaatchiAndSaatchiAssignment/doxygen/html/index.html)**

## Overview

This project demonstrates a **lightweight, explainable social listening analysis pipeline**
designed to support **strategic and creative decision-making** in a marketing or agency context.

Using **synthetic social media comments and interactions**, the pipeline shows how a
Creative Data / Strategy team could:

- Extract sentiment from unstructured text
- Cluster conversations into meaningful themes (e.g. delivery, service, pricing)
- Compare **positive vs negative perceptions** across brands
- Identify **high-impact pain points and strengths** to inform strategic recommendations

The focus is **methodology and insight generation**, not production-scale data engineering.

---

# What the Pipeline Does

This pipeline simulates a real-world social listening and insight extraction workflow used in marketing and brand strategy. It transforms unstructured social media conversations into structured, ranked themes that highlight what consumers positively and negatively associate with each brand.

## Generates synthetic social media mentions
Simulates realistic user comments for two brands across social platforms, including engagement signals such as likes, replies, and shares. This mirrors the structure of data obtained from commercial social listening tools.

## Cleans and normalizes text
Standardizes raw text by lowercasing, removing emojis, URLs, punctuation, and formatting noise to ensure consistent and machine-readable input.

## Performs sentiment analysis
Assigns each mention a sentiment polarity (positive, negative, neutral) along with a sentiment intensity score to capture emotional strength.

## Converts text into semantic embeddings
Transforms each mention into a numerical representation that captures meaning and context, enabling comparison based on semantic similarity rather than keywords.

## Clusters mentions into semantic themes
Groups mentions into themes based on similarity of meaning, allowing differently worded opinions about the same topic to be clustered together.

## Automatically labels themes using TF-IDF
Extracts the most distinctive keywords per cluster to generate human-readable theme labels.

## Ranks themes by impact using a combination of
* Volume (number of mentions)
* Engagement (likes, replies, shares)
* Sentiment intensity

Outputs top positive and negative themes per brand
Produces a ranked list of the most impactful positive and negative themes for each brand, enabling quick comparison and insight generation.



# How the Pipeline Works

This project simulates a real-world **social listening and insight extraction pipeline** used in marketing, strategy, and creative analytics. The goal is to transform large volumes of unstructured social media conversations into **clear, ranked themes** that reveal what people genuinely like or dislike about each brand.

Rather than relying on keywords alone, the pipeline focuses on **meaning**, **emotion**, and **engagement**.

---

## 1. Synthetic Social Media Data Generation

The pipeline begins by generating **synthetic but realistic social media mentions** for two competing brands.
Each mention represents a user post or comment and includes:

* Textual content (what the user says)
* Platform metadata (e.g. Reddit, Twitter, Instagram)
* Engagement signals (likes, replies, shares)

This step mimics the structure of real social listening data when direct access to platforms or paid tools (e.g. Brandwatch, Talkwalker) is not available.
The purpose is to test **methodology and logic**, not to claim real-world truth.
ß
---

## 2. Text Cleaning and Normalization

Raw social media text is noisy and inconsistent. Before analysis, each mention is cleaned to ensure consistency and machine readability.

This includes:

* Converting text to lowercase
* Removing emojis, URLs, hashtags, punctuation
* Normalizing spacing and formatting

This step ensures that different ways of writing the same idea (e.g. “FAST delivery!!!” vs “fast delivery”) are treated as the **same signal**, not separate ones.

---

## 3. Sentiment Analysis

Each cleaned mention is analyzed to determine **emotional polarity and intensity**.

The model assigns:

* Positive, negative, or neutral sentiment
* A sentiment intensity score indicating how strongly the emotion is expressed

This allows the pipeline to distinguish between:

* Mild dissatisfaction vs. strong frustration
* Casual praise vs. enthusiastic advocacy

Sentiment intensity becomes a key input when ranking themes later.

---

## 4. Semantic Embedding of Text

To understand meaning rather than keywords, each mention is converted into a **semantic embedding**.

A semantic embedding represents a sentence as a numerical vector that captures:

* Context
* Intent
* Topic
* Meaning

This allows the system to recognize that differently worded sentences such as
“Shipping arrived the next day” and “Delivery was insanely fast”
are semantically similar, even though they share few words.

This step is the foundation for meaning-based clustering.

---

## 5. Theme Clustering Based on Meaning

Using semantic embeddings, the pipeline groups mentions into **themes** based on similarity of meaning.

Instead of clustering by keywords, it clusters by:

* What people are talking about
* What problem or benefit they are describing

Typical themes that emerge include:

* Delivery speed
* Customer service quality
* Pricing and discounts
* Product reliability
* In-store vs. online experience

Each cluster represents a **shared conversation topic**, not a predefined category.

---

## 6. Automatic Theme Labeling with TF-IDF

Once clusters are formed, they are automatically labeled using **TF-IDF keyword extraction**.

This step identifies the most distinctive words within each cluster compared to the rest of the dataset and uses them to generate short, human-readable theme labels (e.g. “Fast Delivery”, “Poor Support”, “Good In-Store Advice”).

This makes the output interpretable for non-technical stakeholders.

---

## 7. Theme Ranking and Scoring

Each theme is ranked using a combination of:

* **Volume**: how many mentions belong to the theme
* **Engagement**: how much interaction the mentions receive (likes, replies, shares)
* **Sentiment intensity**: how strongly positive or negative the theme is

This ensures that:

* Loud but unimportant topics don’t dominate
* Small but emotionally intense issues are not ignored

---

## 8. Final Output: Top Positive and Negative Themes per Brand

The pipeline outputs:

* The most positive themes per brand
* The most negative themes per brand
* Supporting metrics (volume, engagement, sentiment)

The result is a **clear, ranked view of what matters most to consumers**, suitable for strategic insight, creative briefing, or campaign direction.

---

# Limitations and Practical Constraints

## Synthetic Data vs. Real Consumer Data

This project uses simulated data, which means:

* Patterns reflect assumptions, not reality
* Cultural nuance, slang evolution, and real behavioral noise are limited
* Results demonstrate *process*, not factual market truth

In real-world applications, insights must be validated using **actual consumer-generated content**.

---

## Why Thousands of Data Points Are Necessary

Semantic clustering relies on **statistical density**:

* With thousands of mentions, patterns stabilize
* Themes emerge naturally and consistently
* Outliers are diluted

With only a few hundred records:

* Clusters become unstable
* Themes overlap or fragment
* Rankings fluctuate heavily with small changes

Meaning-based models require **scale** to be reliable.

---

## Why Small Datasets Fail for Strategic Insight

Small datasets:

* Overemphasize individual opinions
* Fail to capture the full range of consumer sentiment
* Produce misleading “top themes” driven by randomness

This is especially risky in marketing and strategy, where decisions impact budgets, messaging, and brand perception.

---

## Computational Requirements and Hardware Constraints

Semantic embeddings and clustering are **computationally expensive**.

Challenges on a typical laptop include:

* High memory usage when embedding thousands of texts
* Long processing times for clustering
* Limited parallelism
* Risk of system slowdown or crashes

In production environments, this pipeline typically runs on:

* Cloud servers (AWS, GCP, Azure)
* GPU-enabled instances
* Scalable memory and compute infrastructure

This allows:

* Faster embedding generation
* Stable clustering at scale
* Continuous data ingestion and updates

---

## Summary

This pipeline demonstrates how raw social conversations can be transformed into **structured, ranked consumer insights** using modern NLP techniques. While synthetic data and limited compute impose constraints, the architecture mirrors real-world systems used in professional social listening, brand strategy, and creative analytics.

The value lies not in the specific outputs, but in the **methodology**:
from signals → meaning → themes → strategic insight.

---


## Tech stack

- Python 3.10
- pandas, numpy
- vaderSentiment (sentiment analysis)
- sentence-transformers (semantic embeddings)
- scikit-learn (clustering, TF-IDF)

The stack is intentionally **lean and interpretable**, suitable for strategy work and interviews.

---

## Setup & Installation

### 1. Create a virtual environment

```bash
python3.10 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

python3 consumerInsight.py