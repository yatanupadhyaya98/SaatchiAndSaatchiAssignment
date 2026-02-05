"""
@file brand_theme_pipeline.py
@brief Synthetic brand mention analysis pipeline using sentiment analysis
       and semantic clustering.

@details
This script implements an end-to-end pipeline that:
- Generates synthetic social media mentions for two retail brands
- Cleans and preprocesses textual content
- Computes sentiment using VADER
- Embeds text using Sentence-BERT
- Clusters mentions into themes using KMeans
- Labels clusters via TF-IDF keywords
- Ranks themes using volume, engagement, and sentiment intensity

All outputs are exported as CSV files for downstream analysis.
"""

# pip install pandas numpy scikit-learn vaderSentiment sentence-transformers

import re
import random
import numpy as np
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


# -----------------------------
# 1) Generate synthetic mentions
# -----------------------------
def synthesize_mentions(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    @brief Generate a synthetic dataset of brand mentions.

    @details
    This function simulates social media mentions for Amazon and MediaMarkt
    across multiple platforms. Each mention is generated from predefined
    templates with controlled probability distributions to ensure that
    dominant themes emerge realistically.

    Ground-truth theme and polarity labels are included for validation
    and debugging purposes.

    @section synth_config Synthesis configuration (documented local elements)

    @subsection brand_weights_doc brand_weights
    @brief Probability distribution for brand occurrence.
    @details
    Amazon is assigned a slightly higher weight (0.55) than MediaMarkt (0.45)
    to reflect its larger online presence and higher expected mention volume
    on social platforms. The values are intentionally close to avoid extreme
    class imbalance while still allowing observable dominance.

    @subsection polarity_weights_doc polarity_weights
    @brief Sentiment polarity distribution per brand.
    @details
    For Amazon, a slight negative skew (52%) is introduced to reflect common
    complaints related to customer service, returns, and third-party sellers.
    MediaMarkt is modeled with a more positive balance (55%) due to its
    in-store support and assisted purchasing experience.
    Values are close to 50/50 to preserve realism and avoid artificial bias.

    @subsection theme_weights_doc theme_weights
    @brief Theme probability distribution conditioned on brand and polarity.
    @details
    These weights are designed so that:
    - Two dominant themes per (brand, polarity) naturally emerge
    - Minor themes remain present but less frequent

    Example:
    - Amazon positive mentions are dominated by price deals (0.42) and fast
      delivery (0.40), reflecting its value and logistics strengths.
    - Amazon negative mentions emphasize customer service and return issues.

    All sub-distributions sum to 1.0 to maintain probabilistic consistency.

    @subsection weighted_choice_doc weighted_choice
    @brief Weighted sampling helper (nested function).
    @details
    This helper is defined inside synthesize_mentions() to keep sampling logic
    scoped locally. Doxygen does not list nested Python functions as standalone
    API entries, so the documentation is provided here to remain visible.

    @subsection rows_doc rows
    @brief Container for synthesized mention records.
    @details
    Each entry represents one synthetic social media mention, including
    brand, platform, text, engagement metrics, and ground-truth labels.

    @subsection loop_doc Main generation loop
    @brief Iteratively construct mention records.
    @details
    For each iteration:
    - A brand is sampled according to brand_weights
    - A platform is sampled uniformly
    - Sentiment polarity is sampled conditionally on brand
    - A theme is sampled conditionally on (brand, polarity)
    - A text template is selected accordingly
    - Engagement metrics are generated using normal distributions

    @subsection engagement_doc Engagement generation
    @brief Engagement is modeled using normal distributions.
    @details
    Engagement metrics are generated using normal distributions to mimic
    real-world variability:
    - Positive mentions receive higher average likes
    - Negative mentions receive more replies (discussion-driven)
    - Shares are kept low and noisy across both polarities

    The max(0, ·) constraint ensures non-negative engagement counts.

    @param n Number of synthetic mentions to generate.
             Default = 100 to keep clustering stable while remaining lightweight.
    @param seed Random seed for reproducibility of sampling and engagement values.
    @return DataFrame containing synthetic mentions and metadata.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Platforms chosen to represent a mix of discussion-heavy and media-heavy channels
    platforms = ["reddit", "youtube", "instagram", "twitter"]

    # Text templates grouped by brand, sentiment polarity, and latent theme
    templates = {
        "amazon": {
            "positive": {
                "good_deals": [
                    "Amazing Black Friday deal—cheaper than everywhere else.",
                    "Best price I found, the discount was actually worth it.",
                    "Great value for money, saved a lot on this electronics purchase."
                ],
                "fast_delivery": [
                    "Delivered the next day—super fast shipping.",
                    "Arrived earlier than expected, delivery was flawless.",
                    "Fast delivery even during Black Friday week, impressive."
                ],
                "easy_ordering": [
                    "Ordering was effortless and checkout was smooth.",
                    "Super convenient to compare options and buy quickly.",
                    "Buying was quick and simple—no hassle."
                ]
            },
            "negative": {
                "bad_customer_service": [
                    "Customer service was useless—no real help at all.",
                    "Support kept sending me in circles, no clear solution.",
                    "It was impossible to reach a helpful human agent."
                ],
                "exchange_returns_pain": [
                    "Exchange was a nightmare, they made it unnecessarily hard.",
                    "Return process was confusing and took too long.",
                    "Refund was delayed and the return instructions were unclear."
                ],
                "delivery_issues": [
                    "Delayed again—still waiting for my package.",
                    "Package arrived damaged / missing items—very frustrating.",
                    "Delivery status kept changing and nothing arrived."
                ],
                "marketplace_trust": [
                    "Got a third-party seller product that felt sketchy.",
                    "The listing was misleading—didn't match what I received.",
                    "Hard to know which sellers are trustworthy."
                ]
            }
        },
        "mediamarkt": {
            "positive": {
                "great_customer_service": [
                    "Staff were super helpful and actually listened to what I needed.",
                    "Great customer service—felt looked after during the purchase.",
                    "They handled my issue quickly and professionally."
                ],
                "helped_choose_product": [
                    "They helped me decide the right laptop for my budget.",
                    "In-store advice was excellent—made me confident buying.",
                    "The staff explained the differences clearly, no pressure."
                ],
                "repair_service": [
                    "They fixed my phone quickly—great service.",
                    "Repair service was smooth and saved me a lot of hassle.",
                    "Got help setting up my device and it worked perfectly."
                ],
                "returns_instore_help": [
                    "Returns were easy in-store, no drama.",
                    "They exchanged the product without making it complicated.",
                    "Refund/return handling was straightforward and fair."
                ],
                "value_for_money": [
                    "Not always the cheapest, but the service made it worth it.",
                    "Good value for money because I got real support.",
                    "Paid a bit more but felt confident and satisfied."
                ]
            },
            "negative": {
                "slightly_expensive": [
                    "A bit more expensive than other online shops for the same product.",
                    "Prices felt slightly higher compared to pure online retailers.",
                    "Good service, but the price wasn’t always the cheapest."
                ],
                "slow_delivery": [
                    "Delivery took too long during Black November.",
                    "Shipping was slower than expected, not ideal.",
                    "Order arrived late, I expected faster delivery."
                ],
                "stock_mismatch": [
                    "Website said in stock, but the store didn’t have it.",
                    "Had to wait because the product wasn’t available.",
                    "Availability info was confusing and not accurate."
                ]
            }
        }
    }

    # Brand distribution chosen to simulate higher Amazon volume
    brand_weights = {"amazon": 0.55, "mediamarkt": 0.45}

    # Polarity distribution reflects known customer experience differences
    polarity_weights = {
        "amazon": {"positive": 0.48, "negative": 0.52},
        "mediamarkt": {"positive": 0.55, "negative": 0.45}
    }

    # Theme weights ensure top-2 themes emerge clearly in clustering
    theme_weights = {
        "amazon": {
            "positive": {"good_deals": 0.42, "fast_delivery": 0.40, "easy_ordering": 0.18},
            "negative": {"bad_customer_service": 0.38, "exchange_returns_pain": 0.34, "delivery_issues": 0.18, "marketplace_trust": 0.10}
        },
        "mediamarkt": {
            "positive": {"great_customer_service": 0.28, "helped_choose_product": 0.24, "repair_service": 0.20, "returns_instore_help": 0.16, "value_for_money": 0.12},
            "negative": {"slightly_expensive": 0.42, "slow_delivery": 0.36, "stock_mismatch": 0.22}
        }
    }

    def weighted_choice(weight_dict: dict) -> str:
        """
        @brief Randomly select a category using weighted probabilities.

        @details
        This helper function abstracts weighted sampling logic to ensure
        consistent and readable probabilistic selection across brands,
        polarities, and themes.

        @param weight_dict Dictionary mapping category names to probability weights.
                           All values are expected to sum to 1.0.
        @return Selected category key based on the provided weight distribution.
        """
        keys = list(weight_dict.keys())
        weights = list(weight_dict.values())
        return random.choices(keys, weights=weights, k=1)[0]

    rows = []

    for i in range(n):
        brand = random.choices(list(brand_weights.keys()), weights=list(brand_weights.values()), k=1)[0]
        platform = random.choice(platforms)

        polarity = random.choices(
            ["positive", "negative"],
            weights=[polarity_weights[brand]["positive"], polarity_weights[brand]["negative"]],
            k=1
        )[0]

        theme = weighted_choice(theme_weights[brand][polarity])
        text = random.choice(templates[brand][polarity][theme])

        base_likes = np.random.normal(loc=22 if polarity == "positive" else 18, scale=14)
        base_replies = np.random.normal(loc=4 if polarity == "positive" else 7, scale=6)
        base_shares = np.random.normal(loc=2 if polarity == "positive" else 1.5, scale=3)

        likes = int(max(0, base_likes))
        replies = int(max(0, base_replies))
        shares = int(max(0, base_shares))

        rows.append({
            "id": f"m_{i:04d}",
            "platform": platform,
            "brand": brand,
            "text": text,
            "true_theme": theme,
            "true_polarity": polarity,
            "likes": likes,
            "replies": replies,
            "shares": shares
        })

    return pd.DataFrame(rows)