# CAPITOL: Classifying American Political Ideologies Through Online Language

***Project Status:** Completed (academic project 2025-2025)*

## Project overview
The goal of **CAPITOL** is to classify U.S. political tweets based on their content, distinguishing between *Democratic* and *Republican* affiliations. 
This project explores the entire NLP pipeline: from preprocessing raw social media text to comparing traditional Machine Learning baselines against Deep Learning architectures (RNNs/LSTMs).

The study highlights how different feature representation techniques (sparse vs dense embeddings) impact classification performance on a balanced dataset of 30,000 tweets.

## Technologies and Tools
* **Language:** Python
* **Libraries:** TensorFlow/Keras, Scikit-Learn, NLTK, Gensim, XGBoost, Pandas, NumPy, Matplotlib/Seaborn.
* **Techniques:** TF-IDF, word embeddings (GloVe, FastText), LSTM, Logistic Regression.


## Methodology

### 1. Preprocessing pipeline
We implemented a robust cleaning pipeline to handle social media noise:
* **Hashtag decomposition:** using 'wordninja' to split tags like "#BuildBackBetter" into "Built Back Better".
* **Cleaning:** removal of emojis, URLs, mentions and non-ASCII characters.
* **Tokenization and padding:** standardized input length (31 tokens) for neural networks.


### 2. Feature representation
We compared two main approaches:
* **Sparse:** ***TF-IDF*** vectorization (bag-of-words approach)
* **Dense:**
  * ***Pre-trained GloVe:*** (glove.6B.100d) trained on Wikipedia/Web.
  * ***Custom FastText:*** trained from scratch on our specific political dataset to capture domain specific slang (e.g. "libtard", "trumper").
 

### 3. Models
We trained and evaluated four distinct configurations:
1. ***Logistic regression*** + TF-IDF (baseline)
2. ***XGBoost*** + TF-IDF
3. ***LSTM (an RNN)*** + pre-trained GloVe
4. ***LSTM (an RNN)*** + custom FastText

## Results
The evaluation was performed on a held-out test using accuracy and F1-score. 

<img width="1163" height="572" alt="image" src="https://github.com/user-attachments/assets/c0b2a28b-435f-48cd-b67f-b52ec3c29088" />


**Key insights:** while the deep learning model with custom FastText embeddings performed competitively (0.82), the simpler logistic regression achieved the best results (0.84). This suggests that specific keywords (captured well by TF-IDF) are highly discriminative markers for political affiliation in this dataset. 

## What I learned:
* How to train custom word embeddings (FastText) to capture sematnic relationsgips in niche domains (political slang).
* The trade-off betweem model complexity and performance: complex LSTMs are not always better than linear models if the features are distinct enough.
* Handling overfitting in neural networks using dropout and validation monitoring.


## Credits
Developed by ***Tsuhuy Ecaterina*** and ***Brusati Lorenzo*** for the "Text Mining and Natural Language Processing" course. 
