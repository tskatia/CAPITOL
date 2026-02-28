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


## Issues Encountered

Throughout the project, we faced limited computational resources on Deepnote, which led us to reduce the dataset to 30,000 samples to ensure manageable training times. Library compatibility issues also slowed down model experimentation. Additionally, downloading large pre-trained models like GloVe increased setup time and disk usage within the constrained cloud environment.

## Project Recap

In summary, our project involved several key steps, preprocessing political tweet data, applying various vectorization and embedding techniques, implementing both classical and deep learning models, and evaluating their performance. This hands-on experience enhanced our understanding of NLP workflows and highlighted how different textual representations and algorithms influence classification outcomes.

## Final Conclusion

Although the performance varied across models, all approaches demonstrated the ability to identify political affiliation in text with reasonable accuracy. Logistic Regression with TF-IDF stood out for its simplicity and effectiveness. Meanwhile, LSTM models with embeddings offered strong performance while also capturing sequential patterns in language. The project provided us with practical experience in implementing and evaluating a real-world NLP task using multiple modeling techniques.

## Contributing

If you have suggestions for improvements, new experiments, or alternative embeddings to try, feel free to open an issue or submit a pull request. Let’s continue improving this project together!


## Credits
Developed by ***Tsuhuy Ecaterina*** and ***Brusati Lorenzo*** for the "Text Mining and Natural Language Processing" course. 
