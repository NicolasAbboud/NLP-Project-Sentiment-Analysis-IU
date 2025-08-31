# 🎬 Sentiment Analysis on Movie Reviews  

This project implements a **Natural Language Processing (NLP)** system for sentiment analysis of movie reviews. The objective is to classify reviews as **positive** or **negative** using both **classical machine learning methods** (Naive Bayes, Support Vector Machines) and **deep learning approaches** (Neural Networks).  

---

## 📌 Project Objectives  
- Collect and preprocess movie reviews data.  
- Apply NLP preprocessing (stopword removal, lemmatization, entity filtering).  
- Train sentiment classifiers with both **classical ML** and **deep learning**.  
- Evaluate models on test data using standard metrics.  
- Deploy a prediction pipeline that can analyze new, unseen reviews.  

---

## 📂 Project Structure  

![Project Structure](https://github.com/NicolasAbboud/NLP-Project-Sentiment-Analysis-IU/blob/main/Screenshot_2.png?raw=true)

---

## 📊 Dataset  

For this project, the **[Stanford Large Movie Review Dataset (IMDb dataset)](https://ai.stanford.edu/~amaas/data/sentiment/)** was used, which is a widely used benchmark in sentiment analysis research:

> Maas, A., Daly, R., Pham, P., Huang, D., Ng, A., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (pp. 142–150). Association for Computational Linguistics. https://aclanthology.org/P11-1015/  

---

## ⚙️ Preprocessing  

The preprocessing pipeline includes:  
- Removing HTML tags and punctuation  
- Lowercasing and normalizing whitespace  
- Removing stopwords (with custom stopword list)  
- Lemmatization with **spaCy**  
- Filtering out named entities of type *PERSON* to reduce noise  

---

## 🤖 Models  

### Classical Machine Learning  
- **Naive Bayes (BernoulliNB)** with TF–IDF features  
- **Support Vector Machines (SVM)** with TF–IDF features  

### Deep Learning  
- **Feed-Forward Neural Network** (trained in `deep_learning.ipynb`)  
- Input text vectorized with TensorFlow’s `TextVectorization`  
- Architecture:  
  - Input → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dense(1, Sigmoid)  

---

## 🧪 Results  

- The **neural network model** achieved the best generalization performance, outperforming Naive Bayes and SVM.  
- Evaluation metrics include accuracy, confusion matrices, and validation curves (see `/results/`).  

---

## 🚀 Usage  

### 1. Clone repository  
```bash
git clone https://github.com/NicolasAbboud/NLP-Project-Sentiment-Analysis-IU.git
cd NLP-Project-Sentiment-Analysis-IU
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run prediction
```python
from tensorflow.keras.models import load_model
from predictor import predict_sentiment

model = load_model("./models/model_NN_final.keras")
review = "The movie is funny and heartwarming!"
predict_sentiment(model, review)
```
```text
Review preview: The movie is funny and heartwarming!...
Sentiment: Positive :) (score=0.8704)
```

## 📈 Future Work
- Integrating transformer-based architectures (e.g., BERT, DistilBERT).
- Exploring cross-domain sentiment analysis (beyond movies).
- Improving interpretability with attention or explainable AI methods.

## 📚 References
> Birjali, M., Kasri, M., & Beni-Hssane, A. (2021). A Comprehensive Survey on Sentiment analysis: Approaches, Challenges and Trends. Knowledge-Based Systems, 226(1), 107–134. https://doi.org/10.1016/j.knosys.2021.107134

> Maas, A., Daly, R., Pham, P., Huang, D., Ng, A., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (pp. 142–150). Association for Computational Linguistics. https://aclanthology.org/P11-1015/

> Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8(4). https://doi.org/10.1002/widm.1253



