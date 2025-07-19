# Fake-News-prediction

# 📰 Fake News Detection Using Machine Learning

## Overview
This project aims to predict whether a news article is **fake or real** using various **Machine Learning algorithms** and **Natural Language Processing (NLP)** techniques. It includes a comparative study of classification models implemented using Python.

## 🛠 Tech Stack

- **Python 3.x**
- **Jupyter Notebook** (via Anaconda)
- **Scikit-learn** for machine learning models
- **Pandas & NumPy** for data manipulation
- **NLTK** for NLP tasks
- **Matplotlib & Seaborn** for data visualization

## 🚀 Features

- Predict fake or real news using machine learning
- NLP preprocessing and vectorization (TF-IDF)
- Model training and evaluation (Logistic Regression, Naive Bayes, SVM, etc.)
- Performance metrics comparison (accuracy, precision, recall, F1-score)
- Visualization of classification results


## ⚙️ Setup Instructions

### ✅ Prerequisites
- Python 3.x installed
- Jupyter Notebook (via [Anaconda](https://www.anaconda.com/))
- (Optional) Create a virtual environment for isolation

### 📦 Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kalaiselvimurugan/Fake-News-prediction.git
   cd Fake-News-prediction
   ```
   
2. **Create and Activate a Virtual Environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   
5. **Run the Notebook**
   ```bash
   Open notebooks/Fake_news_prediction_code.ipynb and run the cells step-by-step.
   ```

## 🔁 Model Workflow
🧹 Data Preprocessing
Tokenization
Stopword removal
Lowercasing
Vectorization using TF-IDF

🧠 Model Training
Logistic Regression
Naive Bayes
Support Vector Machine (SVM)

📊 Model Evaluation
Classification Report (Precision, Recall, F1-score)
Confusion Matrix

📈 Visualization
Performance comparison using plots

## 📑 Sample Dataset
The dataset (fake_or_real_news.csv) contains:
```markdown
title – Title of the news article
text – Full content of the news
label – Binary label: 1 for fake, 0 for real
```

## 📚 Learn More
[Scikit-learn Documentation](https://scikit-learn.org/stable/)
[NLTK Documentation](https://www.nltk.org/book/)
[Jupyter Notebook](https://jupyter.org/)
[Pandas Documentation](https://pandas.pydata.org/docs/)
