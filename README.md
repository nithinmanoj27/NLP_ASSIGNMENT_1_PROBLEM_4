# Sports vs Politics Text Classification Using Machine Learning

##  Project Overview

This project implements a text classification system that automatically categorizes text documents into **Sports** or **Politics**. The goal is to explore how different machine learning algorithms perform on textual data and compare their effectiveness using the same dataset and feature representation.

The project demonstrates the complete workflow of a Natural Language Processing (NLP) task, including dataset preparation, feature extraction, model training, evaluation, and analysis of results.

---

##  Problem Statement

With the increasing volume of digital text data, automatically classifying documents into categories is essential for organizing information efficiently.  

In this project, we design a classifier that reads a text document and predicts whether it belongs to the **Sports** or **Politics** category. The system also compares multiple machine learning techniques to understand their performance differences.

---

##  Dataset

The dataset used in this project is derived from the **BBC News Dataset**, which contains real news articles categorized into multiple domains.

For this implementation:

- Only **Sports** and **Politics** categories were selected.
- Articles were converted into **sentence-level samples** to increase the number of training instances.
- The dataset contains domain-specific vocabulary that helps distinguish between the two classes.

### Example Sports Keywords
match, player, tournament, medal, training

### Example Politics Keywords
government, policy, parliament, election, minister

---

##  Methodology

The project follows the standard NLP pipeline:

1. Data Collection
2. Data Preprocessing
3. Feature Extraction
4. Model Training
5. Model Evaluation
6. Performance Comparison

---

##  Feature Representation

Text data was converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF was chosen because:

- It highlights important words in a document.
- Reduces the impact of common words.
- Works well for text classification problems.
- Provides meaningful representation of textual data.

---

##  Machine Learning Models Used

Three different machine learning algorithms were implemented and compared:

### 1️ Naive Bayes
A probabilistic classifier based on Bayes’ theorem. It performs well for text classification due to its ability to model word frequency distributions.

### 2️ Logistic Regression
A linear classification algorithm that predicts class probabilities using a logistic function. It is simple yet effective for many NLP tasks.

### 3️ Support Vector Machine (SVM)
A powerful classifier that finds the optimal decision boundary separating classes. It performs well in high-dimensional feature spaces.

---

##  Results

The performance of each model was evaluated using **accuracy**.

| Model | Accuracy |
|------|---------|
Naive Bayes | 1.00 |
Logistic Regression | 0.83 |
SVM | 0.83 |

### Result Analysis

Naive Bayes achieved the highest accuracy because it performs particularly well on small text datasets with distinct vocabulary. Logistic Regression and SVM also performed well but were slightly affected by the limited amount of training data.

---

##  How to Run the Project

### Step 1 — Clone Repository


### Step 2 — Navigate to Project Folder


### Step 3 — Install Dependencies

Make sure Python is installed. Then install required library:


### Step 4 — Run Classifier


The program will train the models and display accuracy results.

---

##  PROBLEM Structure

sports-politics-classifier
│
├── problem4_classifier.py # Main classification script
├── sports.txt # Sports dataset
├── politics.txt # Politics dataset
├── B22CS066_prob4.pdf # Detailed project report
└── README.md # Project documentation


---

##  Limitations

- The dataset size is relatively small compared to real-world applications.
- Some words may appear in both domains, leading to possible ambiguity.
- The model relies only on textual features without deep semantic understanding.
- Larger datasets and advanced models could improve performance.

---

##  Future Improvements

- Use larger datasets for better generalization.
- Experiment with n-gram features.
- Apply deep learning models such as LSTM or Transformers.
- Perform hyperparameter tuning.
- Add visualization of results.

---

##  Author

Yerra Nithin Manoj  
Roll Number: B22CS066  

---

##  Course

Natural Language Understanding — Assignment 1  

---

##  Conclusion

This project demonstrates how machine learning can be applied to solve text classification problems. By comparing multiple models and analyzing results, the study highlights the importance of dataset preparation and feature representation in achieving accurate predictions.
