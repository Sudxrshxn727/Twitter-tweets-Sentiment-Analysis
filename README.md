# Twitter Tweets Sentiment Analysis

This repository contains a Python project for sentiment analysis of Twitter text tweets. The sentiment analysis classifies tweets into two categories: "racist/sexist" (labeled as 1) and "non-racist" (labeled as 0). The project utilizes various natural language processing (NLP) techniques and machine learning models to achieve this classification.

## Project Overview

The project follows these key steps:

1. **Data Loading**: The first step involves reading a CSV file containing the Twitter text tweets. This is done using the Pandas library.

2. **Data Visualization**: Visualizations are created to show the distribution of tweets that are classified as racist/sexist and non-racist. These visualizations help in understanding the dataset.

3. **Data Cleaning**: Text data preprocessing is performed. This includes:
   - Tokenization: Splitting text into words or tokens.
   - Lemmatization: Reducing words to their base or root forms.
   - Stopword Removal: Eliminating common English stopwords.

4. **Feature Extraction**: The text data is converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique. This forms the vocabulary for the machine learning models.

5. **Data Splitting**: The dataset is split into training and testing sets (typically a 70-30 split). This is done using the `train_test_split` function from scikit-learn.

6. **Model Building and Training**:
   - **Logistic Regression Model**: A Logistic Regression model is applied to the training data and fitted to it.

7. **Model Testing and Evaluation**:
   - The trained Logistic Regression model is tested using the testing data:
     - Accuracy: Approximately 95%
     - F1 Score: 47%

8. **Alternative Model Testing**:
   - An alternative model, the **Multinomial Naive Bayes Model**, is tested on the same data:
     - Accuracy: 94.6%
     - F1 Score: 95.8%

These numerical values indicate the performance of the models in classifying Twitter text tweets. The Multinomial Naive Bayes model demonstrates a higher F1 score, suggesting better precision and recall compared to the Logistic Regression model.

## Libraries and Modules Used

The following Python libraries and modules were used in this project:

- `pandas` for data loading and manipulation
- `nltk` (Natural Language Toolkit) for text preprocessing
  - `word_tokenize` for tokenization
  - `WordNetLemmatizer` for lemmatization
  - `stopwords` for removing common English stopwords
- `sklearn` (scikit-learn) for machine learning and evaluation
  - `TfidfVectorizer` for feature extraction
  - `train_test_split` for data splitting
  - `LogisticRegression` for logistic regression modeling
  - `MultinomialNB` for Multinomial Naive Bayes modeling
  - `accuracy_score` for accuracy calculation
  - `f1_score` for F1 score calculation
  - `classification_report` for generating a classification report
  - `confusion_matrix` for generating a confusion matrix

Please refer to the code in this repository for a detailed implementation of the project and any further improvements or modifications. If you have any questions or need assistance, feel free to reach out to the project contributors.

**Note**: This README serves as an overview. Detailed code, data, and results can be found in the project files.

