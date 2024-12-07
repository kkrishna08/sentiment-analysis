Reddit Comment Sentiment Classifier
This repository contains a Reddit comment sentiment classifier designed to predict whether a comment is related to depression or not. The project uses various machine learning models, including Logistic Regression, Support Vector Machine (SVM), and BERT, to classify comments.

Table of Contents
Features
Technologies Used
How It Works
Setup
Usage
Results
Future Improvements
Features
Scrapes comments from Reddit using the PRAW API from two subreddits: /r/depression and /r/AskReddit.
Preprocesses text data by normalizing, cleaning, and tokenizing.
Uses TF-IDF features for Logistic Regression and SVM models.
Implements BERT, a state-of-the-art NLP model, for improved classification performance.
Evaluates models with accuracy, precision, recall, F1-score, and classification reports.
Allows prediction on custom user input.
Technologies Used
Python for data processing and modeling.
PRAW for Reddit API integration.
Pandas for data manipulation.
Scikit-learn for Logistic Regression, SVM, and evaluation metrics.
Transformers (Hugging Face) for BERT implementation.
PyTorch for training and evaluating the BERT model.
How It Works
Data Collection:

Scrapes comments from /r/depression and /r/AskReddit to create a dataset of "depressed" (label 1) and "non-depressed" (label 0) comments.
Preprocessing:

Converts text to lowercase.
Removes subreddit mentions (e.g., r/depression) and user mentions (e.g., u/username).
Cleans up extra spaces for better processing.
Feature Extraction:

Converts text data into TF-IDF features for traditional machine learning models.
Tokenizes and pads text for BERT.
Model Training and Evaluation:

Logistic Regression and SVM use TF-IDF features.
BERT uses pre-trained embeddings fine-tuned on the dataset.
Models are evaluated on a validation set using various metrics.
User Input Prediction:

Allows users to input a Reddit comment to predict whether it is "Depressed" or "Non-Depressed" using the trained models.
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/reddit-sentiment-classifier.git  
cd reddit-sentiment-classifier  
Install dependencies:

bash
Copy code
pip install praw pandas scikit-learn transformers torch tqdm  
Set up Reddit API credentials:

Create a Reddit app at https://www.reddit.com/prefs/apps.
Replace the placeholders in the code (client_id, client_secret, username, password) with your credentials.
Run the script:

bash
Copy code
python reddit_sentiment_classifier.py  
Usage
Run the script to scrape Reddit comments and train the models.
After training, you can input a Reddit comment to get predictions from all three models (Logistic Regression, SVM, and BERT).
Results
Model performance is evaluated using accuracy, precision, recall, F1-score, and a detailed classification report.

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression (TF-IDF)	X.XX	X.XX	X.XX	X.XX
SVM (TF-IDF)	X.XX	X.XX	X.XX	X.XX
BERT	X.XX	X.XX	X.XX	X.XX
Replace X.XX with actual values after running the script.

Future Improvements
Expand dataset by scraping more comments from diverse subreddits.
Add hyperparameter tuning for Logistic Regression, SVM, and BERT.
Incorporate additional preprocessing steps, such as lemmatization and stop-word removal.
Explore other transformer-based models like RoBERTa or DistilBERT.
License
This project is licensed under the MIT License.
