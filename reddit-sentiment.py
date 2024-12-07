import praw
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm

# Reddit API credentials
reddit = praw.Reddit(user_agent=True, client_id="ULFm996McjuOA5FTdk632A",
                     client_secret="ipBsPR72JXrv-6tT4E-BIEnEnAMO4Q", username='OkRevenue9975',
                     password='anyareddit@7')

# Function to preprocess comments
def preprocess_text(comment):
    # Convert to lowercase
    comment = comment.lower()
    
    # Remove subreddit mentions (e.g., r/depression)
    comment = re.sub(r'r\/\w+', '', comment)
    
    # Remove user mentions (e.g., u/username)
    comment = re.sub(r'u\/\w+', '', comment)
    
    # Remove extra whitespace
    comment = ' '.join(comment.split())

    return comment

# Function to scrape "Top" and "Hot" comments from a subreddit
def scrape_reddit_comments(subreddit, num_comments=1000, sort_type='top'):
    comments = []
    subreddit = reddit.subreddit(subreddit)

    if sort_type == 'top':
        submissions = subreddit.top(limit=num_comments)
    elif sort_type == 'hot':
        submissions = subreddit.hot(limit=num_comments)
    else:
        raise ValueError("Invalid sort_type. Use 'top' or 'hot'.")

    for submission in submissions:
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            processed_comment = preprocess_text(comment.body)
            comments.append(processed_comment)

    return comments

# Scrape "Top" comments from /r/depression
top_depressed_comments = scrape_reddit_comments('depression', sort_type='top')

# Scrape "Hot" comments from /r/depression
hot_depressed_comments = scrape_reddit_comments('depression', sort_type='hot')

# Scrape "Top" comments from /r/AskReddit
top_non_depressed_comments = scrape_reddit_comments('AskReddit', sort_type='top')

# Scrape "Hot" comments from /r/AskReddit
hot_non_depressed_comments = scrape_reddit_comments('AskReddit', sort_type='hot')

# Create DataFrames with comments and labels
depressed_df_top = pd.DataFrame({'text': top_depressed_comments, 'label': 1})
depressed_df_hot = pd.DataFrame({'text': hot_depressed_comments, 'label': 1})
non_depressed_df_top = pd.DataFrame({'text': top_non_depressed_comments, 'label': 0})
non_depressed_df_hot = pd.DataFrame({'text': hot_non_depressed_comments, 'label': 0})

# Concatenate the DataFrames
df = pd.concat([depressed_df_top, depressed_df_hot, non_depressed_df_top, non_depressed_df_hot], ignore_index=True)
print(df.head())

# Shuffle and split the data
train_data, test_data, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)
val_data, test_data, val_labels, test_labels = train_test_split(
    test_data, test_labels, test_size=0.5, random_state=42
)

# Convert comments to TF-IDF features with norm L2
vectorizer = TfidfVectorizer(norm='l2', min_df=2)
train_features_tfidf = vectorizer.fit_transform(train_data)
val_features_tfidf = vectorizer.transform(val_data)
test_features_tfidf = vectorizer.transform(test_data)

# Logistic Regression model
logistic_model = LogisticRegression(C=1.2)
logistic_model.fit(train_features_tfidf, train_labels)

# Predictions on validation set
val_predictions_logistic = logistic_model.predict(val_features_tfidf)

# SVM model
svm_model = LinearSVC(C=0.2)
svm_model.fit(train_features_tfidf, train_labels)

# Predictions on validation set
val_predictions_svm = svm_model.predict(val_features_tfidf)

# Evaluate models
def evaluate_model(predictions, labels, model_name):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print(f"{model_name} (TF-IDF) Evaluation:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Classification Report:\n", classification_report(labels, predictions))

# Evaluate Logistic Regression model
evaluate_model(val_predictions_logistic, val_labels, "Logistic Regression Model")

# Evaluate SVM model
evaluate_model(val_predictions_svm, val_labels, "SVM Model")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize and pad the input data
train_tokenized = tokenizer(list(train_data), padding=True, truncation=True, return_tensors='pt', max_length=128)
val_tokenized = tokenizer(list(val_data), padding=True, truncation=True, return_tensors='pt', max_length=128)
test_tokenized = tokenizer(list(test_data), padding=True, truncation=True, return_tensors='pt', max_length=128)

# Convert labels to PyTorch tensors
train_labels_tensor = torch.tensor(train_labels.values)
val_labels_tensor = torch.tensor(val_labels.values)
test_labels_tensor = torch.tensor(test_labels.values)

# Create DataLoader for BERT input
train_dataset = TensorDataset(train_tokenized['input_ids'], train_tokenized['attention_mask'], train_labels_tensor)
val_dataset = TensorDataset(val_tokenized['input_ids'], val_tokenized['attention_mask'], val_labels_tensor)
test_dataset = TensorDataset(test_tokenized['input_ids'], test_tokenized['attention_mask'], test_labels_tensor)

# Set up DataLoader
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)
loss_function = torch.nn.CrossEntropyLoss()

# Training loop for BERT model
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

for epoch in range(epochs):
    bert_model.train()
    total_loss = 0
    tqdm_dataloader = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
    for input_ids, attention_mask, labels in tqdm_dataloader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        tqdm_dataloader.set_postfix({'loss': total_loss / len(train_dataloader)})

# Validation loop for BERT model
bert_model.eval()
val_predictions_bert = []
val_labels_bert = []

with torch.no_grad():
    for input_ids, attention_mask, labels in tqdm(val_dataloader, desc='Validation'):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        val_predictions_bert.extend(predictions.cpu().numpy())
        val_labels_bert.extend(labels.cpu().numpy())

# Evaluate BERT model
evaluate_model(val_predictions_bert, val_labels_bert, "BERT Model")
# Function to preprocess user input
def preprocess_user_input(user_input):
    processed_input = preprocess_text(user_input)
    return processed_input

# Get user input
user_input = input("Enter a Reddit comment: ")

# Preprocess user input
processed_input = preprocess_user_input(user_input)

# Transform input using TF-IDF Vectorizer
input_features_tfidf = vectorizer.transform([processed_input])

# Predict with Logistic Regression
logistic_prediction = logistic_model.predict(input_features_tfidf)
print("Logistic Regression Prediction:", "Depressed" if logistic_prediction == 1 else "Non-Depressed")

# Predict with SVM
svm_prediction = svm_model.predict(input_features_tfidf)
print("SVM Prediction:", "Depressed" if svm_prediction == 1 else "Non-Depressed")

# Tokenize and pad the input for BERT
input_tokenized = tokenizer([processed_input], padding=True, truncation=True, return_tensors='pt', max_length=128)

# Predict with BERT
input_ids = input_tokenized['input_ids'].to(device)
attention_mask = input_tokenized['attention_mask'].to(device)
bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
logits = bert_outputs.logits
bert_prediction = torch.argmax(logits, dim=1).item()
print("BERT Prediction:", "Depressed" if bert_prediction == 1 else "Non-Depressed")
