import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import string
from string import punctuation 
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.probability import FreqDist
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, ConfusionMatrixDisplay

train_data = pd.read_csv('path',encoding='latin1')
test_data = pd.read_csv('/path',encoding='latin1')
df = pd.concat([train_data,test_data])
#print(df.head())
df.shape
df.info()

def remove_unnecessary_characters(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text
df['clean_text'] = df['text'].apply(remove_unnecessary_characters)

def tokenize_text(text):
    try:
        text = str(text)
        tokens = word_tokenize(text)
        return tokens
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return []
df['tokens'] = df['text'].apply(tokenize_text)

def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    else:
        text = str(text)
    return text
df['normalized_text'] = df['text'].apply(normalize_text)

def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()        
        filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
        filtered_text = ' '.join(filtered_words)
    else:
        filtered_text = ''
    return filtered_text
df['text_without_stopwords'] = df['text'].apply(remove_stopwords)

df.dropna(inplace=True)
df['sentiment'].value_counts(normalize=True).plot(kind='bar')
#plt.show()
df['sentiment'].value_counts()

df['sentiment_code'] = df['sentiment'].astype('category').cat.codes
sentiment_distribution = df['sentiment_code'].value_counts(normalize=True)
#sentiment_distribution.plot(kind='bar')
#plt.show()

stuff_to_be_removed = list(stopwords.words('english'))+list(punctuation)
stemmer = LancasterStemmer()
corpus = df['text'].tolist()
#print(len(corpus))
#print(corpus[0])

word_freq = FreqDist(word_tokenize(' '.join(df['sentiment'])))
#plt.figure(figsize=(10, 6))
#word_freq.plot(20, cumulative=False)
#plt.title('Word Frequency Distribution')
#plt.xlabel('Word')
#plt.ylabel('Frequency')
#plt.show()

final_corpus = df['text'].astype(str).tolist()
data_eda = pd.DataFrame()
data_eda['text'] = final_corpus
data_eda['sentiment'] = df["sentiment"].values
data_eda.head()

df['Time of Tweet'] = df['Time of Tweet'].astype('category').cat.codes
df['Country'] = df['Country'].astype('category').cat.codes
df['Age of User']=df['Age of User'].replace({'0-20':18,'21-30':25,'31-45':38,'46-60':53,'60-70':65,'70-100':80})
df=df.drop(columns=['textID','Time of Tweet', 'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'])

def wp(text):
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Assuming df is defined somewhere in your code
df['selected_text'] = df["selected_text"].apply(wp)
X=df['selected_text']
y= df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorization = TfidfVectorizer()
XV_train = vectorization.fit_transform(X_train)
XV_test = vectorization.transform(X_test)
lr = LogisticRegression(n_jobs=-1)
lr.fit(XV_train,y_train)

pred_lr= lr.predict(XV_test)
score_lr = accuracy_score(y_test, pred_lr)
#print(score_lr)

#def output_label(n):
    #if n == 0:
        #return "The Text Sentement is Negative"
    #elif n == 1:
        #return "The Text Sentement is Neutral"
    #elif n == 2:
        #return "The Text Sentement is Positive"


def wp(text):
    # Define what you want wp to do, for example:#
    return text.upper()

def manual_testing():
    while True:
        news = input("Enter text to analyze sentiment (or type 'exit' to quit): ")
        if news.lower() == 'exit':  # Check if the user wants to exit
            print("Exiting sentiment analysis.")
            break
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wp)  # Assuming wp is your preprocessing function
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        pred_lr = lr.predict(new_xv_test)
        print((pred_lr[0]))   
    #return pred_lr[0]

#text = "I am happy"
#print(manual_testing(text))

manual_testing()
