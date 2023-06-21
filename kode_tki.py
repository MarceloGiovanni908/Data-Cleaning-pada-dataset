# -*- coding: utf-8 -*-
"""Text Mining

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bKWMDFYkbqcB7Uap923Jy66FT8BpM9vS
"""

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re

data_amazon = pd.read_csv('/content/drive/MyDrive/Dataset/sentiment labelled sentences/amazon_cells_labelled.csv', names=['text','label','1','2','3','4'], delimiter=None)
data_amazon.head()

data_imdb = pd.read_csv('/content/drive/MyDrive/Dataset/sentiment labelled sentences/imdb_labelled.csv', names=['text','label','1','2','3','4','5','6'], delimiter=None)
data_imdb.head()

data_yelp = pd.read_csv('/content/drive/MyDrive/Dataset/sentiment labelled sentences/yelp_labelled.csv', names=['text','label','1','2','3','4'], delimiter=None)
data_yelp.head()

combined_data = pd.DataFrame({
    'Amazon': data_amazon.apply(lambda x: ' '.join([str(val) for val in x.dropna()]), axis=1),
    'IMDB': data_imdb.apply(lambda x: ' '.join([str(val) for val in x.dropna()]), axis=1),
    'Yelp': data_yelp.apply(lambda x: ' '.join([str(val) for val in x.dropna()]), axis=1)
})

combined_data.columns = ['Amazon', 'IMDB', 'Yelp']

combined_data.head()

# Tokenisasi teks
tokenized_data = combined_data.applymap(lambda x: word_tokenize(str(x)))
#print(tokenized_data)
tokenized_data.head()

# Menghapus stop words
stop_words = set(stopwords.words('english'))
filtered_data1 = tokenized_data.applymap(lambda x: [word for word in x if word.lower() not in stop_words])
filtered_data1.head()

# Lemmatisasi teks
lemmatizer = WordNetLemmatizer()
preprocessed_data = filtered_data1.applymap(lambda x: [lemmatizer.lemmatize(word) for word in x])
preprocessed_data.head()

# Menghapus tanda baca
punctuation = set(string.punctuation)
filtered_data2 = preprocessed_data.applymap(lambda x: [word for word in x if word not in punctuation and len(word) > 1])
filtered_data2.head()

# Menghapus kata yang tidak perlu
unnecessary_words = set(['the', 'a', 'an'])
filtered_data3 = filtered_data2.applymap(lambda x: [word for word in x if word.lower() not in unnecessary_words])
filtered_data3.head()

# Menghapus elemen list kosong
filtered_data4 = filtered_data3.applymap(lambda x: list(filter(None, x)))
filtered_data4.head()

# Menggabungkan kata-kata menjadi kalimat
processed_sentences = filtered_data4.applymap(lambda x: ' '.join(x))
processed_sentences.head()

# Simpan hasil preprocessing ke dalam file CSV baru
processed_sentences.to_csv('/content/drive/MyDrive/Dataset/sentiment labelled sentences/preprocessing_combined_data.csv', index=False)