## working directory
import  os
os.chdir('/Users/valerieurrutia/PycharmProjects/ContentAnalysis/venv/bin')


##starting content analysis
print("Starting content analysis...")
import bs4 as bs
import re
from string import ascii_lowercase
import gensim, re, io, pymongo, itertools, nltk, snowballstemmer
from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

## removing stop words
stop_words = set(stopwords.words('english'))
file1 = open('yelp_full_data.csv')
line = file1.read()# Use this to read file content as a stream:
## run once
words = line.split()
for r in words:
    if not r in stop_words:
        appendFile = open('yelp_full_data.csv','a')
        appendFile.write(" "+r)
        appendFile.close()
## split() returns list of all the words in the string
split_it = file1.split()

## Pass the split_it list to instance of Counter class.
Counter = Counter(split_it)

## most_common() produces k frequently encountered
## input values and their respective counts.
most_occur = Counter.most_common(4)

print(most_occur)