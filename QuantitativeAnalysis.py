## working directory
import  os
os.chdir('/Users/valerieurrutia/PycharmProjects/ContentAnalysis/venv/bin')

## this is the NLP package for content analysis
import nltk
print(nltk.__version__)
##run this code if it's the first time downloading NLP package
# nltk.download()

import pandas as pd

## download the csv file
print("Opening data file...")
yelp_df = pd.read_csv("yelp_full_data.csv")
print(yelp_df.columns) # tells you the name of the columns in the file
yelp_df.info() # tells you what type of values are in a column (i.e. integer, string, etc.)

## filter dataframe to restaurants with 3 stars
print("Processing 3 star data...")
is_three = yelp_df['rating']==3
print(is_three.head(10))

three_stars = yelp_df[is_three]
print(three_stars.shape)
print(three_stars)

## filter dataframe to restaurants with 3.5 stars
print("Processing 3.5 star data...")
is_three_half = yelp_df['rating']==3.5
print(is_three_half.head(10))

three_half_stars = yelp_df[is_three_half]
print(three_half_stars.shape)
print(three_half_stars)

## filter dataframe to restaurants with 4 stars
print("Processing 4 star data...")
is_four = yelp_df['rating']==4
print(is_four.head(10))

four_stars = yelp_df[is_four]
print(four_stars.shape)
print(four_stars)

## filter dataframe to restaurants with 4.5 stars
print("Processing 4.5 star data...")
is_four_half = yelp_df['rating'] == 4.5
print(is_four_half.head(10))

four_half_stars = yelp_df[is_four_half]
print(four_half_stars.shape)
print(four_half_stars)

## filter dataframe to restaurants with 5 stars
print("Processing 5 star data...")
is_five = yelp_df['rating']==5
print(is_five.head(10))

five_stars = yelp_df[is_five]
print(five_stars.shape)
print(five_stars)

##cleaning up list - 3 stars
three_list = []  # list containing all words of all texts
for elmnt in three_stars['review_text']:  # loop over lists in df
    three_list += elmnt  # append elements of lists to full list (three_list)

## make temporary Series to count
val_counts_three = pd.Series(three_list).value_counts()
print(val_counts_three)

##vector process - 3 stars
print('Starting vector process...')
threestarreviews = three_stars['review_text'].values
threeVec = [nltk.word_tokenize(review_text) for review_text in threestarreviews]
print(threeVec)

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=100000)
model.wv.most_similar('man')
model.wv('man')