import streamlit as st
import pickle
import numpy as np
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import distance
import re
from bs4 import BeautifulSoup

rf = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))

def preprocess_text(q):
    q=str(q).lower().strip()
    #replace certain special characters with their string equivalent
    q = q.replace("%", " percent")
    q = q.replace("$", " dollar ")
    q = q.replace("₹", " rupee ")
    q = q.replace("€", " euro ")
    q = q.replace("@", " at ")
#THE patter '[math] appears around 900 times in whole dataset. It is used to represent mathematical expressions. We will replace it with a space.
    q = q.replace("[math]", " ")
    #replace some numbers with string equivalent
    q = q.replace(',000,000,000','b')
    q = q.replace(',000,000','m')
    q = q.replace(',000','k')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    #decontracting words

    contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    #remove html tags
    q = BeautifulSoup(q)
    q = q.get_text()

    #remove punctuations
    pattern = re.compile(r'[^\w\s]')
    q = re.sub(pattern, '', q).strip()
    return q
def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)
def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))


# advanced features



def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']
    SAFE_DIV = 0.0001  # to prevent division by zero
    STOP_WORDS = (stopwords.words("english"))
    token_features = [0.0] * 8

    # Converting sentences into tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # GET the non stop words in the questions:
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # GET the stop words in the questions:
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non stop words in the questions:
    common_word_count = len(q1_words & q2_words)

    # get the common stop words in the questions:
    common_stop_count = len(q1_stops & q2_stops)

    # Get the common tokens in the questions:
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words),
                                                 len(q2_words)) + SAFE_DIV)  # common non stop words divided by the min number of non stop words in a question
    token_features[1] = common_word_count / (max(len(q1_words),
                                                 len(q2_words)) + SAFE_DIV)  # common non stop words divided by the max number of non stop words in a question
    token_features[2] = common_stop_count / (min(len(q1_stops),
                                                 len(q2_stops)) + SAFE_DIV)  # common stop words divided by the min number of stop words in a question
    token_features[3] = common_stop_count / (max(len(q1_stops),
                                                 len(q2_stops)) + SAFE_DIV)  # common stop words divided by the max number of stop words in a question
    token_features[4] = common_token_count / (min(len(q1_tokens),
                                                  len(q2_tokens)) + SAFE_DIV)  # common tokens divided by the min number of tokens in a question
    token_features[5] = common_token_count / (max(len(q1_tokens),
                                                  len(q2_tokens)) + SAFE_DIV)  # common tokens divided by the max number of tokens in a question

    # Last word of both is similar or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both is similar or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features




def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']
    length_features = [0.0] * 3
    # Converting the sentence into tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # Absolute length features

    length_features[0] = abs(len(q1) - len(q2))  # absolute length difference between the questions

    # Aveage token length of both questions

    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2  # average token length of both questions

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1),
                                             len(q2)) + 1) if strs else 0  # length of longest common substring divided by the min length of the questions

    return length_features
def fetch_fuzzy_features(row):
    q1 = row['question1']
    q2 = row['question2']
    fuzzy_features = [0.0]*4
    fuzzy_features[0] = fuzz.QRatio(q1, q2) / 100 #QRatio is a simple Levenshtein distance similarity measure.
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2) / 100 #Partial ratio is a measure of the similarity of the shorter string to substrings of the longer string.
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2) / 100 #Token sort ratio is a measure of similarity that ignores word order.
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2) / 100 #Token set ratio is a measure of similarity that ignores duplicate words and word order.
    return fuzzy_features

def query_point_creator(q1,q2):

    input_query = []

    q1 = preprocess_text(q1)
    q2 = preprocess_text(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split()))
    input_query.append(len(q2.split()))

    row = {"question1": q1, "question2": q2}

    cw = common_words(row)
    tw = total_words(row)

    input_query.append(cw)
    input_query.append(tw)
    input_query.append(round(cw/(tw+0.0001),2))

    token_features = fetch_token_features(row)
    input_query.extend(token_features)

    length_features = fetch_length_features(row)
    input_query.extend(length_features)

    fuzzy_features = fetch_fuzzy_features(row)
    input_query.extend(fuzzy_features)

    q1_tfidf = cv.transform([q1]).toarray()[0]
    q2_tfidf = cv.transform([q2]).toarray()[0]

    return np.hstack((np.array(input_query),q1_tfidf,q2_tfidf)).reshape(1,-1)
st.title("Quora Duplicate Question Detector")

q1 = st.text_area("Enter Question 1")
q2 = st.text_area("Enter Question 2")
if st.button("Check Similarity"):

    x = query_point_creator(q1,q2)

    result = rf.predict(x)[0]

    if result == 1:
        st.success("Duplicate Question")
    else:
        st.error("Not Duplicate")