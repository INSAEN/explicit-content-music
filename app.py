import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.base import BaseEstimator, TransformerMixin
import paralleldots
from paralleldots.config import get_api_key
import requests
import json
from musixmatch import Musixmatch

musixmatch = Musixmatch('4a3a70d22cfd1fd8cece7b3b4e5d1679')
model_lo = joblib.load(open('model_lo.pkl', 'rb'))
model_dt = joblib.load(open('model_dt.pkl', 'rb'))
model_knn = joblib.load(open('model_knn.pkl', 'rb'))
model_rf = joblib.load(open('model_rf.pkl', 'rb'))
model_svm = joblib.load(open('model_svm.pkl', 'rb'))


def load_bad_words():
    file = open('D:/Work/ipd/ipd_streamLit/abbo.txt', 'r')
    file = list(file)
    bad_words = []
    for w in file:
        bad_words.append(re.sub(r'\n', '', w))
    return bad_words


def get_bad_words(review):
    target_word = load_bad_words()
    count = 0
    threshold = 0
    for t in target_word:
        if review.find(t) != -1:
            count += 1
    return count > threshold


def get_num_words(review):
    threshold = 0
    words = review.split(' ')
    count = len(list(words))
    return count > threshold


def get_lda_words(review):
    target_word = ['chorus', 'girl', 'money', 'baby', 'nigga', 'bitch',
                   'want', 'love', 'wanna', 'gonna', 'come', 'right', 'shit', 'feel']
    count = 0
    threshold = 0
    for t in target_word:
        if review.find(t) != -1:
            count += 1
    return count > threshold


class CustomFeats(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feat_names = set()

    def fit(self, x, y=None):
        return self

    @staticmethod
    def features(review):
        return {
            'num_word': get_num_words(review),
            'bad_word': get_bad_words(review),
            'lda_word': get_lda_words(review)
        }

    def get_feature_names(self):
        return list(self.feat_names)

    def transform(self, reviews):
        feats = []
        for review in reviews:
            f = self.features(review)
            [self.feat_names.add(k) for k in f]
            feats.append(f)
        return feats


feats = joblib.load(open('feats.pkl', 'rb'))


def classify(text, feats, model):
    test = []
    test.append(text)
    test = pd.DataFrame(test)
    re_drop = re.compile(r'\n')
    test = test.applymap(lambda x: re_drop.sub(' ', x))
    test_vecs = feats.transform(test[0])
    test_preds = model.predict(test_vecs)
    return test_preds


def predicter(lyrics):
    pred_arr = []
    pred_arr.extend(classify(lyrics, feats, model_lo))
    pred_arr.extend(classify(lyrics, feats, model_rf))
    pred_arr.extend(classify(lyrics, feats, model_knn))
    pred_arr.extend(classify(lyrics, feats, model_dt))
    pred_arr.extend(classify(lyrics, feats, model_svm))
    print(pred_arr)
    if(pred_arr.count(1) > 2):
        return "Explicit"
    else:
        return "Clean (Not Explicit)"


def get_custom_classifier(text, category_list):
    api_key = paralleldots.get_api_key()
    if type(category_list) == list:
        category_list = json.dumps(category_list)
    response = requests.post(" https://apis.paralleldots.com/v4/custom_classifier", data={
                             "api_key": api_key, "text": text, "category_list": category_list}).text
    response = json.loads(response)
    return response


def context(lyrics):
    keywords = ""
    paralleldots.set_api_key("9isljmYvlsCHGsnGTbxHgXOOZIj0IZHeWcDxsdAnpDY")
    paralleldots.get_api_key()
    category_list = ["sexual", "homophobic", "sexist", "racist",
                     "suicidal", "hate-speech", "violent", "substance-abuse"]
    response = get_custom_classifier(lyrics, category_list)
    for i in range(len(response['taxonomy'])):
        if(response['taxonomy'][i]['confidence_score'] >= 0.7):
            # print(response['taxonomy'][i]['tag'])
            keywords = keywords+response['taxonomy'][i]['tag']+", "
            print(keywords)
    return keywords


st.title("Explicit Lyrics Detector")
st.write("Input your lyrics below to see if it is explicit or not")
lyrics = st.text_input("Enter Song Lyrics: ")
if st.button("Check for Explicit Lyrics"):
    st.write("The given song is ", predicter(lyrics))
    with st.spinner('Analysing ...'):
        if(context(lyrics)!=""):
            st.write("The given song has phrases that might be considered", context(lyrics))
    st.button("report for re-evaluation")
    
st.write("OR ")
st.write("Input Song Name and Artist")
sname=st.text_input("Enter Song Name: ")
aname=st.text_input("Enter Artist Name: ")

if st.button("Check"):
    rucha= musixmatch.matcher_lyrics_get(sname, aname)
    lyric_from_data=rucha['message']['body']['lyrics']['lyrics_body']
    st.write("The given song is ", predicter(lyric_from_data))
    with st.spinner('Analysing ...'):
        if(context(lyric_from_data)!=""):
            st.write("The given song has phrases that might be considered", context(lyric_from_data)) 
    st.button("report for re-evaluation")
# text_file = open("D:/Work/ipd/ipd_streamLit/input.txt", "r")
# lyrics = text_file.read()
# text_file.close()
# predicter(lyrics)
