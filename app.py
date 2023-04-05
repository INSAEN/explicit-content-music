import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.base import BaseEstimator, TransformerMixin


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


st.title("Explicit Lyrics Detector")
st.write("Input your lyrics below to see if it is explicit or not")
lyrics = st.text_input("Enter Song Lyrics: ")
if st.button("Check for Explicit Lyrics"):
    st.write("The given song is ", predicter(lyrics))
    st.button("report for re-evaluation")
# text_file = open("D:/Work/ipd/ipd_streamLit/input.txt", "r")
# lyrics = text_file.read()
# text_file.close()
# predicter(lyrics)
