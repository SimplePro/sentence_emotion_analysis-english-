import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
import pandas as pd


def num_to_emotion(x):
    if x == 0: return "anger"
    elif x == 1: return "fear"
    elif x == 2: return "joy"
    elif x == 3: return "love"
    elif x == 4: return "sadness"
    elif x == 5: return "surprise"

okt = Okt()
tfv = TfidfVectorizer(tokenizer = okt.morphs, ngram_range=(1, 2), min_df=3, max_df=0.9)
loaded_model = joblib.load("./sentence_emotion_analysis_model.pkl")

data = pd.read_csv("./datasets/data.csv", index_col=0)
data.head()

features = data["sentence"]
label = data["emotion"]

train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.2, random_state=0)

tfv.fit(train_x)


def emo(s):
    sentence = tfv.transform([s])
    return num_to_emotion(loaded_model.predict(sentence)[0])