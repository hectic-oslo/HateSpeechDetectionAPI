#IMPORTNG LIBRARIES
from flask import Flask,render_template,request
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))
data = pd.read_csv("tweet.csv")
print(data.head())
import re

# importing the countvectoriser 
from sklearn.feature_extraction.text import CountVectorizer

# stemming 
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords

# removing stopwords
stopword=set(stopwords.words('english'))
app = Flask(__name__)
app.config['DEBUG'] = True
#ROUTES 
data["labels"] = data["class"].map({0: "Hate Speech", 
                                    1: "Offensive Language", 
                                    2: "No Hate and Offensive"})
print(data.head())

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

@app.route("/")
def home():
    title="HOME"
    return render_template('index.html',title=title)

# route for testing api
@app.route("/rishabh")
def homes():
    return {
        "name":"rishabh"
    }

@app.route('/hate-speech',methods=['POST'])   
def hatepredict():
    if request.method=='POST' :
        sample = request.form['text']
        data=clean(sample)
        data = cv.transform([sample]).toarray()
        prediction = clf.predict(data)
        return {"prediction": str(prediction[0])}
        
if __name__ == '__main__':
    app.run(debug=True,port=8000)


    
