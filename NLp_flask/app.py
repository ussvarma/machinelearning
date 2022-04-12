import nltk
from flask import Flask, request, render_template
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')  # downloading stopwords

set(stopwords.words('english'))

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')

    # convert to lowercase
    text1 = request.form['text1'].lower()

    text_final = ''.join(c for c in text1 if not c.isdigit())

    # remove stopwords
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)  # returns polarity score

    # finding out greater polarity score
    if dd['pos'] > dd['neg']:
        result = "Positive"
        polarity = dd['pos']
    elif dd['pos'] < dd['neg']:
        result = "Negative"
        polarity = dd['neg']
    else:
        result = "neutral"
        polarity = "0.5"
    print(dd['pos'])
    print(dd['neg'])

    return render_template('form.html', text2=result, text1=polarity)


if __name__ == "__main__":
    app.run(debug=True)
