import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('heart_disease_model.sav','rb'))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/signup/')
def signup():
    return render_template("signup.html")


@app.route('/diabetes/')
def diabetes():
    return render_template("diabetes.html")


@app.route('/home/')
def home():
    return render_template("home.html")

@app.route('/heart/')
def heart():
    return render_template("heart.html")


# diabetes_dataset = pd.read_csv('./diabetes.csv')
# diabetes_dataset.describe()
# diabetes_dataset['Outcome'].value_counts()
# diabetes_dataset.groupby('Outcome').mean()
# X = diabetes_dataset.drop(columns = 'Outcome',axis=1)
# Y = diabetes_dataset['Outcome']
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
# classifier = svm.SVC(kernel='linear')
# classifier.fit(X_train, Y_train)

filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        final = ''
        if my_prediction == 1:
            final = "Indivdual should contact his/her doctor" 
        else:
            final = "Patient is Negative"
        return render_template('diabetes.html', prediction=final)
    

    
    
@app.route('/result_heart', methods=['POST', 'GET'])
def result_heart():
    if request.method == 'POST':
        print(request.form)
        age2 = request.form['age2']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
    # input_features = [float(x) for x in request.form.values()]
    features_value = np.array([[age2,sex,cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # features_name = ["age", "sex_0", "  sex_1", "cp_0", "cp_1", "cp_2", "cp_3", "trestbps", "chol", "thalach", "oldpeak", "  fbs_0", "  fbs_1",
    #                  "restecg_0", "restecg_1", "restecg_2", "exang_0", "exang_1", "slope_1", "slope_2", "slope_3", "ca_0", "ca_1", "ca_2", "ca_3", "thal_1",
    #                  "thal_2", "thal_3"]
    
    # print(len(features_value))
    # print(len(features_name))

    df = pd.DataFrame(features_value)
    output = model.predict(df)

    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    return render_template('heart.html', prediction_text='Patient has {}'.format(res_val))
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)
