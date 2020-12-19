from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger




app=Flask(__name__)
Swagger(app)

rf = pickle.load(open('rf_model.pkl','rb'))
res = pickle.load(open('res_vect.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        vect =  res.transform(data)
        my_prediction = rf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)
    