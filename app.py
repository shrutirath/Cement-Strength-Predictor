from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import requests
app = Flask(__name__)

scalar = pickle.load(open('scalar.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def MainPage():
    return render_template("index.html")
    #return "Shruti"
@app.route('/PredictedStrength' , methods = ['POST'])
def getstrength():
    if request.method == 'POST':
        x1 = float(request.form["x1"])
        x2 = float(request.form["x2"])
        x3 = float(request.form["x3"])
        x4 = float(request.form["x4"])
        x5 = float(request.form["x5"])
        x6 = float(request.form["x6"])
        x7 = float(request.form["x7"])
        x8 = float(request.form["x8"])
        df1 = pd.DataFrame(data = [[x1,x2,x3,x4,x5,x6,x7,x8]], columns = ["Cement (component 1)(kg in a m^3 mixture)", "Blast Furnace Slag (component 2)(kg in a m^3 mixture)", "Fly Ash (component 3)(kg in a m^3 mixture)", "Water  (component 4)(kg in a m^3 mixture)", "Superplasticizer (component 5)(kg in a m^3 mixture)", "Coarse Aggregate  (component 6)(kg in a m^3 mixture)", "Fine Aggregate (component 7)(kg in a m^3 mixture)", "Age (day)"])
        for c in df1.columns:
            df1[c] += 1
            df1[c] = np.log(df1[c])
        df1_scaled = scalar.transform(df1)
        result = model.predict(df1_scaled)
        print(df1_scaled)
        print(model.predict([[540,0,0,162,2.5,1040,676,28]]))
        print(result)
        #return str(result[0])
        return render_template("result.html",y = str(result[0]))




if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3000)