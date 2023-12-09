
from flask import Flask,render_template,request
import numpy as np
import pickle

with open('air_quality.pkl','rb') as f:
    model = pickle.load(f)

with open('scaler.pkl','rb') as m :
    scaler = pickle.load(m)

with open('minmax.pkl','rb') as v:
    ms = pickle.load(v)
#create an object instance
app = Flask(__name__)

@app.route('/sai')
def me():
    return "This is sai"

    

@app.route('/')
def new():
    return render_template("index.html")
@app.route('/predict',methods=['POST'])
def predict():
    PT08_S1 = int(request.form['PT08_S1'])
    NMHC = int(request.form["NMHC"])
    C6H6 = int(request.form["C6H6"])
    PT08_S2 = int(request.form["PT08_S2"])
    NOx = int(request.form["NOx"])
    PT08_S3 = int(request.form["PT08_S3"])
    NO2 = int(request.form["NO2"])
    PT08_S4 = int(request.form["PT08_S4"])
    PT08_S5 = int(request.form["PT08_S5"])
    T = int(request.form["T"])
    RH = int(request.form["RH"])
    AH = int(request.form["AH"])
    
    user_data = np.array([[PT08_S1, NMHC, C6H6, PT08_S2, NOx, PT08_S3,NO2, PT08_S4, PT08_S5,  T, RH, AH]])

    user_data = ms.transform(user_data)
    user_data = scaler.transform(user_data)

    # Make predictions
    co_prediction = model.predict(user_data)
    print('**************************************',co_prediction)
    # Display the predicted CO concentration
    concentration = co_prediction[0]

    aqi_co = (concentration/10)*100    
    if int(aqi_co) in range(0,33):
        d=f'Very Good {int(aqi_co)}'
    elif int(aqi_co) in range(34,66):
        d=f'Good {int(aqi_co)}'
    elif int(aqi_co) in range(67,99):
        d=f'Fair {int(aqi_co)}'
    else:
        d=f'Unhealthy {int(aqi_co)}'
    #print(f"Predicted CO Concentration: {rounded_concentration}")
    return render_template('index.html',Ard=d)
app.run(debug=True,use_reloader=True)