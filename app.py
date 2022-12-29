#from crypt import methods
from distutils.log import debug
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.models import load_model
from flask_mysqldb import MySQL
import numpy as np
import os
import random
import json
from json import JSONEncoder

app = Flask(__name__)
CORS(app)

#Data base connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'nuclius_master'

mysql = MySQL(app)

#sequance generator function
def sequesnceGenerator(arr,n):
    i=0
    arr1 = []
    temp = []
    while(i < len(arr)):
        if(i%n == 0 and i != 0):
            arr1.append(temp)
            temp = [arr[i]]

        else:
            temp.append(arr[i])
        i+=1
    #arr1.append(temp)
    return arr1

#json serialization class
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#main route
@app.route("/mlAdapter" , methods = ['GET'])
def mlAdapter():

    #getting n data from main db
    cursor = mysql.connection.cursor()
    cursor.execute(''' SELECT power,time_stamp from dcsub001 ''')
    #mysql.connection.commit()
    #cursor.close()
    row_headers=[x[0] for x in cursor.description]
    row = cursor.fetchall()
    json_data=[]
    for result in row:
        json_data.append(dict(zip(row_headers,result)))
    
    #nessory arrays
    data_out_power= []
    data_out_time=[]

    #formating data
    for i in range(0,len(json_data)):
        data_out_power.append(json_data[i]['power'])
        data_out_time.append(json_data[i]['time_stamp'])

    #dataset list
    total_set ={"power":data_out_power,"time_stamp":data_out_time}
    #data fareme making
    df = pd.DataFrame(data=total_set)
    df = df.set_index('time_stamp')
    df.index = pd.to_datetime(df.index)
    #resampling 
    df = df.resample('10s').mean()
    df = df.dropna()
    #seq value
    seq_val = 100

    X = df["power"]
    xx = np.array(X)
    X_TRAIN=np.array(sequesnceGenerator(xx,seq_val))
    
    #model loading
    model_name = "NILM_BASE_MODEL_v_1_3.h5"
    loded_model = load_model(model_name)

    #prediction
    predicted = loded_model(X_TRAIN)
    array = predicted.numpy()
    result = array.flatten()
    print(result)

    #JSON Conversion
    encodedNumpyData = json.dumps(array, cls=NumpyArrayEncoder)

    return jsonify({
    "statusCode": "200",
    "statusDesc": "Success",
    "nilmPrediction": encodedNumpyData
    })

app.run(host="0.0.0.0",port=5080,debug = True)
# app.run(debug = True)