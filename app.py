from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename

from distutils.log import Log
from flask import Flask, request, render_template
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from keras.models import load_model
import h5py
from tensorflow import keras
import re
import featureExtract as fx
from sklearn.preprocessing import LabelBinarizer
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#*** Flask configuration
 
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
@app.route('/')
def index():
    return render_template('index_upload_and_show_data.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
 
        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
 
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
 
        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
 
        return render_template('index_upload_and_show_data_page2.html')
 
@app.route('/show_data')
def showData():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)
 
    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)
 
    # pandas dataframe to html table flask
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html', data_var = uploaded_df_html)

frame_size = 50
hop_size = 20

@app.route('/show_predict_result')
def showPredictionResult():
    # Unpickle classifier
        clf =keras.models.load_model("1DCNN_model_98_66.h5")
        data_file_path = session.get('uploaded_data_file_path', None)
        df= pd.read_csv(data_file_path)
        
        X, y=fx.get_frames(df, frame_size, hop_size)

        #One Hot Encode our Labels
        encoder = LabelBinarizer()
        labe = encoder.fit_transform(y)
        rounded_labels=np.argmax(labe, axis=1)

        # Get prediction
        prediction = clf.predict(X)
        rounded_predictions=np.argmax(prediction,axis=1)
       
        # predict 
        pred = clf.predict(X)
        pred = np.argmax(pred, axis=1)
        #label
        test_label = np.argmax(labe, axis=1)

        #confusioin matrix 
        cm = confusion_matrix(rounded_labels, rounded_predictions)
        prediction_output=classification_report(test_label, pred)

        report = classification_report(test_label, pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
           
        rounded_predictions = df.to_html()
               
        return render_template("show_csv_data.html", output = rounded_predictions)


# @app.route('/run_data', methods=['GET', 'POST'])
# def main():
    
#     # If a form is submitted
#     if request.method == "POST":

#         # Unpickle classifier
#         clf =keras.models.load_model("1DCNN_model_98_66.h5")
#         data_file_path = session.get('uploaded_data_file_path', None)
#         df= pd.read_csv(data_file_path)

#         X_1=[]
        
#         X, y=fx.get_frames(df, frame_size, hop_size)

#         #One Hot Encode our Labels
#         encoder = LabelBinarizer()
#         labe = encoder.fit_transform(y)
#         rounded_labels=np.argmax(labe, axis=1)

#         # Get prediction
#         prediction = clf.predict(X)
#         rounded_predictions=np.argmax(prediction,axis=1)

       
#                 # predict 
#         pred = clf.predict(X)
#         pred = np.argmax(pred, axis=1)
#         # label
#         test_label = np.argmax(labe, axis=1)

#         #confusioin matrix 
#         cm = confusion_matrix(rounded_labels, rounded_predictions)
#         prediction_output=classification_report(test_label, pred)

#         report = classification_report(test_label, pred, output_dict=True)
#         df = pd.DataFrame(report).transpose()


#         # report = pd.DataFrame(list(classification_report(test_label, pred)),
#         # index=['Precision', 'Recall', 'F1-score', 'Support']).T

#         # # Now add the 'Avg/Total' row
#         # report.loc['Avg/Total', :] = classification_report(test_label, pred,
#         #     average='weighted')
#         # report.loc['Avg/Total', 'Support'] = report['Support'].sum()


#         # prices = ['AAPL', 'ADBE', 'AMD', 'AMZN', 'CRM', 'EXPE', 'FB', 'FB', 'FB', 'FB', 'FB']

#         # df = pd.DataFrame(rounded_predictions, index = ['Row_' + str(i + 1) 
#         #                 for i in range(rounded_predictions.shape[0])],
#         #                 columns = ['Column_'+ str(i + 1) 
#         #                 for i in range(rounded_predictions.shape[1])])

#         #df = pd.DataFrame(prediction_output).transpose()
                        
#         rounded_predictions = df.to_html()
       
#         #rounded_predictions[1]
        
#     else:
#         rounded_predictions = ""
        
#     return render_template("website.html", output = rounded_predictions)


if __name__=='__main__':
    app.run(debug = True)