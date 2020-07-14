import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)



#load the trained model
cali_model = pickle.load(open('ca_model.pkl','rb'))


@app.route('/predict_ca',methods=['POST'])
def predict_ca():

	form_input = request.get_json(force=True)

	input_list = [np.array(form_input['test'])]

	prediction = cali_model.predict(input_list)
	
	price = round(prediction[0],2)

	return jsonify(price)



if __name__ == '__main__':
    app.run(debug=True)
