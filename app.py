from flask import Flask, render_template, request

import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("retention.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output < str(0.5):
        return render_template('retention.html', pred='The employee may quit the job.\nProbability of the employee staying in the job is {}'.format(output), bhai=" ")
    else:
        return render_template('retention.html', pred="Your employee isn't considering to quit.\n Probability of the employee staying in the job is {}".format(output), bhai=" ")


if __name__ == '__main__':
    app.run(debug=True)
