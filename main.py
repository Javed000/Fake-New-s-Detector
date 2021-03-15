from flask import Flask
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')
LR = pickle.load(file)
file.close()

@app.route('/')
def hello_world():

    pred_lr=LR.predict(xv_test)
    LR.score(xv_test, y_test)
    return 'Hello, World!'  + str(classification_report(y_test, pred_lr))

if __name__ == "__main__":    
    app.run(debug=True) 
