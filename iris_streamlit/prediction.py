import joblib

def predict(data):
    model = joblib.load("rf_model.sav")
    answer = model.predict(data)
    return answer

