import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv(r"data\iris.csv")
X = df.drop(columns = "species")
y = df.species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

#train the model
rf_model = RandomForestClassifier()
rf_model.fit(X,y)

# Checking the accuracy
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)



joblib.dump(rf_model, "rf_model.sav")
