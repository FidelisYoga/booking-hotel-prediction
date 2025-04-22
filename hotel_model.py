#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pickle
from sklearn.ensemble import RandomForestClassifier


# In[30]:


class HotelBookingModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import classification_report
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))

    def save_model(self, path="best_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

