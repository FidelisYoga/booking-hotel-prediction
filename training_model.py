#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


# In[35]:


class HotelBookingModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.encoders = {}
        
    def load_data(self):
        """Load dataset from CSV file"""
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def preprocess_data(self):
        """Handle missing values and encode categorical variables"""
        #split x y
        X = self.df.drop(columns=['booking_status', 'Booking_ID'])
        y = self.df['booking_status']
        
        #split train-test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        #handle missing values
        self._handle_missing_values()
        
        #encode categorical columns
        self._encode_categorical()
        
        # Encode target variable
        self.y_train = self.y_train.map({'Not_Canceled': 0, 'Canceled': 1})
        self.y_test = self.y_test.map({'Not_Canceled': 0, 'Canceled': 1})
        
    def _handle_missing_values(self):
        """Internal method to handle missing values"""
        #impute categorical dengan mode
        mode_meal_plan = self.X_train['type_of_meal_plan'].mode()[0]
        self.X_train['type_of_meal_plan'].fillna(mode_meal_plan, inplace=True)
        self.X_test['type_of_meal_plan'].fillna(mode_meal_plan, inplace=True)
        
        #impute numerical dengan median
        median_price = self.X_train['avg_price_per_room'].median()
        self.X_train['avg_price_per_room'].fillna(median_price, inplace=True)
        self.X_test['avg_price_per_room'].fillna(median_price, inplace=True)
        
        #impute binary dengan mode
        mode_parking = self.X_train['required_car_parking_space'].mode()[0]
        self.X_train['required_car_parking_space'].fillna(mode_parking, inplace=True)
        self.X_test['required_car_parking_space'].fillna(mode_parking, inplace=True)
    
    def _encode_categorical(self):
        """Internal method to encode categorical variables"""
        categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.X_train[col] = le.fit_transform(self.X_train[col])
            self.X_test[col] = le.transform(self.X_test[col])
            self.encoders[col] = le
    
    def train_model(self):
        """Train Random Forest model"""
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if not self.model:
            raise ValueError("Model has not been trained yet.")
            
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        return accuracy, report
    
    def save_model(self, model_path='best_model.pkl'):
        """Save trained model and encoders to disk"""
        if not self.model:
            raise ValueError("Model has not been trained yet.")
            
        # Save both model and encoders
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'encoders': self.encoders
            }, f)
        
        print(f"Model saved to {model_path}")


# In[36]:


if __name__ == "__main__":
    # Example usage
    trainer = HotelBookingModelTrainer('Dataset_B_hotel.csv')
    trainer.load_data()
    trainer.preprocess_data()
    trainer.train_model()
    trainer.evaluate_model()
    trainer.save_model()

