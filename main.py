import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# This is a small prompt introducing us to the world of Machine Learning. For this demonstration, 
# we use the RandomForestClassifier, a robust algorithm that uses the power of decision trees,
# combining multiple trees to improve accuracy and prevent overfitting. The data will be preprocessed, features engineered, 
# and the model will be trained to make predictions on new data. 

# Made by Fares Laadjel, Achraf Bayi, Wiame Kotbi and Mohammed Amine Dakli
# CodeML 2024, October 5th
#----------------------------------------------------------------------------------------------------------------------------------------------
#FUNCTIONS USED

# Converting non-numeric data
def convert_to_numeric(df):
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column], errors='coerce') # Errors will be set to NaN
        except:
            print(f"Couldn't convert column {column} to numeric.")
    return df

# Adding time attributes for better approximations 
def preprocess_dates(df):
    
    # Converting dates into Pandas datetime objects
    df['purchase_datetime'] = pd.to_datetime(df['purchase_datetime'])
    df['flight_departure_datetime'] = pd.to_datetime(df['flight_departure_datetime'])
    
    # Extracting and adding the new attributes
    df['purchase_day_of_week'] = df['purchase_datetime'].dt.dayofweek  # Monday=0, Sunday=6
    df['purchase_hour'] = df['purchase_datetime'].dt.hour  # Hour of purchase
    df['flight_day_of_week'] = df['flight_departure_datetime'].dt.dayofweek  # Monday=0, Sunday=6
    df['flight_hour'] = df['flight_departure_datetime'].dt.hour  # Hour of takeoff
    
    # Calculating and adding leadtime attribute in hours
    df['lead_time'] = (df['flight_departure_datetime'] - df['purchase_datetime']).dt.total_seconds() / 3600  

    # Removing the all ready processed attributes
    df = df.drop(['purchase_datetime', 'flight_departure_datetime'], axis=1)
    
    return df

def train_model(training_file):
    try:
        data = pd.read_csv(training_file)
        print("Training data loaded successfully.")
        
        data_cleaned = preprocess_dates(data)
        print("Time was successfully processed")

        data_cleaned = data_cleaned.dropna(subset=['choice'])
        print("Dropped rows with missing 'choice' values.")

        # One-hot encoding was done to put more emphasis on these values
        data_cleaned = pd.get_dummies(data_cleaned, columns=['od', 'trip_type', 'branded_fare'])
        print("One-hot encoding done.")

        # Convert string responses with numerical responses
        choice_mapping = {'advs': 0, 'pref': 1, 'nochoice': 2}
        data_cleaned['choice'] = data_cleaned['choice'].map(choice_mapping).astype(int)
        print("Mapped 'choice' to numeric values.")

        data_cleaned = convert_to_numeric(data_cleaned)
        print("Converted data to numeric where possible.")

        data_cleaned = data_cleaned.dropna()
        print("Dropped rows with NaN values after conversion.")

        
        X = data_cleaned.drop('choice', axis=1) # This will be the features used for training
        y = data_cleaned['choice'] # This will be the target (what we want to predict)
        print(f"Training data shape: {X.shape}")

        # Train_test_split returns a tuple of 4 elements
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Train and test split done. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # RandomForestClassifier was finally chosen for better results and less overfitting
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Model training completed.")

        return model, X_train.columns
    
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None

def predict_choices(model, new_data_file, output_file, train_columns):
    try:
        new_data = pd.read_csv(new_data_file)
        print("New data loaded successfully.")

        new_data_cleaned = preprocess_dates(new_data)

        # Same as before
        new_data_cleaned = pd.get_dummies(new_data_cleaned, columns=['od', 'trip_type', 'branded_fare'])
        print("One-hot encoding for new data done.")

        new_data_cleaned = convert_to_numeric(new_data_cleaned)
        print("Converted new data to numeric where possible.")

        # Reindex is used to make sure no additional columns were added and matches the training columns exactly
        X_baseline = new_data_cleaned.reindex(columns=train_columns, fill_value=0)
        print(f"New data aligned with training columns. Shape: {X_baseline.shape}")

        predictions = model.predict(X_baseline)
        print("Prediction completed.")

        # Converting the predictions back to string
        reverse_choice_mapping = {0: 'advs', 1: 'pref', 2: 'nochoice'}
        new_data['choice'] = [reverse_choice_mapping[pred] for pred in predictions] # Replacing the hole column with a list of all predictions
        print("Mapped predictions back to strings.")

        # Saving in a file
        new_data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    except Exception as e:
        print(f"Error during prediction: {e}")


#----------------------------------------------------------------------------------------------------------------------------------------------
#EXECUTING THE FUNCTIONS

trained_model, train_columns = train_model('participant_data.csv')
if trained_model:
    predict_choices(trained_model, 'baseline.csv', 'output_with_predictions.csv', train_columns)
else:
    print("Model training failed. Check the error messages above.")
