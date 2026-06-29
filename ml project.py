from sklearn.linear_model import LinearRegression
from joblib import memory
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

# category_encoders is needed for Target Encoding
try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Target Encoding will not work.")

def main():
    print("Loading dataset...")
    file_path = "employee_data.csv"

    if not os.path.exists(file_path):
        print(f"Error: Cannot find '{file_path}'")
        return

    # Load dataset
    df = pd.read_csv(file_path,sep='\t')
    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    print(df.head())
    print(df.columns)
    print(df.shape)
   
    # Handling Missing Values
    print("Handling missing values...")

    # Artificially create one missing value in column H
    df.loc[0, 'Age'] = np.nan

    # Fill missing value using median
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']]).ravel()
    print(f"Missing values in Age: {df['Age'].isnull().sum()}")

    # Log Transformation
    print("\nApplying Log Transformation...")
    df['LogAge'] = np.log1p(df['Age'])
    print(f"Skewness after log transformation: {df['LogAge'].skew():.2f}")

    # High Cardinality
    print("\nCreating High Cardinality Feature...")
    df['Employee_ID'] = ['Employee_' + str(np.random.randint(1, 150))
    for _ in range(len(df))]

    # Target Encoding
    if TargetEncoder is not None:
        print("Applying Target Encoder...")
        # Check whether target column exists
        if 'LeaveOrNot' not in df.columns:
            print("Error: Target column 'LeaveOrNot' not found in dataset.")
            return
        print("Missing values in LeaveOrNot:", df['LeaveOrNot'].isnull().sum())

        # Remove rows where target is missing
        df = df.dropna(subset=['LeaveOrNot'])
        encoder = TargetEncoder()
        df['Employee_ID_Encoded'] = encoder.fit_transform(df[['Employee_ID']],df['LeaveOrNot'] )
        print("Target Encoding completed successfully.")
    else:
        print("Category Encoders package is not installed.")

    #feature selection    
    features_to_test=['JoiningYear','PaymentTier','ExperienceInCurrentDomain']
    x_features=df[features_to_test].fillna(0)
    x_target=df['Age']
    selector=SelectKBest(score_func=mutual_info_regression,k=2)
    selector.fit(x_features,x_target)
    winning_features=selector.get_support()
    best_features=x_features.columns[winning_features].tolist()
    print("Best features:",best_features)    

    #splitting data
    x=df[best_features]
    y=df['Age']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    print(f'training data size:{x_train.shape}')
    print(f'testing data size:{x_test.shape}')

    #training model
    model=LinearRegression()
    model.fit(x_train,y_train)
    predictions=model.predict(x_test)
    print("prediction:",predictions)

    #comparing model predcition to the actual values
    actual_age=y_test.head(3).values
    predicted_age=predictions[:3]
    
    for i in range(3):
        predicted=round(predicted_age[i])
        actual=actual_age[i]
        difference=abs(actual-predicted)

        print(f"model guessed:{predicted}")
        print(f"Real answer:{actual}")
        print(f"Differences:{difference}")

if __name__ == "__main__":
    main()