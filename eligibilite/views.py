import joblib
import pandas as pd
from django.http import JsonResponse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

model_path = r'C:\Users\lahbi\OneDrive - ESPRIT\Bureau\pfe_final\elg\Nouveau dossier (2)\Census-Data-Income-Prediction-Using-TensorFlow\random_forest_model.joblib'
loaded_model = joblib.load(model_path)
print(loaded_model)
columns_used_in_training  = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'income_bracket',
             'income', 'workclass_ Federal-gov', 'workclass_ Local-gov', 'workclass_ Never-worked',
               'workclass_ Private', 'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc',
                 'workclass_ State-gov', 'workclass_ Without-pay', 'education_ 11th', 'education_ 12th', 
                 'education_ 1st-4th', 'education_ 5th-6th', 'education_ 7th-8th', 'education_ 9th', 'education_ Assoc-acdm', 
                 'education_ Assoc-voc', 'education_ Bachelors', 'education_ Doctorate', 'education_ HS-grad', 'education_ Masters',
                   'education_ Preschool', 'education_ Prof-school', 'education_ Some-college', 'marital_status_ Married-AF-spouse',
                     'marital_status_ Married-civ-spouse', 'marital_status_ Married-spouse-absent', 
                     'marital_status_ Never-married', 'marital_status_ Separated', 'marital_status_ Widowed',
                       'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair',
                         'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners',
                           'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv',
                             'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales',
                               'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Not-in-family',
                                 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried',
                                   'relationship_ Wife', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White',
                                     'gender_ Male', 'native_country_ Cambodia', 'native_country_ Canada', 'native_country_ China',
                                       'native_country_ Columbia', 'native_country_ Cuba', 'native_country_ Dominican-Republic', 
                                       'native_country_ Ecuador', 'native_country_ El-Salvador', 'native_country_ England',
                                         'native_country_ France', 'native_country_ Germany', 'native_country_ Greece', 'native_country_ Guatemala', 
                                         'native_country_ Haiti', 'native_country_ Holand-Netherlands', 'native_country_ Honduras', 'native_country_ Hong', 
                                         'native_country_ Hungary', 'native_country_ India', 'native_country_ Iran', 'native_country_ Ireland',
                                           'native_country_ Italy', 'native_country_ Jamaica', 'native_country_ Japan', 'native_country_ Laos',
                                             'native_country_ Mexico', 'native_country_ Nicaragua', 'native_country_ Outlying-US(Guam-USVI-etc)', 
                                             'native_country_ Peru', 'native_country_ Philippines', 'native_country_ Poland', 'native_country_ Portugal',
                                               'native_country_ Puerto-Rico', 'native_country_ Scotland', 'native_country_ South', 'native_country_ Taiwan', 
                                               'native_country_ Thailand', 'native_country_ Trinadad&Tobago', 'native_country_ United-States', 'native_country_ Vietnam',
                                                 'native_country_ Yugoslavia']

# Identify the categorical columns
cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']

# ... (previous code)

def eligibilite(request):
    print("Inside eligibilite function")
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            print(data)

            # Create a DataFrame from the received data
            new_user_data = pd.DataFrame(data, index=[0])
            print('before')
            # Perform label encoding for the categorical columns in the new user data
            for col in cat_cols:
                encoder = LabelEncoder()
                new_user_data[col] = encoder.fit_transform(new_user_data[col])
            print('before')    
            # Ensure the columns match the columns used in training
            for col in columns_used_in_training:
                if col not in new_user_data.columns:
                    new_user_data[col] = 0
            print('before')
            # Reorder the columns to match the order in the training data
            new_user_data = new_user_data[columns_used_in_training]

            print('before')
            # Drop the 'income_bracket' and 'income' columns as they are not needed for prediction
            new_user_data.drop(['income_bracket', 'income'], axis=1, inplace=True)

            # Make predictions using the loaded model
            predictions = loaded_model.predict(new_user_data)
            print('after')

            predicted_category = True if predictions[0] == 1 else False
            print(predicted_category)

            # Return the predicted income category as a JSON response
            response_data = {'eligibilite': predicted_category}
            return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})



