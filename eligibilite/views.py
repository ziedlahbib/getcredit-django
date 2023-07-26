from django.shortcuts import render
from django.http import HttpResponse, JsonResponse  # Import JsonResponse here
import json
import tensorflow as tf

# Replace 'saved_model_directory' with the path where your model was saved
saved_model_directory = r'C:\my_saved_model\1690126746\1690127283'

# Load the saved model
loaded_model = tf.saved_model.load(saved_model_directory)

# Get the prediction function from the loaded model
prediction_fn = loaded_model.signatures["predict"]

def eligibilite(request):
    try:
        # Check if the request method is POST
        if request.method == 'POST':
            # Parse JSON data from the request body
            data = json.loads(request.body)

            # Print the received data for debugging purposes
            print("Received data:", data)

            # Prepare the input data for prediction
            input_data = {
                'age': tf.constant([int(data['age'])], dtype=tf.int64),
                'workclass': tf.constant([data['workclass']], dtype=tf.string),
                'education': tf.constant([data['education']], dtype=tf.string),
                'education_num': tf.constant([int(data['education_num'])], dtype=tf.int64),
                'marital_status': tf.constant([data['marital_status']], dtype=tf.string),
                'occupation': tf.constant([data['occupation']], dtype=tf.string),
                'relationship': tf.constant([data['relationship']], dtype=tf.string),
                'race': tf.constant([data['race']], dtype=tf.string),
                'gender': tf.constant([data['gender']], dtype=tf.string),
                'capital_gain': tf.constant([int(data['capital_gain'])], dtype=tf.int64),
                'capital_loss': tf.constant([int(data['capital_loss'])], dtype=tf.int64),
                'hours_per_week': tf.constant([int(data['hours_per_week'])], dtype=tf.int64),
                'native_country': tf.constant([data['native_country']], dtype=tf.string)
            }

            # Make the prediction using the prediction function
            predictions = prediction_fn(**input_data)

            # Access the predicted class using the 'class_ids' key
            predicted_class = predictions['class_ids'].numpy()[0]

            # Return the predicted income category as a JSON response
            # Set content type as "application/json"
            response_data = {'eligibilite': bool(predicted_class == 1)}  # Convert to Python native bool
            return JsonResponse(response_data)

    except Exception as e:
        # Return an error response if there was an exception during prediction
        return JsonResponse({'error': str(e)})

    # Return an error response if the request method is not POST
    return JsonResponse({'error': 'Invalid request method'})
