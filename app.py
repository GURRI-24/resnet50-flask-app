from flask import *


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

app = Flask(__name__)

# Load the trained model (replace with your actual model path)
# model = load_model(r"C:\Users\GURRI\Documents\dogcata(79)v(75).h5")  # Adjust the path to your model file
@app.route('/')
def index():
    return render_template('layout.html')
# Pretrained ResNet model for broader classification (optional)
resnet_model = ResNet50(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_path = "temp.jpg"
    file.save(img_path)

    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))  # Adjust size based on your model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Normalize for ResNet
    # Predict image class
    predictions = resnet_model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=3)  # Get top 3 predictions
    results = []
# Show result
    for i, (imagenet_id, label, score) in enumerate(decoded_preds[0]):
        # Print the result
        
        results.append({"rank": i + 1, "label": label, "confidence": f"{score:.2%}"})
        

    # # Make prediction
    # prediction = model.predict(img_array)

    # # If using a categorical model (softmax), get the class with highest probability
    # # predicted_label = np.argmax(prediction, axis=1)[0]
    # # If using a binary model, threshold the prediction 
   
    # # Adjust the threshold based on your model's output
    # predicted_label = 1 if prediction[0][0] > 0.5 else 0  # Adjust threshold as needed
    
    return jsonify({"predictions": results})  # ✅ Correct return statement
if __name__ == '__main__':
    app.run(debug=True)