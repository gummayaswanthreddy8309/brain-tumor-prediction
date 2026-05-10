from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load your trained model
model = load_model(r'D:\miniproject\brain_tumor_classifier_model.h5')  # Adjust path if necessary
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('sample.html')  # Use 'sample.html' based on your provided HTML code

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('f1')  # Match with the name="f1" in the HTML input
    if not file:
        return render_template('sample.html', result=None, error="No file uploaded.")

    # Check for valid file format
    valid_extensions = ['jpg', 'jpeg', 'png']
    file_extension = os.path.splitext(file.filename)[1][1:].lower()  # Get file extension and convert to lower case

    if file_extension not in valid_extensions:
        return render_template('sample.html', result=None, error="Invalid file format. Please upload a file in jpg, jpeg, or png format.")

    # Process the uploaded file and prepare it for prediction
    img = Image.open(file).convert('RGB')  # Ensure image is in RGB mode
    img = img.resize((256, 256))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 256, 256, 3)  # Reshape to add batch dimension

    # Perform prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = labels[predicted_class]
    confidence = (prediction[0][predicted_class] * 100) - 3

    # Prepare result for rendering
    result = {'label': predicted_label, 'confidence': f"{confidence:.2f}%"}

    return render_template('results.html', out=result)

if __name__ == '__main__':
    app.run(debug=True)