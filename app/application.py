from flask import Flask, render_template, request, send_from_directory, url_for
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('models/brain_tumor_model.h5')

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']








# Define the uploads folder (use absolute path)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is in request
        if 'file' not in request.files:
            return render_template('brainscan_updated.html', result=None, error="No file uploaded")
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return render_template('brainscan_updated.html', result=None, error="No file selected")
        
        # Check if file type is allowed
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Debug prints
            print(f"✅ File saved to: {file_path}")
            print(f"✅ File exists: {os.path.exists(file_path)}")

            # Predict the tumor
            result, confidence = predict_tumor(file_path)
            
            # Generate URL for the uploaded file
            file_url = url_for('get_uploaded_file', filename=filename)
            
            print(f"✅ File URL: {file_url}")

            # Return result along with image path for display
            return render_template('index.html', 
                                 result=result, 
                                 confidence=f"{confidence*100:.2f}",
                                 file_path=file_url)
        else:
            return render_template('index.html', 
                                 result=None, 
                                 error="Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP")

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    print(f"📁 Serving file: {filename} from {app.config['UPLOAD_FOLDER']}")
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)