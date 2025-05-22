from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Load the model
try:
    model = tf.keras.models.load_model('best_model.h5')
    logger.info("Model loaded successfully")
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def preprocess_image(image_data, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    Adjust target_size based on your model's input requirements
    """
    try:
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values (adjust based on your model's training)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'ML Model API is running',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expects JSON with 'image' field containing base64 encoded image
    """
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400
        
        # Preprocess image
        processed_image = preprocess_image(data['image'])
        
        if processed_image is None:
            return jsonify({
                'status': 'error',
                'message': 'Error processing image'
            }), 400
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Process predictions based on your model type
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Classification model
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            confidence = float(np.max(predictions))
            probabilities = predictions[0].tolist()
            
            result = {
                'status': 'success',
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities
                }
            }
        else:
            # Regression model or single output
            prediction_value = float(predictions[0][0])
            result = {
                'status': 'success',
                'prediction': {
                    'value': prediction_value
                }
            }
        
        logger.info(f"Prediction made successfully: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    try:
        info = {
            'status': 'success',
            'model_info': {
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'layers': len(model.layers),
                'trainable_params': model.count_params()
            }
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting model info: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)