import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_leaflet as dl
import plotly.graph_objects as go
import mxnet as mx
import pandas as pd
import numpy as np
from PIL import Image
import exifread
import os
import joblib
import base64
from io import BytesIO

# Constants and fixed data
train_images_dir = os.path.join(os.getcwd(), "train")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
IMAGE_HEIGHT = 300  # Height of image display
COLORS = {"Plastic": "yellow", "No plastic": "blue"}

# Loading the pre-trained model (equivalent to R's load('model400.RData'))
model_default = joblib.load("model400.pkl")

# Functions
def extract_feature_test(dir_path, width, height):
    img_size = width * height
    images_names = os.listdir(dir_path)
    feature_list = []
    
    for img_name in images_names:
        img_path = os.path.join(dir_path, img_name)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_resized = img.resize((width, height))
        img_vector = np.asarray(img_resized).flatten()
        feature_list.append(img_vector)
    
    feature_matrix = np.vstack(feature_list)
    feature_df = pd.DataFrame(feature_matrix, columns=[f"pixel{i}" for i in range(1, img_size + 1)])
    
    return feature_df

def predict_images(model, test_image_dir):
    print("Reading the test images...")
    size_foto = 25
    test_features = extract_feature_test(test_image_dir, width=size_foto, height=size_foto)
    test_array = test_features.to_numpy().reshape(size_foto, size_foto, 1, -1)
    
    # Prediction
    predictions = model.predict(test_array)
    predicted_labels = np.argmax(predictions, axis=1)
    
    return predicted_labels

# Dash app setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Identification of floating litter from aerial imaging"),
    
    dcc.Upload(id="upload-image", children=html.Button("Upload JPG File"), multiple=False),
    
    dcc.Input(id="num-split", type="number", value=1, placeholder="Image splitting (rows and columns)"),
    dcc.Input(id="altura", type="number", value=200, placeholder="Viewpoint height (m)"),
    dcc.Input(id="dfocal", type="number", value=40, placeholder="Focal distance (mm)"),
    
    html.Button("Analyze Image", id="analyze-button"),
    html.Div(id="output-status"),
    
    dcc.Graph(id="image-graph"),
    
    dl.Map([dl.TileLayer(), dl.Marker(position=(51.505, -0.09))], id="leaflet-map", style={'width': '100%', 'height': '400px'})
])

@app.callback(
    [Output("output-status", "children"),
     Output("image-graph", "figure")],
    [Input("analyze-button", "n_clicks")],
    [Input("upload-image", "contents"),
     Input("num-split", "value"),
     Input("altura", "value"),
     Input("dfocal", "value")]
)
def analyze_image(n_clicks, image_contents, num_split, altura, dfocal):
    if n_clicks is None:
        return "", go.Figure()
    
    if image_contents is not None:
        content_type, content_string = image_contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(BytesIO(decoded))
        
        # Process the image and extract features
        # Assuming the directory for storing the uploaded image is "output_directory"
        output_dir = "output_directory"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        img.save(os.path.join(output_dir, "uploaded_image.jpg"))
        
        # Predicting the labels using the model
        predictions = predict_images(model_default, output_dir)
        
        # Create figure for displaying image with predictions
        fig = go.Figure()
        fig.add_trace(go.Image(z=np.array(img)))
        fig.update_layout(title="Analyzed Image")
        
        return f"Image analyzed. Predictions: {predictions}", fig
    
    return "Please upload an image to analyze.", go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
