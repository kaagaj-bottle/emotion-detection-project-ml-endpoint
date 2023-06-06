import numpy as np
from PIL import Image
import base64
from io import BytesIO
from flask import Flask,jsonify,request
from flask_cors import CORS
from model import model_output

app = Flask(__name__)
CORS(app,origins="*")


@app.route("/",methods=["POST"])
def predict():
    #receive the base64 encoded image 
    base64_image_input = request.get_json()['image'].split(",")[1]
    
    #decode the base64 encoded image
    image_input_data=base64.b64decode(base64_image_input)
    
    #open the image using Image class from pillow
    image=Image.open(BytesIO(image_input_data))

    #resize image to 48x48 pixel as that is what the model accepts
    resized_image_input=image.resize((48,48))

    #convert the image into grayscale
    grayscale_image=resized_image_input.convert("L")

    #conver the image into pixel data array
    pixel_image_input=np.array(grayscale_image)

    #rescaling the image so that pixel values are between 0 and 1
    pixel_image_input=pixel_image_input/255.

    #reshape the image
    reshaped_image_input=np.expand_dims(pixel_image_input,axis=0)

    #get the output from the model
    output=int(model_output(reshaped_image_input))

    return jsonify(prediction=output)

if __name__ == "__main__":
    app.run(debug=True)