import io
import os
import cv2
import numpy as np
import requests

from IPython.display import Image, display

## ------------------------------------------------------ ##
base_url = 'http://localhost:8000'
endpoint = '/predict'
model = 'yolov3-tiny'

## ------------------------------------------------------ ##
url_with_endpoint_no_params = base_url + endpoint
url_with_endpoint_no_params

## ------------------------------------------------------ ##
full_url = url_with_endpoint_no_params + "?model=" + model
full_url

## ------------------------------------------------------ ##
def response_from_server(url, image_file, verbose=True):
    files = {'file' : image_file}
    response = requests.post(url, files = files)
    status_code = response.status_code

    if verbose:
        msg = "Everything went well!" if status_code == 200 else ("There was an error when "
                                                                    "handling the request.")
        print(msg)

    return response

## ------------------------------------------------------ ##
with open("images/clock2.jpg", "rb") as image_file:
    prediction = response_from_server(full_url, image_file)

## ------------------------------------------------------ ##
dir_name = "images_predicted"

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

## ------------------------------------------------------ ##
def display_image_from_response(response):
    image_stream = io.BytesIO(response.content)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype = np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    filename = "image_with_objects.jpeg"
    cv2.imwrite(f'images_predicted/{filename}', image)
    display(Image(f'images_predicted/{filename}'))

## ------------------------------------------------------ ##
display_image_from_response(prediction)

## ------------------------------------------------------ ##
image_files = ['car2.jpg', 'clock3.jpg', 'apples.jpg']

for image_file in image_files:
    with open(f"images/{image_file}", "rb") as image_file:
        prediction = response_from_server(full_url, image_file, verbose = False)

    display_image_from_response(prediction)
