import io
import os
import subprocess
import cv2
import cvlib as cv
import nest_asyncio
import numpy as np
import uvicorn

from enum import Enum
from cvlib.object_detection import draw_bbox
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from IPython.display import Image, display


## ------------------------------------------------------ ##
image_files = ['apple.jpg', 'clock.jpg', 'oranges.jpg', 'car.jpg']

for image_file in image_files:
    print(f"\nDisplaying image: {image_file}")
    display(Image(filename=f"images/{image_file}"))

## ------------------------------------------------------ ##
dir_name = "images_with_boxes"

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

## ------------------------------------------------------ ##
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def detect_and_draw_box(filename, model = "yolov3-tiny", confidence = 0.5):
    img_filepath = f'images/{filename}'

    img = cv2.imread(img_filepath)

    bbox, label, conf = cv.detect_common_objects(img, confidence = confidence, model = model)

    print(f"========================\nImage processed: {filename}\n")

    for l, c in zip(label, conf):
        print(f"Detected object: {l} with confidence level of {c}\n")

    output_image = draw_bbox(img, bbox, label, conf)

    cv2.imwrite(f'images_with_boxes/{filename}', output_image)

    display(Image(f'images_with_boxes/{filename}'))

## ------------------------------------------------------ ##
for image_file in image_files:
    detect_and_draw_box(image_file)

## ------------------------------------------------------ ##
detect_and_draw_box("fruits.jpg")

## ------------------------------------------------------ ##
detect_and_draw_box("fruits.jpg", confidence = 0.2)

## ------------------------------------------------------ ##
dir_name = "images_uploaded"

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

## ------------------------------------------------------ ##
app = FastAPI(title = 'Deploying an ML Model with FastAPI')


class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://serve/docs"

@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")

    if not fileExtension:
        raise HTTPException(status_code = 415, detail = "Unsupported file provided.")

    image_stream = io.BytesIO(file.file.read())

    image_stream.seek(0)

    file_bytes = np.asarray(bytearray(image_stream.read()), dtype = np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    bbox, label, conf = cv.detect_common_objects(image, model = model)

    output_image = draw_bbox(image, bbox, label, conf)

    cv2.imwrite(f'images_uploaded/{filename}', output_image)

    file_image = open(f'images_uploaded/{filename}', mode = "rb")

    return StreamingResponse(file_image, media_type = "image/jpeg")

## ------------------------------------------------------ ##
def print_service_access_url(port: int = 8000, path: str = "docs") -> None:
    ip = os.environ["HOSTNAME"].split(".")[0][3 : ]
    base_domain = os.environ["REV_PROXY_BASE_DOMAIN"]
    lab_url = base_domain.format(ip = ip, port = port)

    print(f"✅ Access your app at:\n{lab_url}/{path}\n")

## ------------------------------------------------------ ##
nest_asyncio.apply()

host = "0.0.0.0"

print_service_access_url()

uvicorn.run(app, host = host, port = 8000)
