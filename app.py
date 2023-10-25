import sys
import os
import cv2
import numpy as np
import imutils
from PyPDF2 import PdfWriter, PdfReader
import io
from io import BytesIO


from imutils import perspective
from rembg.bg import remove as rembg
from PIL import Image

APPROX_POLY_DP_ACCURACY_RATIO = 0.02
IMG_RESIZE_H = 500.0

def scan(data):
    bytes = np.frombuffer(rembg(data), np.uint8)

    img = cv2.imdecode(bytes, cv2.IMREAD_UNCHANGED)
    orig = img.copy()

    ratio = img.shape[0] / IMG_RESIZE_H

    img = imutils.resize(img, height=int(IMG_RESIZE_H))
    _, img = cv2.threshold(img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    img = cv2.medianBlur(img, 15)

    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    outline = None

    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, APPROX_POLY_DP_ACCURACY_RATIO * perimeter, True)

        if len(polygon) == 4:
            outline = polygon.reshape(4, 2)

    if outline is None:
        r = orig
    else:
        r = perspective.four_point_transform(orig, outline * ratio)
    
    print(r.shape)

    _, buf = cv2.imencode(".jpeg", r)
    return buf.tobytes(), outline, r.shape

with open('lego.jpeg', "rb") as f:
    buffer , outline, shape =scan(f.read())
    print(buffer)
    print((shape[0],shape[1]))
    im_1 = Image.open(io.BytesIO(buffer)) 
    im_1 = im_1.convert('RGB')
    im_1.save(r'scanned.pdf')

    reader = PdfReader('scanned.pdf') 
    writer = PdfWriter()

    for page in reader.pages:
        page.cropbox.lower_left= outline[0]
        page.cropbox.lower_right= outline[1]
        page.cropbox.upper_left=outline[2]
        page.cropbox.upper_right= outline[3]
    writer.add_page(page) 
        
    with open('result.pdf','wb') as fp:
        writer.write(fp)
