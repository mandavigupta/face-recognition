import dlib
import numpy as np
import gradio as gr
from PIL import Image

predictor_path = "shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
sp=dlib.shape_predictor(predictor_path)
face=dlib.face_recognition_model_v1(face_rec_model_path)

def face_description(img ,detector,sp,face):
    img= np.array(img.convert("RGB"))
    dets= detector(img,1)
    if len(dets)==0:
        return None
    shape= sp(img,dets[0])
    descriptor =face.compute_face_descriptor(img,shape)
    return np.array(descriptor)
def compare_face(img1,img2):
    des1=face_description(img1,detector,sp,face)
    des2=face_description(img2,detector,sp,face)
    if des1 is None or des2 is None:
        return "face cant be detect "
    distance=np.linalg.norm(des1-des2)
    result="same person" if distance<0.6 else "different person"
    return f"Euclidean distance:{distance:.4f}\nResult:{result}"

interface=gr.Interface(
    fn=compare_face,
    outputs="text",
    description= "upload two images and find out they are same person or different person ",
    inputs = [gr.Image(type="pil",label="img1"),gr.Image(type="pil",label="img2")] ,
    title ="Face comparison "

    )
interface.launch()   