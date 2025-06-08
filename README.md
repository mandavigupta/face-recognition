👤 Face Comparison App using Dlib & Gradio
---

This project uses Dlib's face recognition model to compare two face images and determine if they belong to the same person or different people, using Euclidean distance between face descriptors. The app is deployed using Gradio for easy web-based interaction.

📸 Features
---
1.Upload two images

2.Detect faces using a pretrained Dlib face detector

3.Compute 128D face embeddings using Dlib's face_recognition_resnet_model_v1

4.Compare embeddings using Euclidean distance

5.Return similarity result (same person or different person)

6.User-friendly interface via Gradio

🔧 Requirements
---
Python 3.8+

Dlib

Numpy

Pillow

Gradio

🧪 Installation

1.Clone the repository:
```bash
git clone https://github.com/your-username/face-comparison-app.git
cd face-comparison-app
```
2.Install dependencies:
```bash
pip install dlib numpy pillow gradio
```
3.Download required Dlib models:

- [shape_predictor_5_face_landmarks.dat](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)
- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

Unzip and place them in your working directory.

🚀 Run the App

Gradio will launch a local web interface (usually at http://localhost:7860) where you can upload and compare two face images.

#🧠 How It Works
---
1.Each image is processed using Dlib's face detector.

2.The shape_predictor extracts facial landmarks.

3.A 128-dimensional face descriptor is computed using Dlib's ResNet.

4.Euclidean distance between two descriptors is calculated.

5.If the distance < 0.6 → same person, else → different person.

🖼️ Sample Output:
```bash
Euclidean distance: 0.4123
Result: same person
```
