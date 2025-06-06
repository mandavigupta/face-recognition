import sys
import dlib
import numpy as np

def load_face_descriptor(image_path, detector, sp, facerec):
    img = dlib.load_rgb_image(image_path)
    dets = detector(img, 1)

    if len(dets) == 0:
        raise ValueError(f"No face detected in image {image_path}")

    # Assume the first detected face
    shape = sp(img, dets[0])
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python compare_faces.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat image1.jpg image2.jpg")
        sys.exit(1)

    predictor_path = sys.argv[1]
    face_rec_model_path = sys.argv[2]
    img1_path = sys.argv[3]
    img2_path = sys.argv[4]

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    try:
        descriptor1 = load_face_descriptor(r"D:\mandavi\recognition\image.jpg", detector, sp, facerec)
        descriptor2 = load_face_descriptor(r"D:\mandavi\recognition\download.jpg", detector, sp, facerec)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Compute Euclidean distance
    dist = np.linalg.norm(descriptor1 - descriptor2)
    print(f"Euclidean distance between faces: {dist:.4f}")

    # Threshold: less than 0.6 means same person (adjust as needed)
    if dist < 0.6:
        print("Result: Same person")
    else:
        print("Result: Different persons")
