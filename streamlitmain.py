import os
import dlib
import numpy as np
from skimage import io
import cv2
import streamlit as st
import matplotlib.image as mpimg



# Globals
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("E:/Scaledge/Internship/Attendance_system/face_recognition_dlib/dlib/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("E:/Scaledge/Internship/Attendance_system/face_recognition_dlib/dlib/dlib_face_recognition_resnet_model_v1.dat")


def get_face_encodings(face):
    bounds = face_detector(face, 1)
    faces_landmarks = [shape_predictor(face, face_bounds) for face_bounds in bounds]
    return [np.array(face_recognition_model.compute_face_descriptor(face, face_pose, 1)) for face_pose in faces_landmarks]


def get_face_matches(known_faces, face):
    return np.linalg.norm(known_faces - face, axis=1)


def find_match(known_faces, person_names, face):
    matches = get_face_matches(known_faces, face) # get a list of True/False
    min_index = matches.argmin()
    min_value = matches[min_index]
    if min_value < 0.55:
        return person_names[min_index]+"! ({0:.2f})".format(min_value)
    if min_value < 0.58:
        return person_names[min_index]+" ({0:.2f})".format(min_value)
    if min_value < 0.65:
        return person_names[min_index]+"?"+" ({0:.2f})".format(min_value)
    return 'Not Found'


def load_face_encodings(faces_folder_path):
    folder = os.listdir(faces_folder_path)
    
    image_filenames = filter(lambda x: x.endswith('.jpg'), folder)
    image_filenames = sorted(image_filenames)
    # print(image_filenames)
    # for i in folder:
    #     print(os.walk(i))
    # files = map(lambda x: os.listdir(x), os.listdir(faces_folder_path))
    # files = sorted(files)
    
    person_names = [x[:-4] for x in image_filenames]
    print(person_names)

    full_paths_to_images = [faces_folder_path + x for x in image_filenames]
    print(full_paths_to_images)
    face_encodings = []

    # win = dlib.image_window()

    for path_to_image in full_paths_to_images:
        face = io.imread(path_to_image)

        faces_bounds = face_detector(face, 1)
        # print(faces_bounds)

        face_bounds = faces_bounds[0]
        face_landmarks = shape_predictor(face, face_bounds)
        face_encoding = np.array(face_recognition_model.compute_face_descriptor(face, face_landmarks, 1))

        # win.clear_overlay()
        # win.set_image(face)
        # win.add_overlay(face_bounds)
        # win.add_overlay(face_landmarks)
        face_encodings.append(face_encoding)

        # print(face_encoding)

        #dlib.hit_enter_to_continue()
    return face_encodings, person_names


def recognize_faces_in_video(face_encodings, person_names):
    faceClassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rects = faceClassifier.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (50, 50),
            flags = cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]
            face_encodings_in_image = get_face_encodings(face)
            if (face_encodings_in_image):
                match = find_match(face_encodings, person_names, face_encodings_in_image[0])
                cv2.putText(frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("bilde", frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


# face_encodings, person_names = load_face_encodings(faces_folder_path)
# recognize_faces_in_video(face_encodings, person_names)





# Define Streamlit app
def main():
    st.title("Face Recognition App")


    st.subheader("Welcome to the Face Recognition App")
    st.write("This app performs face recognition using a webcam stream. or using given image")
    st.write("select the option to proceed")
    if st.button("Start livestream"):
        st.subheader("Face Recognition Execution")


        # Load known face encodings and person names
        faces_folder_path = 'E:/Scaledge/Internship/Attendance_system/face_recognition_dlib/data/'
        face_encodings, person_names = load_face_encodings(faces_folder_path)

        # Initialize video capture
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            face_rects = face_detector(gray)

            for rect in face_rects:
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = frame[y:y + h, x:x + w]
                face_encodings_in_image = get_face_encodings(face)
                if face_encodings_in_image:
                    match = find_match(face_encodings, person_names, face_encodings_in_image[0])
                    cv2.putText(frame, match, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            st.image(frame, channels="BGR", use_column_width=True)

        cap.release()

    elif st.button("Image"):
        st.subheader("Get Image from Folder")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            st.image(bytes_data)
            s_image = io.BytesIO(bytes_data)
            input_shape = (224, 224, 3)
            image = image.load_img(s_image, target_size=(input_shape[0], input_shape[1]))
            st.image(mpimg.imread(image), use_column_width=True)
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            face_rects = face_detector(gray)
    
            for rect in face_rects:
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = image[y:y + h, x:x + w]
                face_encodings_in_image = get_face_encodings(face)
                if face_encodings_in_image:
                    match = find_match(face_encodings, person_names, face_encodings_in_image[0])
                    cv2.putText(image, match, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            

            # st.image(mpimg.imread(image), use_column_width=True)
            st.write(image, channels="BGR", use_column_width=True)

if __name__ == '__main__':
    main()

