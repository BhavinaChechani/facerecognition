import cv2
import os
from time import sleep

# Create a folder to store the images
folder_name = "data"
os.makedirs(folder_name, exist_ok=True)

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
sleep(2)

while True:
    try:
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        print(frame)  # prints matrix values of each frame

        # Display instruction on the frame
        cv2.putText(frame, "Press 's' to capture image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capturing", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):

            # Prompt the user to enter the name for the saved image
            image_name = input("Enter the name for the saved image: ")
            filename = f"{folder_name}/{image_name}.jpg"

            cv2.imwrite(filename=filename, img=frame)
            webcam.release()
            break

        elif key == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break

   

    except KeyboardInterrupt:
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break