import cv2
import winsound

# Open the default camera (index 0)
cam = cv2.VideoCapture(0)

while cam.isOpened():
    # Read two consecutive frames
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()

    # Find the difference between the frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over the contours
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        winsound.PlaySound('alert.wav', winsound.SND_ASYNC)
    
    # Display the processed frame
    cv2.imshow('Granny Cam', frame1)
    
    # Check for key press to exit the loop
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()