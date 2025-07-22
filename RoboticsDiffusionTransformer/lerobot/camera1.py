import time
import cv2

cap = cv2.VideoCapture(0 ,cv2.CAP_V4L2)
# cap = cv2.VideoCapture(0 ,cv2.CAP_V4L2)

fps = 90
width = 1600
height = 1200
capture_dir = 'outputs/images_from_opencv_cameras/'

cap.set(cv2.CAP_PROP_FPS, fps)
act_fps = cap.get(cv2.CAP_PROP_FPS)

print(act_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


frame_index = 0
start_time = time.time()
while time.time() - start_time <= 1.1:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break
    cv2.imwrite(capture_dir + f"frame_{frame_index:06d}.png", frame)
    frame_index += 1

# Release the capture
cap.release()