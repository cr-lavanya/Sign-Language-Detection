import numpy as np
import track_hand as htm
import time
import cv2
import urllib.request  # Ensure urllib is imported

url = "http://192.168.0.156/800x600.jpg"  # IP Camera URL

wCam, hCam = 800, 600
frameR = 100  # Frame margin
smoothening = 7

pTime = 0

detector = htm.handDetector(maxHands=1)

# Define gestures for all alphabets (A to Z)
sign_language_mapping = {
    'A': [0, 0, 0, 0, 0],  # Closed fist
    'B': [1, 1, 1, 1, 1],  # Open palm
    'C': [0, 1, 1, 0, 0],  # Thumb and index forming a 'C'
    'D': [0, 1, 0, 0, 0],  # Index finger up
    'E': [1, 0, 0, 0, 0],  # Thumb up, others down
    'F': [1, 1, 0, 0, 0],  # Thumb and index finger form an "O"
    'G': [1, 1, 0, 0, 0],  # Thumb and index finger pointing outwarda
    'H': [0, 1, 1, 0, 0],  # Index and middle finger extended
    'I': [0, 0, 0, 0, 1],  # Pinky finger up
    'J': [0, 0, 0, 0, 1],  # Pinky finger moving in a 'J' motion
    'K': [1, 1, 0, 0, 1],  # Index and middle finger up with thumb pointing sideways
    'L': [1, 1, 0, 0, 0],  # Thumb and index finger form an 'L'
    'M': [0, 0, 0, 1, 1],  # Thumb covers three fingers, pinky visible
    'N': [0, 0, 0, 1, 1],  # Thumb covers two fingers, pinky visible
    'O': [1, 1, 1, 1, 1],  # All fingers form an 'O' shape
    'P': [1, 1, 0, 0, 1],  # Index finger and thumb form a 'P'
    'Q': [1, 0, 0, 1, 1],  # Thumb and index finger forming a 'Q'
    'R': [0, 1, 1, 0, 0],  # Index and middle finger crossed
    'S': [0, 0, 0, 0, 0],  # Closed fist
    'T': [0, 0, 0, 1, 0],  # Thumb between index and middle finger
    'U': [0, 1, 1, 0, 0],  # Index and middle fingers up
    'V': [0, 1, 1, 0, 0],  # Index and middle fingers form a 'V'
    'W': [0, 1, 1, 1, 0],  # Index, middle, and ring fingers up
    'X': [0, 1, 0, 0, 0],  # Index finger bent
    'Y': [1, 0, 0, 0, 1],  # Thumb and pinky finger extended
    'Z': [0, 0, 0, 0, 1],  # Draw a 'Z' motion with the index finger
}

while True:
    try:
        # 1. Fetch frame from IP Camera
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)

        # 2. Process the frame to detect hands
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        # 3. Check if landmarks were found
        if len(lmList) != 0:
            # 4. Determine which fingers are up (using landmark positions)
            fingers = detector.fingersUp()

            # 5. Match the gesture to the predefined signs
            detected_sign = "Unknown"
            for sign, finger_pattern in sign_language_mapping.items():
                if fingers == finger_pattern:
                    detected_sign = sign
                    break

            # Display the detected sign on the screen
            cv2.putText(img, f'Sign: {detected_sign}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 4)

            # Draw rectangle on the screen
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 6. Calculate and Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # 7. Display the Image
        cv2.imshow("Sign Language Detection", img)
        cv2.waitKey(1)

    except Exception as e:
        print(f"Error: {e}")
        break
