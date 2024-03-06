import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read the first frame
    _, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert the first frame to grayscale
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the ROI (Region of Interest) as an ellipse
    center = (width // 2, height // 2)
    axes = (width // 4, height // 6)  # Adjust based on your setup
    mask = np.zeros_like(prev_frame)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 0, 0), -1)  # White ellipse on black background

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 0, 0), 1)  # Blue ellipse

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(prev_frame, gray_frame)

        # Apply the mask to focus only on the dartboard area
        roi_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=mask)

        # Threshold the difference to get significant changes
        _, thresh = cv2.threshold(roi_diff, 15, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed

        # Apply morphological operations to enhance the dart tip
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours of the changes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_dart = None
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if aspect_ratio > 1.2:  # Filter contours based on aspect ratio
                    # Check if the contour is within the ellipse
                    # if cv2.pointPolygonTest(mask, (x + w // 2, y + h // 2), False) >= 0:
                    detected_dart = contour
                    break

        # Draw rectangle around the detected dart contour
        if detected_dart is not None:
            x, y, w, h = cv2.boundingRect(detected_dart)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display frames
        cv2.imshow('Dart Detection', frame)
        cv2.imshow('Threshold', thresh)  # To see the thresholded image

        # Update the previous frame
        prev_frame = gray_frame.copy()

        # Break the loop with 'q'
        if cv2.waitKey(10000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
