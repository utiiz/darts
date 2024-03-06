import cv2
import numpy as np


def get_changes_on_ellipse(frame, prev_frame, axes):
    # Compute difference
    difference = cv2.absdiff(prev_frame, frame)

    # Convert to grayscale and threshold to get a binary mask of differences
    conv_hsv_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, diff_mask = cv2.threshold(
        conv_hsv_gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Create an ellipse mask
    ellipse_mask = np.zeros_like(conv_hsv_gray)
    center = (ellipse_mask.shape[1] // 2,
              ellipse_mask.shape[0] // 2)  # Image center
    cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

    # Combine the difference mask and the ellipse mask to get the final mask
    # final_mask = cv2.bitwise_and(diff_mask, ellipse_mask)
    diff_mask[ellipse_mask == 0] = 0

    # Create an image with a black background
    black_background = np.zeros_like(prev_frame)

    # Apply the final mask to make the changes appear white within the ellipse
    # Note: Since we want white, we set all channels to 255 where there are changes
    black_background[diff_mask != 0] = [0, 0, 255]

    # The resulting image (black_background) now has a black background with white changes
    cv2.imshow('diff_Ellipse_White_Changes.png', black_background)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read the first frame
    _, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert the first frame to grayscale
    prev_frame = frame

    # Define the ROI (Region of Interest) as an ellipse
    center = (width // 2, height // 2)
    axes = (300, 150)  # Adjust based on your setup
    # mask = np.zeros_like(prev_frame)
    # cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 0, 0), -1)  # White ellipse on black background

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        cv2.ellipse(frame, center, axes, 0, 0, 360,
                    (255, 0, 0), 1)  # Blue ellipse

        get_changes_on_ellipse(frame, prev_frame, axes)

        # Apply morphological operations to enhance the dart tip
        # kernel = np.ones((5, 5), np.uint8)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #
        # # Find contours of the changes
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # detected_dart = None
        # for contour in contours:
        #     if cv2.contourArea(contour) > 10:  # Filter small contours
        #         x, y, w, h = cv2.boundingRect(contour)
        #         aspect_ratio = w / float(h)
        #         if aspect_ratio > 1.2:  # Filter contours based on aspect ratio
        #             # Check if the contour is within the ellipse
        #             # if cv2.pointPolygonTest(mask, (x + w // 2, y + h // 2), False) >= 0:
        #             detected_dart = contour
        #             break
        #
        # # Draw rectangle around the detected dart contour
        # if detected_dart is not None:
        #     x, y, w, h = cv2.boundingRect(detected_dart)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display frames
        cv2.imshow('Dart Detection', frame)
        # cv2.imshow('Threshold', thresh)  # To see the thresholded image

        # Update the previous frame
        prev_frame = frame.copy()

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
