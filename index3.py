from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)


def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2()  # Background subtractor

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)  # Apply background subtraction

        # Create an ellipse mask
        ellipse_mask = np.zeros_like(fgmask)
        center = (ellipse_mask.shape[1] // 2, ellipse_mask.shape[0] // 2)
        axes = (200, 100)  # Example axes lengths, adjust to your needs
        cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)

        # Apply the ellipse mask to the foreground mask
        fgmask = cv2.bitwise_and(fgmask, fgmask, mask=ellipse_mask)

        # Find contours in the masked foreground image
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original frame
        for contour in contours:
            # Filter out small contours to avoid noise
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert fgmask to a three channel image to concatenate with the original frame
        fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        # Apply morphological operations to fill in the holes
        kernel = np.ones((5, 5), np.uint8)
        fgmask_bgr = cv2.morphologyEx(fgmask_bgr, cv2.MORPH_CLOSE, kernel)

        # Concatenate the images side by side ([height, width*2, channels])
        combined = np.hstack((frame, fgmask_bgr))

        # Display the combined image in a single window
    #     cv2.imshow('Combined Image', combined)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
        ret, buffer = cv2.imencode('.jpg', combined)
        combined_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + combined_frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
