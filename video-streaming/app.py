from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # use 0 for web camera

# function to generate frames


def gen_frames():
    while True:

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier(
                './opencv/haarcascades/haarcascade_frontalface_default.xml')
            eye_casecade = cv2.CascadeClassifier(
                './opencv/haarcascades/haarcascade_eye.xml')
            faces = detector.detectMultiScale(frame, 1.3, 5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_casecade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex+ew, ey+eh), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# API Routes
# route /
@ app.route('/')
def index():
    return render_template('index.html')

# route /video


@ app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
