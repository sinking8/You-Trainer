

from flask import Flask, render_template,Response
import cv2
import cv2 as cv

from posturedetect import Model


DEVELOPMENT_ENV = True
font = cv.FONT_HERSHEY_COMPLEX

app = Flask(__name__)


camera = None

@app.route("/")
def index():
    # cv.destroyAllWindows()
    return render_template("index.html")

@app.route('/fitness_detector')
def fitness_detector():
    # print("mask detector")
    return render_template('detector.html')

def generate_frames():
    model = Model()
    global camera
    while True:        
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            
            resized_img = cv.resize(frame,(24,24))
            # img = img.reshape(1,-1)
            
            response = model.predictions_frame(resized_img)["response"]
            cv.putText(frame,response,(100,80),font, 1, (244,250,250), 2)
            cv.imshow('Result', frame)


            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    global camera
    camera=cv2.VideoCapture(0)
    print("video")
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route("/about")
# def about():
#     return render_template("about.html", app_data=app_data)


# @app.route("/service")
# def service():
#     return render_template("service.html", app_data=app_data)


# @app.route("/contact")
# def contact():
#     return render_template("contact.html", app_data=app_data)


if __name__ == "__main__":
    app.run(debug=DEVELOPMENT_ENV)
