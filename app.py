from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    Response
)
import threading
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "elder_safe_secret_key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["UPLOAD_FOLDER"] = "static/uploads"

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)

db = SQLAlchemy(app)


# =========================================================
# DATABASE MODELS
# =========================================================

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )


class Notification(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        nullable=True
    )

    message = db.Column(
        db.String(300),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=True
    )

    timestamp = db.Column(
        db.DateTime,
        default=db.func.current_timestamp()
    )


with app.app_context():
    db.create_all()


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = "model/fall3_cnn_detection.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# LOAD CLASS NAMES
# =========================================================

CLASS_FILE = "model/class_names.txt"

if os.path.exists(CLASS_FILE):

    with open(CLASS_FILE, "r") as file:

        CLASS_NAMES = [
            line.strip()
            for line in file
            if line.strip()
        ]

else:

    # Fallback if class_names.txt doesn't exist
    CLASS_NAMES = [
        "fallen",
        "not_fallen"
    ]


print("\nLoaded classes:")

for index, class_name in enumerate(CLASS_NAMES):

    print(
        index,
        "->",
        class_name
    )


# =========================================================
# FALL CLASS DETECTION
# =========================================================

def is_fall_class(class_name):

    """
    Returns True if the predicted class
    represents a fall.
    """

    class_name = class_name.lower().strip()

    fall_names = [
        "fallen",
        "fall",
        "falling"
    ]

    return class_name in fall_names


# =========================================================
# EMAIL CONFIGURATION
# =========================================================

EMAIL_ADDRESS = os.environ.get(
    "ELDERSAFE_EMAIL"
)

EMAIL_PASSWORD = os.environ.get(
    "ELDERSAFE_EMAIL_PASSWORD"
)


def send_email_alert(recipient_email, confidence):

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("Email credentials are not configured.")
        return

    try:

        message = MIMEMultipart()

        message["From"] = EMAIL_ADDRESS
        message["To"] = recipient_email
        message["Subject"] = "ElderSafe AI - Possible Fall Detected"

        body = f"""
Hello,

ElderSafe AI has detected a possible fall event.

Detection confidence: {confidence:.2f}%

Please check the monitored area and take appropriate action.

This is an AI-generated alert and should be verified by a caregiver.

Regards,
ElderSafe AI
"""

        message.attach(
            MIMEText(body, "plain")
        )

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=15
        ) as server:

            server.starttls()

            server.login(
                EMAIL_ADDRESS,
                EMAIL_PASSWORD
            )

            server.sendmail(
                EMAIL_ADDRESS,
                recipient_email,
                message.as_string()
            )

        print(
            "Email alert sent to:",
            recipient_email
        )

    except Exception as e:

        print(
            "Email sending failed:",
            e
        )
def send_email_in_background(recipient_email, confidence):

    email_thread = threading.Thread(
        target=send_email_alert,
        args=(recipient_email, confidence),
        daemon=True
    )

    email_thread.start()
# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(
    image_path
):

    """
    Prepares an uploaded image for the CNN.

    The trained model already contains
    Rescaling(1/255), so we DO NOT divide
    by 255 here.
    """

    image = Image.open(
        image_path
    ).convert("RGB")


    image = image.resize(
        (150, 150)
    )


    image_array = np.array(
        image,
        dtype=np.float32
    )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# =========================================================
# VIDEO CONFIGURATION
# =========================================================

CAMERA_SOURCE = 0

FALL_CONFIRMATION_FRAMES = 5


# =========================================================
# LIVE VIDEO GENERATOR
# =========================================================

def generate_frames():

    camera = cv2.VideoCapture(CAMERA_SOURCE)

    if not camera.isOpened():

        print("Unable to open camera.")

        return

    fall_count = 0

    alert_sent = False

    while True:

        success, frame = camera.read()

        if not success:
            print("Unable to read camera frame.")
            break


        # =================================================
        # CONVERT BGR → RGB
        # =================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # =================================================
        # PREPARE FRAME FOR CNN
        # =================================================

        input_frame = cv2.resize(
            frame_rgb,
            (150, 150)
        )

        input_frame = np.array(
            input_frame,
            dtype=np.float32
        )

        input_frame = np.expand_dims(
            input_frame,
            axis=0
        )


        # =================================================
        # CNN PREDICTION
        # =================================================

        predictions = model.predict(
            input_frame,
            verbose=0
        )[0]


        class_index = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            class_index
        ]

        confidence = float(
            predictions[class_index]
        ) * 100


        # =================================================
        # FALL DETECTION
        # =================================================

        if is_fall_class(predicted_class):

            fall_count += 1

        else:

            fall_count = 0

            # Once the current fall event ends,
            # allow a future fall to generate another alert.
            alert_sent = False


        # =================================================
        # FALL CONFIRMED
        # =================================================

        if (
            fall_count >= FALL_CONFIRMATION_FRAMES
            and not alert_sent
        ):

            print(
                f"FALL DETECTED - "
                f"Confidence: {confidence:.2f}%"
            )


            # ---------------------------------------------
            # SAVE DATABASE NOTIFICATION
            # ---------------------------------------------

            user_id = session.get(
                "user_id"
            )


            if user_id:

                try:

                    with app.app_context():

                        notification = Notification(

                            user_id=user_id,

                            message=(
                                "A possible fall was detected "
                                "by the monitoring system."
                            ),

                            confidence=confidence

                        )

                        db.session.add(
                            notification
                        )

                        db.session.commit()


                        # Get email before leaving context
                        user = db.session.get(
                            User,
                            user_id
                        )


                        if user:

                            recipient_email = user.email

                        else:

                            recipient_email = None


                        db.session.remove()


                    # -------------------------------------
                    # SEND EMAIL IN BACKGROUND
                    # -------------------------------------

                    if recipient_email:

                        send_email_in_background(
                            recipient_email,
                            confidence
                        )


                except Exception as e:

                    print(
                        "Notification error:",
                        e
                    )


            # IMPORTANT
            # Prevent repeated alerts for the
            # same fall event.

            alert_sent = True


        # =================================================
        # DISPLAY PREDICTION
        # =================================================

        if is_fall_class(predicted_class):

            display_text = (
                f"FALL: {confidence:.1f}%"
            )

        else:

            display_text = (
                f"{predicted_class.upper()}: "
                f"{confidence:.1f}%"
            )


        cv2.putText(

            frame,

            display_text,

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


        # =================================================
        # FALL CONFIRMATION DISPLAY
        # =================================================

        if fall_count > 0:

            confirmation_text = (
                f"Fall confirmation: "
                f"{fall_count}/"
                f"{FALL_CONFIRMATION_FRAMES}"
            )

            cv2.putText(

                frame,

                confirmation_text,

                (20, 75),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2

            )


        # =================================================
        # SHOW ALERT STATUS
        # =================================================

        if alert_sent:

            cv2.putText(

                frame,

                "ALERT SENT",

                (20, 110),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2

            )


        # =================================================
        # ENCODE FRAME
        # =================================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue


        frame_bytes = buffer.tobytes()


        # =================================================
        # SEND FRAME TO BROWSER
        # =================================================

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

        )


    camera.release()

# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# SUBSCRIPTION
# =========================================================

@app.route("/subscribe")
def subscribe():

    return render_template(
        "subscribe.html"
    )


# =========================================================
# SIGNUP
# =========================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()


        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        password = request.form.get(
            "password",
            ""
        )


        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if (
            not name
            or len(name) < 2
        ):

            flash(
                "Name must be at least 2 characters long.",
                "error"
            )

            return redirect(
                url_for("signup")
            )


        if (
            not email
            or "@" not in email
        ):

            flash(
                "Please enter a valid email.",
                "error"
            )

            return redirect(
                url_for("signup")
            )


        if (
            len(password) < 6
            or not any(
                char.isalpha()
                for char in password
            )
            or not any(
                char.isdigit()
                for char in password
            )
        ):

            flash(
                "Password must contain at least 6 characters, including letters and numbers.",
                "error"
            )

            return redirect(
                url_for("signup")
            )


        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("signup")
            )


        # -------------------------------------------------
        # CHECK EXISTING USER
        # -------------------------------------------------

        existing_user = User.query.filter_by(
            email=email
        ).first()


        if existing_user:

            flash(
                "Email already registered.",
                "error"
            )

            return redirect(
                url_for("signup")
            )


        # -------------------------------------------------
        # CREATE USER
        # -------------------------------------------------

        hashed_password = generate_password_hash(
            password
        )


        new_user = User(

            name=name,
            email=email,
            password=hashed_password

        )


        try:

            db.session.add(
                new_user
            )

            db.session.commit()


            flash(
                "Registration successful! Please log in.",
                "success"
            )


            return redirect(
                url_for("signin")
            )


        except Exception as e:

            db.session.rollback()

            print(
                "Signup error:",
                e
            )


            flash(
                "An error occurred during registration.",
                "error"
            )


            return redirect(
                url_for("signup")
            )


    return render_template(
        "signup.html"
    )


# =========================================================
# SIGNIN
# =========================================================

@app.route(
    "/signin",
    methods=["GET", "POST"]
)
def signin():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        password = request.form.get(
            "password",
            ""
        )


        user = User.query.filter_by(
            email=email
        ).first()


        if (
            user
            and check_password_hash(
                user.password,
                password
            )
        ):

            session["user_id"] = user.id

            session["user_name"] = user.name

            session["user_email"] = user.email


            flash(
                "Login successful!",
                "success"
            )


            return redirect(
                url_for("dashboard")
            )


        flash(
            "Invalid email or password.",
            "error"
        )


    return render_template(
        "signin.html"
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        flash(
            "Please log in first.",
            "error"
        )

        return redirect(
            url_for("signin")
        )


    return render_template(
        "dashboard.html"
    )


# =========================================================
# VIDEO PAGE
# =========================================================

@app.route("/video")
def video():

    if "user_id" not in session:

        flash(
            "Please log in first.",
            "error"
        )

        return redirect(
            url_for("signin")
        )


    return render_template(
        "video.html"
    )


# =========================================================
# VIDEO STREAM
# =========================================================

@app.route("/video_feed")
def video_feed():

    if "user_id" not in session:

        return "Unauthorized", 401

    return Response(

        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )

    )

# =========================================================
# IMAGE DETECTION
# =========================================================

@app.route(
    "/detection-panel",
    methods=["GET", "POST"]
)
def detection():

    if "user_id" not in session:

        return redirect(
            url_for("signin")
        )


    prediction = None

    confidence = None

    image_path = None


    if request.method == "POST":

        file = request.files.get(
            "image"
        )


        if file and file.filename:

            filename = secure_filename(
                file.filename
            )


            image_path = os.path.join(

                app.config["UPLOAD_FOLDER"],

                filename

            )


            file.save(
                image_path
            )


            image = preprocess_image(
                image_path
            )


            predictions = model.predict(
                image,
                verbose=0
            )[0]


            class_index = int(
                np.argmax(predictions)
            )


            prediction = CLASS_NAMES[
                class_index
            ]


            confidence = round(
                float(
                    predictions[
                        class_index
                    ]
                ) * 100,
                2
            )


    return render_template(

        "detection.html",

        prediction=prediction,

        confidence=confidence,

        image_path=image_path

    )


# =========================================================
# NOTIFICATIONS
# =========================================================

@app.route("/notifications")
def notifications():

    if "user_id" not in session:

        flash(
            "Please log in first.",
            "error"
        )

        return redirect(
            url_for("signin")
        )


    alerts = Notification.query.filter_by(

        user_id=session["user_id"]

    ).order_by(

        Notification.timestamp.desc()

    ).all()


    return render_template(

        "notifications.html",

        notifications=alerts

    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()


    flash(
        "You have been logged out.",
        "success"
    )


    return redirect(
        url_for("home")
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )