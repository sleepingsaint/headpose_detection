import cv2
import sys
import numpy as np
from mediapipe.python.solutions.face_detection import FaceDetection
from google.protobuf.json_format import MessageToDict


def drawBoxes(detection, frame, width, height):

    det = MessageToDict(detection)
    bbox = det["locationData"]["relativeBoundingBox"]
    keypoints = det["locationData"]["relativeKeypoints"]

    x1 = int(bbox["xmin"] * width)
    y1 = int(bbox["ymin"] * height)
    x2 = int(bbox["width"] * width) + x1
    y2 = int(bbox["height"] * height) + y1

    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    landmarks = []
    for point in keypoints:
        x = point['x'] * width
        y = point['y'] * height
        landmarks.append((x, y))

        x, y = int(x), int(y)
        frame = cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)

    return frame, np.array(landmarks, dtype="double")


def detectGaze(model_points, landmarks, camera_matrix, dist_coeffs):
    (success, R, T) = cv2.solvePnP(model_points, landmarks, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), R, T, camera_matrix, dist_coeffs)
        return nose_end_point2D
    return None


def runInference():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        sys.exit("Unable to access webcam")

    # input video constraints
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # video camera contraints
    focal_length = width
    center = (width/2, height/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]],
        dtype="double")
    dist_coeffs = np.zeros((4, 1))

    # worlds model co-ordinates
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    with FaceDetection() as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if results.detections:
                for detection in results.detections:
                    frame, landmarks = drawBoxes(detection, frame, width, height)
                    res_point = detectGaze(model_points, landmarks, camera_matrix, dist_coeffs)

                    if res_point is not None:
                        p1 = (int(landmarks[2][0]), int(landmarks[2][1]))
                        p2 = (int(res_point[0][0][0]), int(res_point[0][0][1]))
                        cv2.line(frame, p1, p2, (255, 0, 0), 2)


            cv2.imshow("output", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    runInference()
