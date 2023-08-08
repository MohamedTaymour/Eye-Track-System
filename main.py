import cv2
import mediapipe as mp
import numpy as np
import math
import serial


#determining face and eye positions
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

RIGHT_IRIS = [474,475,476,477]
LEFT_IRIS = [469,470,471,472]

L_H_LEFT = [33]
L_H_RIGHT = [133]
L_H_UP = [27]
L_H_DOWN = [23]

R_H_LEFT = [362]
R_H_RIGHT = [263]
R_H_UP = [257]
R_H_DOWN = [254]

ser = serial.Serial("com6", 9600)
def esp32(Vertical,Horiztonal):
    if Vertical == "Top":
        ser.write(b'T\n')

    if Vertical == "Bottom":
        ser.write(b'B\n')

    if Vertical == "Center":
        ser.write(b'V\n')

    if Horiztonal == "Left":
        ser.write(b'L\n')

    if Horiztonal == "Right":
        ser.write(b'R\n')

    if Horiztonal == "Center":
        ser.write(b'H\n')
def euclidean_distance(point1,point2):
    x1,y1 = point1.ravel()
    x2,y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance
def Iris_Hposition(iris_center , right_point , left_point):
    try:
        center_to_right_distance = euclidean_distance(iris_center, right_point)
        total_distance = euclidean_distance(right_point, left_point)
        ratio = center_to_right_distance / total_distance
    except ZeroDivisionError:
        ratio = 0
    iris_position = ""
    if ratio == 0:
        iris_position = "Closed Eyes"
    elif ratio<= 0.42:
        iris_position = "Right"
    elif ratio >0.42 and ratio <= 0.57:
        iris_position = "Center"
    else:
        iris_position = "Left"
    return iris_position,ratio

def Iris_Vposition(iris_center , top_point , bottom_point):
    try:
        center_to_top_distance = euclidean_distance(iris_center, top_point)
        total_distance = euclidean_distance(top_point, bottom_point)
        ratio = center_to_top_distance / total_distance
    except:
        ratio = 0
    iris_position = ""
    if ratio == 0:
        iris_position = "Closed Eyes"
    elif ratio <= 0.6:
        iris_position = "Top"
    elif ratio > 0.6 and ratio <= 0.7:
        iris_position = "Center"
    else:
        iris_position = "Bottom"
    return iris_position,ratio

#enabling camera and adjusting detection features
cam = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        #Camera input
        ret, frame = cam.read()
        if not ret:
            break

        #enabling features to help ease the facial detection
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        img_h,img_w = frame.shape[:2]

        if results.multi_face_landmarks:
            #placing all landmarks in the face
            mesh_points = np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            #making circle around the iris and determining the iris position
            #print (mesh_points.shape)
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left= np.array([l_cx,l_cy],dtype=np.int32)
            center_right = np.array([r_cx,r_cy],dtype=np.int32)

            cv2.circle(frame, center_right, int(l_radius),(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)

            cv2.circle(frame, mesh_points[R_H_RIGHT][0],2,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(frame, mesh_points[R_H_LEFT][0],2,(0,255,0),1,cv2.LINE_AA)

            cv2.circle(frame, mesh_points[L_H_RIGHT][0], 2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[L_H_LEFT][0], 2, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.circle(frame, mesh_points[R_H_UP][0], 2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[R_H_DOWN][0], 2, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.circle(frame, mesh_points[L_H_UP][0], 2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[L_H_DOWN][0], 2, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.circle(frame, center_right, 2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, center_left, 2, (0, 255, 0), 1, cv2.LINE_AA)

            iris_h_position,ratio_h = Iris_Hposition(center_right, mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0])
            iris_v_position, ratio_v = Iris_Vposition(center_right, mesh_points[R_H_UP], mesh_points[R_H_DOWN][0])

            #esp32(iris_v_position,iris_h_position)

            print(iris_v_position,iris_h_position)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cam.release()
cv2.destroyAllWindows()
