from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp
import tempfile

app = Flask(__name__)

CORS(app, resources={r"/upload": {"origins": "http://localhost:3000"}, r"/get-analysis": {"origins": "http://localhost:3000"}})
# CORS(app, resources={r"/upload": {"origins": "http://ai.idance.site/"}, r"/get-analysis": {"origins": "http://ai.idance.site/"}})

response = ''

@app.route('/upload', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        pic = request.files['pic']
        poseName = request.form.get('poseName')
        if not pic:
            return 'No pic uploaded', 400
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        ok = 0
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_filename = temp_file.name
            pic.save(temp_filename)
            image1 = cv2.imread(temp_filename)

            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2) as pose:
                results = pose.process(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
                annotated_image = image1.copy()
                for i in range(11):
                    try:
                        results.pose_landmarks.landmark[i].x = -1
                        ok = 1
                    except:
                        ok = 0
                        
                mp_draw.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                second_img_filename = "annotated.jpg"
                cv2.imwrite(second_img_filename, annotated_image)
        if(ok != 0):
            
            landmarks = results.pose_landmarks.landmark


            # Define the landmark coordinates
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            left_heel=[landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]
            right_heel=[landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]
            right_foot_index=[landmarks[32].x, landmarks[32].y, landmarks[32].z]
            left_foot_index=[landmarks[31].x, landmarks[31].y, landmarks[31].z]


            for landmark_name in mp_pose.PoseLandmark:
                name = str(landmark_name).replace('PoseLandmark.', '').lower()
                exec(f'{name} = [landmarks[landmark_name].x, landmarks[landmark_name].y, landmarks[landmark_name].z]')





            def calculate_angle(a, b, c):
                a = np.array(a)  # First
                b = np.array(b)  # Mid
                c = np.array(c)  # End

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360 - angle

                return angle



            def calculate_slope(pointA, pointB, direction):
                if direction == 'vertical':
                    return 90 - np.arctan((pointA[1] - pointB[1]) / (pointA[0] - pointB[0])) * 180 / np.pi
                return np.arctan((pointA[1] - pointB[1]) / (pointA[0] - pointB[0])) * 180 / np.pi


            def calc_angle_four_points(a, b, c, d):
                a = np.array(a)
                b = np.array(b)
                c = np.array(c)
                d = np.array(d)

                e = a - b
                f = c - d

                cos_angle = (e[0] * f[0] + e[1] * f[1]) / (((e[0] ** 2 + e[1] ** 2) ** 0.5) * ((f[0] ** 2 + f[1] ** 2) ** 0.5))
                angle = np.arccos(cos_angle) * 180 / np.pi
                return angle


            user_left_knee= calculate_angle(left_hip,left_knee,left_ankle)
            user_right_knee=calculate_angle(right_hip,right_knee,right_ankle)
            user_legs= calc_angle_four_points(right_hip, right_knee, left_hip, left_knee)
            user_body=calculate_slope(right_shoulder,right_hip,"horizontal")

            user_right_elbow = calculate_angle(right_wrist, right_elbow, right_shoulder)
            user_left_elbow = calculate_angle(left_wrist, left_elbow, left_shoulder)
            user_right_shoulder = calculate_angle(right_hip, right_shoulder, right_elbow)
            user_left_shoulder = calc_angle_four_points(left_shoulder, left_hip, left_shoulder, left_elbow)
            user_turnout=calc_angle_four_points(right_heel, right_foot_index, left_heel, left_foot_index)

            user=np.array([user_left_knee,user_right_knee,user_legs,user_body])
            message=["Keep your left leg straight.","Keep your right leg straight.","Lift your leg higher.", "Lift your body."]
            goodmessage=["your left leg is straight.", "your right leg is straight.", "your leg is high enough.", "your body position is good."]
            arabesque = np.array([174.1, 179.5, 92.2, 70.5]) # ["Keep your left leg straight","Keep your right leg straight","Lift your leg higher", "Lift your body"])
            user_plie=np.array([user_right_knee,user_left_knee,user_turnout])
            user_arabesque=np.array([user_left_knee,user_right_knee,user_legs,user_body]) # add distance between arms, position of leg compared to knee
            user_passe= np.array([user_legs,max(user_left_knee,user_right_knee),user_left_elbow,user_right_elbow,user_body])
            message_plie=[" Bend your knees more!", " ", "Turnout your feet:Move your heels forward wihou moving your toes"]
            message_passe= [ "Lift you knee higher! ", " ","Straighten your left elbow","Straighten your right elbow", "Straighten your body"]
            message_arabesque=["Keep your left leg straight","Keep your right leg straight","Lift your leg higher", "Lift your body"]
            passe=np.array([ 90, 160.1, 83.9,164.6 ,  71.3 ])
            plie=np.array([107.8871148 ,108.03751473 ,157.55223741])


            def FeedbackAngle(user_angle,alpha, beta, gama, text): #1. user_angle- calculated angle, 2. alpha- desired angle, 3.beta - okay angle, 4. gama- bad angle, text- personal feedback text
                if (user_angle>=alpha):
                    return "Amazing"
                elif (alpha>user_angle>=beta):
                    return text
                else :
                    return text
        

            def FeedbackPose(users, arraypose):
                    
                    feedback = []
                    item = 0
                    sc=[]
                    scoring_points=0
                    for ang in arraypose:
                        result = FeedbackAngle(users[item], 0.9 * ang, 0.8 * ang, 0.7 * ang, message[item],goodmessage[item])
                        if(item!=3):
                          scoring_points=scoring_points + users[item]/ang
                        feedback.append(result)
                        item += 1
                    scoring_percent=round((scoring_points/(item-1))*100, 1)
                    if (scoring_percent>100):
                     score='Total:100% Amazing'
                    elif (scoring_percent>=90):
                     score=f'Total:{scoring_percent}% Amazing'
                    elif (scoring_percent>=80):
                     score=f'Total:{scoring_percent}% Almost there!'
                    else:
                     score= f'Total: {scoring_percent}% Keep up the hard work!'
                    feedback.append(score)
                    return feedback
             
            def FeedbackPasse(users, arrayPose):
                item=0
                feedback = []
                sc=[]
                scoring_points=0
                for ang in arrayPose:
                    result=""
                    if(item==0):
                        result=FeedbackAngle(users[item],0.95*ang,0.8*ang,0.7*ang,message_passe[item]) # angle between legs

                    if(item==2 or item==3):
                        result=FeedbackAngle(users[item], 0.9*ang,0.8*ang,0.7*ang,message_passe[item])
                    scoring_points=scoring_points + users[item]/ang
                    feedback.append(result)
                    item=item+1
                scoring_percent=round((scoring_points/(item-1))*100, 1)
                if (scoring_percent>100):
                        score='Total:100% Amazing'
                elif (scoring_percent>=90):
                       score=f'Total:{scoring_percent}% Amazing'
                elif (scoring_percent>=80):
                       score=f'Total:{scoring_percent}% Almost there!'
                else:
                       score= f'Total: {scoring_percent}% Keep up the hard work!'
                feedback.append(score)
                return feedback
                
            def FeedbackPlie(users, arrayPose):

              item=0
              feedback = []
              sc=[]
              scoring_points=0
              for ang in arrayPose:
                 result=" "
                 if(item==0):
                    if( users[item]>plie[item]):
                       result= message_plie[item]
                    scoring_points=scoring_points + plie[item]/users[item]
                    
            
                 if(item==2):
                    result=FeedbackAngle(users[item], 0.9*ang,0.8*ang,0.7*ang,message_plie[item])
                 #scoring_points=scoring_points + users[item]/ang
                 feedback.append(result)
                 item=item+1
              scoring_percent=round(scoring_points*100, 1)
              #scoring_percent=round((scoring_points/(item-1))*100, 1)
              if (scoring_percent>100):
                     score='Total:100% Amazing'
              elif (scoring_percent>=90):
                     score=f'Total:{scoring_percent}% Amazing'
              elif (scoring_percent>=80):
                     score=f'Total:{scoring_percent}% Almost there!'
              else:
                     score= f'Total: {scoring_percent}% Keep up the hard work!'
              feedback.append(score)
              return feedback
            
            def FeedbackArabesque(users,arrayPose):
                 item=0
                 feedback = []
                 sc=[]
                 scoring_points=0
                 for ang in arrayPose:
                    result=FeedbackAngle(users[item], 0.9*ang,0.8*ang,0.7*ang,message_arabesque[item])
                    if(item!=3):
                        scoring_points=scoring_points + users[item]/ang
                    feedback.append(result)
                    item=item+1
                 scoring_percent=round((scoring_points/(item-1))*100, 1)
                 if (scoring_percent>100):
                  score='Total:100% Amazing'
                 elif (scoring_percent>=90):
                     score=f'Total:{scoring_percent}% Amazing'
                 elif (scoring_percent>=80):
                     score=f'Total:{scoring_percent}% Almost there!'
                 else:
                     score= f'Total: {scoring_percent}% Keep up the hard work!'
                 feedback.append(score)
                 return feedback
            
            if(poseName == 'Plie'):
                arr = 0
            elif(poseName == 'Passe'):
                arr = 1
            elif(poseName == 'Arabesque'):
                arr = 2
                
              # 0 for plie, 1 for passe, 2 for arabesque
            if(arr==0):
                print("analysis for plie")
                analysis = ' '.join(str(item) for item in FeedbackPlie(user_plie, plie))
            elif(arr==1):
                print("analysis for passe")
                analysis = ' '.join(str(item) for item in FeedbackPasse(user_passe, passe))
            elif(arr==2):
                print("analysis for arabesque")
                analysis = ' '.join(str(item) for item in FeedbackArabesque(user_arabesque, arabesque))
                
            

        else:
            analysis = "No humans"
            
        # annotated_image
        
        global response
        response = analysis

        # return send_file('./annotated.jpg', mimetype='image/jpeg')
            
        print("analysis: " + analysis)
        return send_file('./annotated.jpg', mimetype='image/jpeg')
    
@app.route('/get-analysis', methods=['POST', 'GET'])
def get_analysis():
    print(response)
    return response

            
if __name__ == '__main__':
    app.run(debug=True)