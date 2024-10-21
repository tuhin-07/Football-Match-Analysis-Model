from ultralytics import YOLO
import supervision as sv
import pickle as pk
import os
import sys
from utils import get_center_bound_box, get_bound_box_width, get_foot_position
import cv2
import numpy as np
import pandas as pd

class Tracker():
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bound_box']
                    if object == 'ball':
                        position= get_center_bound_box(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def ball_interpolation(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bound_box',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1:{"bound_box":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self,frames):
        detection=[]
        batch_size=20
        for i in range(0,len(frames),batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detection += detection_batch

        return detection
    
    def get_object_track(self,frames,read_from_stub=False,stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pk.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        tracks={
                'players':[],
                'referees':[],
                'ball':[]
            }
        for frame_num , detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()} 

            #convert to supervision detection format
            detection_supervison = sv.Detections.from_ultralytics(detection)

            for obj_index, cls_id in enumerate(detection_supervison.class_id):
                if cls_names[cls_id] == 'goalkeeper':
                    detection_supervison.class_id[obj_index] = cls_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervison)

            
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bound_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bound_box':bound_box}

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bound_box':bound_box}

            for frame_detection in detection_supervison:
                bound_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bound_box':bound_box}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pk.dump(tracks,f)

            
        return tracks

    def draw_ellipse(self,frame,bound_box,color,track_id=None):
        y2 = int(bound_box[3])
        x_center,_ = get_center_bound_box(bound_box)
        width = get_bound_box_width(bound_box)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color= color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rect_ht = 20
        rect_wd = 40
        rect_x1 = int(x_center-rect_wd//2)
        rect_x2 = int(x_center+rect_wd//2)
        rect_y1 = int(y2-rect_ht//2) + 15
        rect_y2 = int(y2+rect_ht//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,(rect_x1,rect_y1),(rect_x2,rect_y2),color,cv2.FILLED)

            x1_text = rect_x1+11
            if track_id >99:
                x1_text -= 10
            
            cv2.putText(frame,f"{track_id}",(int(x1_text),int(rect_y1)+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

        return frame

    def draw_triangle(self,frame,bound_box,color):
        y= int(bound_box[1])
        x,_ = get_center_bound_box(bound_box)
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])

        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)

        return frame
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
         
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Possesion: {team_1*100 :2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame, f"Team 2 Ball Possesion: {team_2*100 :2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        
        return frame

    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_video_frames = []

        for frame_num , frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                color= player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame,player['bound_box'],color,track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame,player['bound_box'],(0,0,255))
            
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee['bound_box'],(0,255,255))
            
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball['bound_box'],(0,255,0))

            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
