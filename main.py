from utils import read_video , save_video
from tracker import Tracker
from team_assign import TeamAssigner
from assign_ball_player import Assignballplayer
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance import SpeedAndDistance

def main():
    # read video frames
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    # initilize class
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_track(video_frames,read_from_stub=True,stub_path="stubs/track_stubs.pkl")

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Transformer
    view_tranformer = ViewTransformer()
    view_tranformer.add_transform_position_to_tracks(tracks)
    
    # Interpolate Ball Positions
    tracks['ball'] = tracker.ball_interpolation(tracks['ball'])

    # Speed and Distance
    speed_and_distance_estimator = SpeedAndDistance()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id , track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],track['bound_box'],player_id)

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team]


     # Assign Ball Aquisition
    player_assigner = Assignballplayer()
    team_ball_control = []
    for frame_num , player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bound_box']
        assigned_player = player_assigner.assign_ball_player(player_track,ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
        
    team_ball_control = np.array(team_ball_control)


    # Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)

    # Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    # save video frames
    save_video(output_video_frames,'output_videos/output_video.avi')

if __name__=='__main__':
    main()