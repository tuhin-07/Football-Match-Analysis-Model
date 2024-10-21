
from utils import get_center_bound_box , measure_distance

class Assignballplayer():
    def __init__(self):
        self.max_ball_player_dist = 70

    def assign_ball_player(self,players,ball_box):
        ball_position = get_center_bound_box(ball_box)

        minimum_dist = 99999
        assigned_player = -1

        for player_id , player in players.items():
            player_box = player['bound_box']

            dist_left = measure_distance((player_box[0],player_box[-1]),ball_position)
            dist_right = measure_distance((player_box[2],player_box[-1]),ball_position)
            distance = min(dist_left,dist_right)

            if distance < self.max_ball_player_dist:
                if distance < minimum_dist:
                    minimum_dist = distance
                    assigned_player = player_id

        return assigned_player
