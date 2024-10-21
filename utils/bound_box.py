def get_center_bound_box(bound_box):
    x1,y1,x2,y2 = bound_box
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bound_box_width(bound_box):
    return bound_box[2]-bound_box[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)