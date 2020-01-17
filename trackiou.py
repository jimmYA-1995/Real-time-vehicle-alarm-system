import time
import numpy as np

class TrackID():
    def __init__(self):
        self.n = 0
    def get_num(self):
        self.n += 1
        return self.n

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    modified from https://github.com/bochinski/iou-tracker/blob/master/iou_tracker.py
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def get_d_bbox_size(track, d=10):
    track_len = len(track)
    if track_len < d:
        track = [0] * (d - track_len) + track
    elif track_len > d:
        track = track[-d:]
    # print("track in get bbox size: ", track)
    return np.array([0 if isinstance(p, int) or p is None else (p[2]-p[0]) * (p[3]-p[1]) / (600 * 800) for p in track])

def test_notice(bbox, gaze_data, grid=None):
    (startX, startY, endX, endY) = bbox
    if grid is True:
        k = 50
        startX, startY, endX, endY = startX - startX%k, startY - startY%k, k * (endX//k+1), k * (endY//k+1)
    for p in gaze_data:
        if p[0]>= startX and p[0]<=endX and p[1]>=startY and p[1]<=endY:
            return True
    else:
        return False

class Tracker():
    def __init__(self,items, collide_interpreter):
        super(Tracker,self).__init__()
        self.tracks_active = []
        self.tracks_finished = []
        self.sliding_window = items['sliding_window']
        self.sigma_iou = items['sigma_iou']
        self.lifespan = items['lifespan']
        self.iou_decay = 0.6
        self.track_id = TrackID()

        self.interpreter = collide_interpreter
        self.collide_input = self.interpreter.tensor(19)
        self.collide_output = self.interpreter.tensor(17)
        self.time_acc = 0.

    def track(self,detection, gaze_data=None):
        dets = detection
        updated_tracks = []

        for track in self.tracks_active:
            if len(dets) > 0:
                bbox_length = len(track['bboxes'])
                sigma_iou = self.sigma_iou
                for sliding in range(-1,-self.sliding_window-1,-1):
                    # 若track的bbox長度比sliding長，加上該bbox不為None
                    if bbox_length >= abs(sliding) and track['bboxes'][sliding] is not None:
                        # print(track['bboxes'][sliding])
                        best_match = max(dets, key=lambda x: iou(track['bboxes'][sliding], x['bbox']))
                        if iou(track['bboxes'][sliding], best_match['bbox']) >= self.sigma_iou and \
                            track['label'] == best_match['label']:
                            track['bboxes'].append(best_match['bbox']) # FIXME - yct: wrong data structure
                            track['lifespan'] = self.lifespan
                            if track['noticed'] is not True and gaze_data is not None:
                                track['noticed'] = test_notice(best_match['bbox'], gaze_data, True)
                            updated_tracks.append(track)

                            dets = list(filter(lambda x: not all(x['bbox'] == best_match['bbox']), dets))
                            break

                    sigma_iou = sigma_iou * self.iou_decay
            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                track['lifespan'] -= 1
                if track['lifespan'] > 0:
                    track['bboxes'].append(None)
                    updated_tracks.append(track)

        # create new track & assign a trackID
        new_tracks = [{'bboxes': [det['bbox']], 'label': det['label'], 'trackid': self.track_id.get_num(), 'lifespan': self.lifespan, 'noticed': None, 'dangerous': None} for det in dets]
        self.tracks_active = updated_tracks + new_tracks

        self.predict_collide()

    def get_current(self):
        return [{'bbox':track['bboxes'][-1],'label': track['label'],'trackid': track['trackid'], 'noticed': track['noticed'], 'dangerous': track['dangerous']} for track in self.tracks_active if track['bboxes'][-1] is not None]

    def predict_collide(self):
        start = time.monotonic()
        for track in self.tracks_active:
            if track['bboxes'][-1] is not None and track['dangerous'] is not True:
                b_sizes = get_d_bbox_size(track['bboxes']).astype(np.float32)
                # tensor ID, in this case, 19 and 17, depends on tflite model
                self.interpreter.set_tensor(19, [b_sizes])
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(17)[0]
                result = np.argmax(output)
                if result == 1:
                    print("detect dangerous object!!!!")
                    track['dangerous'] = True
        
        print("collide_prediction: {:2f} ms".format(time.monotonic() - start))
