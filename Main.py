import cv2
from darknet import darknet
from my_tracker.selected_track import SelectedTracker

if __name__ == '__main__':
    size = 512
    input = 'data/videos/1.mp4'
    output = 'data/output/heihei.mp4'

    # img = cv2.imread("data/timg.jpg")
    # print(darknet.performDetect(img))

    selected_tracker = SelectedTracker(darknet)
    selected_tracker.track(input, output, skip_frames=20, verbose=False,
                           use_CF=True, frame_width=size)
