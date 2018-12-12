from my_tracker.selected_tracker_manager import SelectedTrackerManager
# from my_tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from YOLO_v2.yolo_model import YOLO_model
from my_tracker.utils import pop_up_box
import imutils
import time
import cv2
import tensorflow as tf


class SelectedTracker:
    def __init__(self, detection_model):
        self.detection_model = detection_model

    def track(self, input_vedio=None, output_vedio=None, frame_width=500, skip_frames=20, confidence=0.3,
              verbose=False, use_CF=True):
        if input_vedio is None:
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(2.0)
        else:
            print("[INFO] opening video file...")
            vs = cv2.VideoCapture(input_vedio)

        # initialize the frame dimensions
        W = None
        H = None

        # 初始化视频输出对象
        writer = None

        # 初始要追踪的目标ID
        selected_IDs = []

        tm = SelectedTrackerManager(use_CF=use_CF, debug=verbose)
        # trackableObjects = {}

        # initialize the total number of frames processed thus far
        totalFrames = 0

        # start the frames per second throughput estimator
        fps = None

        tracking = False

        # loop over frames from the video stream
        while True:
            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream
            frame = vs.read()
            frame = frame[1] if input_vedio is not None else frame

            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if input_vedio is not None and frame is None:
                break

            # resize the frame to have a maximum width of frame_width
            frame = imutils.resize(frame, width=frame_width)

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                self.detection_model.detect(frame)

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if output_vedio is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                writer = cv2.VideoWriter(output_vedio, fourcc, 30,
                                         (W, H), True)

            if tracking:
                # check to see if we should run a more computationally expensive
                # object detection method to aid our tracker
                if totalFrames % skip_frames == 0:
                    raw_boxes, scores, classes = self.detection_model.detect(frame)

                    # loop over the detections
                    boxes = []
                    for i in range(len(scores)):
                        # filter out weak detections by requiring a minimum onfidence
                        if scores[i] > confidence:
                            # if the class label is not a person, ignore it
                            if classes[i] == "person" or classes[i] == 'car':
                                boxes.append(raw_boxes[i])

                    if totalFrames == 0:
                        rects = tm.init_manager(frame, boxes)
                    else:
                        rects = tm.update_after_detection(frame, boxes)

                # otherwise, we should utilize our object *trackers* rather than
                # object *detectors* to obtain a higher frame processing throughput
                else:
                    # loop over the trackers
                    rects = tm.update_trackers(frame)

                for rect in rects.values():
                    (startX, startY, endX, endY) = rect
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)

                # loop over the tracked objects
                for (objectID, box) in rects.items():
                    (startX, startY, endX, endY) = box
                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    centroid = (cX, cY)
                    # # check to see if a trackable object exists for the current
                    # # object ID
                    # to = trackableObjects.get(objectID, None)
                    #
                    # # if there is no existing trackable object, create one
                    # if to is None:
                    #     trackableObjects[objectID] = TrackableObject(objectID, box)
                    # else:
                    #     to.boxes.append(box)

                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)

                if totalFrames == 0:
                    cv2.imshow("Frame", frame)
                    selected_IDs.append(pop_up_box())
                    tm.discard_unselected_IDs(selected_IDs)
                    fps = FPS().start()

                # increment the total number of frames processed thus far and
                # then update the FPS counter
                totalFrames += 1
                fps.update()

                if verbose:
                    tm.show_info('Frame{}'.format(totalFrames))

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # show the output frame
            cv2.imshow("Frame", frame)
            if not tracking:
                key = cv2.waitKey(20) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            elif key == ord("s"):
                tracking = True

            # if not tracking:
            #     cv2.waitKey(10)

        # stop the timer and display FPS information
        if fps is not None:
            fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()

        # if we are not using a video file, stop the camera video stream
        if input_vedio is None:
            vs.stop()

        # otherwise, release the video file pointer
        else:
            vs.release()

        # close any open windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    size = 512
    with tf.Session() as sess:
        yolo = YOLO_model(sess, '../YOLO_v2/yolo2_model/yolo2_coco.ckpt', '../YOLO_v2/yolo2_data/coco_classes.txt',
                          input_size=(size, size))
        selected_tracker = SelectedTracker(yolo)
        # print(yolo.detect(cv2.imread('../YOLO_v2/yolo2_data/timg.jpg')))
        selected_tracker.track('./videos/girl.mp4', './output/girl.mp4', skip_frames=20, verbose=True,
                               use_CF=True,  frame_width=size)
