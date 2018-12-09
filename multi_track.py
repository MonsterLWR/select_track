from my_tracker.tracker_manager import TrackerManager
from my_tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from YOLO_v2.yolo_model import YOLO_model
import imutils
import time
import cv2
import tensorflow as tf


class Multi_Tracker:
    def __init__(self, detection_model):
        self.detection_model = detection_model

    def track(self, input_vedio=None, output_vedio=None, frame_width=1024, skip_frames=30, confidence=0.4,
              verbose=False, draw_id=True, use_CF=True):
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

        tm = TrackerManager(use_CF=use_CF)
        trackableObjects = {}

        # initialize the total number of frames processed thus far
        totalFrames = 0

        # start the frames per second throughput estimator
        fps = FPS().start()

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

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if output_vedio is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                writer = cv2.VideoWriter(output_vedio, fourcc, 30,
                                         (W, H), True)

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
                        if classes[i] != "person":
                            continue
                        boxes.append(raw_boxes[i])
                rects = tm.init_trackers(frame, boxes)

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
                # check to see if a trackable object exists for the current
                # object ID
                to = trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    trackableObjects[objectID] = TrackableObject(objectID, box)
                else:
                    to.boxes.append(box)

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                if draw_id:
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            fps.update()

            if verbose:
                tm.show_info('Frame{}'.format(totalFrames))

        # stop the timer and display FPS information
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
    with tf.Session() as sess:
        yolo = YOLO_model(sess, '../YOLO_v2/yolo2_model/yolo2_coco.ckpt', '../YOLO_v2/yolo2_data/coco_classes.txt',
                          input_size=(512, 512))
        multi_tracker = Multi_Tracker(yolo)
        # print(yolo.detect(cv2.imread('../YOLO_v2/yolo2_data/timg.jpg')))
        multi_tracker.track('./videos/soccer_02.mp4', './output/soccer.mp4', skip_frames=20, verbose=True,
                            draw_id=True, use_CF=False)
