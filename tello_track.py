# coding=utf-8
import threading
import time
from Queue import Queue
import imutils
import numpy as np
import cv2

import tello
from darknet import darknet
from my_tracker.selected_tracker_manager import SelectedTrackerManager
from my_tracker.utils import pop_up_box


class TelloTracker:
    def __init__(self, detect_model, tello, frame_width=512, skip_frames=20, confidence=0.4, output_video=None):
        self.detect_model = detect_model
        self.tello = tello
        self.frame_width = frame_width
        self.skip_frames = skip_frames
        self.confidence = confidence
        self.output_video = output_video

        self.frame = None
        self.init_area = None
        self.frame_pool = Queue()
        self.stopEvent = threading.Event()

        # meters
        self.distance = 0.5
        self.degree = 10

        self.vedio_thread = threading.Thread(target=self._videoLoop, args=())
        self.vedio_thread.start()

        self.sending_command_thread = threading.Thread(target=self._sendingCommand)

    def _get_pooled_frame(self):
        if self.frame_pool.qsize() != 0:
            return self.frame_pool.get()

    # def show_frame(self, win_name):
    #     if self.frame is not None:
    #         cv2.imshow(win_name, self.frame)

    def _videoLoop(self):
        """
        a thread used to read frame from tello object
        """
        try:
            time.sleep(0.5)
            self.sending_command_thread.start()
            # self.sending_command_thread.start()
            while not self.stopEvent.is_set():
                # read the frame for GUI show
                # self.frame = np.array(self.tello.read()).astype(int)
                frame = self.tello.read()
                if frame is None or frame.size == 0:
                    continue
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_pool.put(self.frame)
                time.sleep(0.05)
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def _close(self):
        """
        set the stop event, cleanup the camera
        """
        print("[INFO] closing...")
        self.stopEvent.set()
        del self.tello

    def track(self):
        # 初始要追踪的目标ID
        selected_IDs = []
        tm = SelectedTrackerManager(filter_size=(64, 64))
        tracking = False
        width, height = None, None
        writer = None
        total_track_frames = 0

        # loop over frames from the video stream
        while True:
            # grab the next frame
            frame = self._get_pooled_frame()
            if frame is None or frame.size == 0:
                continue

            # resize the frame to have a maximum width of frame_width
            frame = imutils.resize(frame, width=self.frame_width)

            # if the frame dimensions are empty, set them
            if width is None or height is None:
                (height, width) = frame.shape[:2]
                # initialize the detect model so that there won't be much delay when detecting later
                self.detect_model.performDetect(None, initOnly=True)

            # if we are supposed to be writing a video to disk, initialize the writer
            if self.output_video is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                writer = cv2.VideoWriter(self.output_video, fourcc, 30,
                                         (width, height), True)

            if tracking:
                # check to see if we should run a more computationally expensive
                # object detection method to aid our tracker
                if total_track_frames % self.skip_frames == 0:
                    raw_boxes, scores, classes = self.detect_model.performDetect(frame)
                    # print(raw_boxes, scores, classes)

                    # loop over the detections
                    boxes = []
                    for i in range(len(scores)):
                        # filter out weak detections by requiring a minimum onfidence
                        if scores[i] > self.confidence:
                            # if the class label is not a person, ignore it
                            if classes[i] == "person" or classes[i] == 'laptop':
                                boxes.append(raw_boxes[i])

                    if total_track_frames == 0:
                        rects = tm.init_manager(frame, boxes)
                    else:
                        # match tracking objects and detected objects
                        rects = tm.update_after_detection(frame, boxes)
                else:
                    # otherwise, we should utilize our object *trackers* rather than
                    # object *detectors* to obtain a higher frame processing throughput
                    rects = tm.update_trackers(frame)

                # draw rectangles of corresponding objects
                for rect in rects.values():
                    (startX, startY, endX, endY) = rect
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # draw both the ID of the object and the centroid of the object on the output frame
                for (objectID, box) in rects.items():
                    (startX, startY, endX, endY) = box
                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    centroid = (cX, cY)

                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)

                # select an object to track
                if total_track_frames == 0:
                    cv2.imshow('frame', frame)
                    selected_IDs.append(pop_up_box())
                    tm.discard_unselected_IDs(selected_IDs)
                # decide what action to choose
                elif total_track_frames % 10 == 0:
                    if len(rects) != 0:
                        # there is only one box
                        box = rects.values()[0]
                        (startX, startY, endX, endY) = box
                        area = (endX - startX) * (endY - startY)
                        cX = (startX + endX) / 2.0
                        cY = (startY + endY) / 2.0
                        x_ratio = cX / width
                        y_ratio = cY / height
                        # print "area:%f" % area
                        # print "x_ratio:%f,y_ratio:%f" % (x_ratio, y_ratio)
                        if x_ratio < 0.4:
                            self.tello.rotate_ccw(self.degree)
                        elif x_ratio > 0.6:
                            self.tello.rotate_cw(self.degree)

                        if y_ratio < 0.4:
                            self.tello.move_up(self.distance)
                        elif y_ratio > 0.6:
                            self.tello.move_down(self.distance)

                        if self.init_area is None:
                            self.init_area = area
                        else:
                            area_ratio = area / self.init_area
                            if area_ratio < 0.8:
                                self.tello.move_forward(self.distance)
                            elif area_ratio > 1.2:
                                self.tello.move_backward(self.distance)

                # increment the total number of frames processed thus far
                total_track_frames += 1

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # show the output frame
            cv2.imshow('frame', frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                tracking = True
            elif key == ord("t"):
                self.tello.takeoff()
            elif key == ord("u"):
                self.tello.move_up(self.distance)
            elif key == ord("d"):
                self.tello.move_down(self.distance)
            elif key == ord("l"):
                self.tello.land()
            # elif key == ord("f"):
            #     self.tello.move_forward(self.distance)
            # elif key == ord("b"):
            #     self.tello.move_backward(self.distance)

            # if key == ord("q"):
            #     break
            # elif key == ord("t"):
            #     self.tello.takeoff()
            # elif key == ord("u"):
            #     self.tello.move_up(0.5)
            # elif key == ord("d"):
            #     self.tello.move_down(0.5)
            # elif key == ord("l"):
            #     self.tello.land()
            # elif key == ord("s"):
            #     self.tello.get_speed()
            # elif key == ord("f"):
            #     self.tello.move_forward(0.5)
            # elif key == ord("b"):
            #     self.tello.move_backward(0.5)

        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()

        # close any open windows
        cv2.destroyAllWindows()
        self._close()

    def _sendingCommand(self):
        """
        start a while loop that sends 'command' to tello every 60 second
        """

        while True:
            self.tello.get_battery()
            # self.tello.send_command('command')
            time.sleep(30)


if __name__ == '__main__':
    # detect_model = darknet
    tracker = TelloTracker(darknet, tello.Tello('', 8889))

    tracker.track()
    # while True:
    #     tracker.show_frame('frame')
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord("q"):
    #         tracker.close()
    #         break
