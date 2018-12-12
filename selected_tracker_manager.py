from scipy.spatial import distance as dist
from my_tracker.CF_helper import *
from my_tracker.utils import *
import numpy as np
import cv2
import multiprocessing


class SelectedTrackerManager:
    def __init__(self, tracker_type='kcf', use_CF=True, **kwargs):
        self.nextTrackerID = 0
        self.trackers = {}
        self.boxes = {}
        self.counts = {}
        # 检测算法未检测到目标的次数
        self.disappear = {}
        # 目标是否被追踪到
        self.tracked = {}
        # 是否使用相关滤波器进行匹配
        self.use_CF = use_CF
        self.template_denominator = {}
        self.template_numerator = {}
        # self.last_frame = None
        self.filter_size = kwargs.setdefault('filter_size', (128, 128))
        self.fft_gauss_response = np.fft.fft2(gauss_response(self.filter_size[0], self.filter_size[1]))
        # cv2.imshow('g', gauss_response(self.filter_size[0], self.filter_size[1]))

        self.maxDistance = kwargs.setdefault('maxDistance', 150)
        self.maxDisappear = kwargs.setdefault('maxDisappear', 5)
        self.areaThreshold = kwargs.setdefault('areaThreshold', 100)
        self.psrThreshold = kwargs.setdefault('psrThreshold', 8.0)
        self.updatePsr = kwargs.setdefault('updatePsr', 16.0)
        self.areaChangeRatio = kwargs.setdefault('areaChangeRatio', 0.5)

        # 每次追踪的目标匹配为检测到的目标时，由于框的大小不同，可能导致psr也比较小。
        # 该值用于每次追踪的目标匹配为检测到的目标时，可以无视psr的值，更新滤波器的次数。
        self.safeUpdateCount = kwargs.setdefault('safeUpdateCount', 3)

        self.tracker_type = tracker_type

        self.debug = kwargs.setdefault('debug', False)

        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        self.new_tracker = OPENCV_OBJECT_TRACKERS[tracker_type]

        self.pool = multiprocessing.Pool()

    def __register(self, frame, box):
        """通过frame,box,新建并初始化一个tracker"""
        tracker = self.new_tracker()
        tracker.init(frame, convert_to_wh_box(box))
        self.trackers[self.nextTrackerID] = tracker

        self.tracked[self.nextTrackerID] = True
        self.disappear[self.nextTrackerID] = 0
        self.boxes[self.nextTrackerID] = box
        self.counts[self.nextTrackerID] = self.safeUpdateCount
        self.nextTrackerID += 1

    def __renew_tracker(self, trackerID, frame, box):
        """重置tracker"""
        tracker = self.new_tracker()
        tracker.init(frame, convert_to_wh_box(box))
        self.trackers[trackerID] = tracker

        self.tracked[trackerID] = True
        self.disappear[trackerID] = 0
        self.boxes[trackerID] = box
        self.counts[trackerID] = self.safeUpdateCount

    def __deregister(self, trackerID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.trackers[trackerID]
        del self.boxes[trackerID]
        del self.disappear[trackerID]
        del self.tracked[trackerID]
        del self.counts[trackerID]

        if self.use_CF:
            del self.template_denominator[trackerID]
            del self.template_numerator[trackerID]

    def init_manager(self, frame, boxes):
        """初始化manager,注册可能作为追踪目标的trackerID"""
        boxes = discard_boxes(boxes, self.areaThreshold, frame.shape[0], frame.shape[1])

        if len(self.trackers.keys()) == 0:
            for box in boxes:
                self.__register(frame, box)
            # self.last_frame = frame.copy()

        if self.use_CF:
            self.__set_templates(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        return self.__appearing_boxes()

    def update_after_detection(self, frame, boxes):
        """检测算法之后进行的update"""
        boxes = discard_boxes(boxes, self.areaThreshold, frame.shape[0], frame.shape[1])
        # 未检测到目标
        if len(boxes) == 0:
            for ID in list(self.trackers.keys()):
                self.disappear[ID] += 1
                if self.disappear[ID] >= self.maxDisappear:
                    # 超出设定的消失次数，销毁tracker
                    self.__deregister(ID)
            # self.last_frame = frame.copy()
            return self.__appearing_boxes()

        if self.use_CF:
            self.__update_tracker_with_CF(frame, boxes)
        else:
            self.__update_tracker_without_CF(frame, boxes)

        # cv2.waitKey()

        # self.last_frame = frame.copy()
        return self.__appearing_boxes()

    def discard_unselected_IDs(self, selected_IDs):
        for ID in list(self.trackers.keys()):
            if ID not in selected_IDs:
                self.__deregister(ID)

    def __update_tracker_without_CF(self, frame, boxes):
        existing_IDs = list(self.boxes.keys())
        existing_boxes = list(self.boxes.values())
        existing_centroids = compute_centroids(existing_boxes)
        input_centroids = compute_centroids(boxes)

        # 计算目前的跟踪目标的中点和检测目标的中点的距离矩阵
        D = dist.cdist(existing_centroids, input_centroids)
        # 根据每一行上的最小值对行索引进行排序
        rows = D.min(axis=1).argsort()
        # 得到每一行上最小值对应的列索引，并依据rows重新排序
        cols = D.argmin(axis=1)[rows]

        # self.__show_table(D, existing_IDs)

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue

            self.__renew_tracker(existing_IDs[row], frame, boxes[col])

            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        # unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        # 没有对应检测到的目标 的tracker
        for row in unusedRows:
            ID = existing_IDs[row]
            self.disappear[ID] += 1
            if self.disappear[ID] >= self.maxDisappear:
                # 超出设定的消失次数，销毁tracker
                self.__deregister(ID)

    def __update_tracker_with_CF(self, frame, boxes):

        existing_IDs = list(self.boxes.keys())
        existing_boxes = list(self.boxes.values())
        existing_centroids = compute_centroids(existing_boxes)
        input_centroids = compute_centroids(boxes)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算目前的跟踪目标的中点和检测目标的中点的距离矩阵
        D = dist.cdist(existing_centroids, input_centroids)
        rows = range(D.shape[0])
        cols = range(D.shape[1])

        psr_table = np.zeros_like(D)
        psr_with_penalty = np.zeros_like(D)
        for row in rows:
            ID = existing_IDs[row]
            area1 = compute_area(self.boxes[ID])
            for col in cols:
                area2 = compute_area(boxes[col])
                if D[row][col] < self.maxDistance and (1 + self.areaChangeRatio) * area2 > area1 > (
                        1 - self.areaChangeRatio) * area2:
                    # 未超出设定的最大距离，计算psr
                    # 得到追踪目标的相关滤波器H
                    H = self.template_numerator[ID] / self.template_denominator[ID]
                    # cv2.imshow('tar', hi)

                    # 得到检测到的目标f。使用追踪目标的box大小和检测到的目标的中心点，防止resize导致的不同程度的缩放
                    startX, startY, endX, endY = self.boxes[ID]
                    w, h = endX - startX, endY - startY
                    x, y = input_centroids[col]
                    startX, startY, endX, endY = clip_box((x - w // 2, y - h // 2, x + w // 2, y + h // 2),
                                                          gray_frame.shape[0], gray_frame.shape[1])
                    f1 = gray_frame[startY:endY, startX:endX]
                    f1 = cv2.resize(f1, self.filter_size)
                    psr1 = correlation(f1, H)

                    startX, startY, endX, endY = boxes[col]
                    w, h = endX - startX, endY - startY
                    startX, startY, endX, endY = clip_box((x - w // 2, y - h // 2, x + w // 2, y + h // 2),
                                                          gray_frame.shape[0], gray_frame.shape[1])

                    f2 = gray_frame[startY:endY, startX:endX]
                    f2 = cv2.resize(f2, self.filter_size)
                    psr2 = correlation(f2, H)

                    psr = max(psr1, psr2)
                    psr_table[row][col] = psr

                    # 加入距离的惩罚项
                    penalty = D[row][col] / self.maxDistance * 5
                    psr_with_penalty[row][col] = psr - penalty

                    if self.debug:
                        # f = f1 if psr == psr1 else f2
                        cv2.imshow('f1', f1)
                        cv2.imshow('f2', f2)
                        print('id:{}, psr1:{},psr2:{},psr:{}, psr_with_penalty:{}, '
                              'cen:{}, tar_cen:{}'.format(ID, psr1, psr2, psr, psr - penalty, existing_centroids[row],
                                                          input_centroids[col]))
                        cv2.waitKey()

        # 根据每一行上的最大值对行索引进行排序
        rows = np.flip(psr_with_penalty.max(axis=1).argsort())
        # 得到每一行上最大值对应的列索引，并依据rows重新排序
        cols = psr_with_penalty.argmax(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if psr_table[row, col] == 0 or psr_with_penalty[row, col] < self.psrThreshold:
                continue

            ID = existing_IDs[row]
            self.__renew_tracker(ID, frame, boxes[col])
            self.__update_template(ID, gray_frame)

            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        # unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        for row in unusedRows:
            ID = existing_IDs[row]
            self.disappear[ID] += 1
            if self.disappear[ID] >= self.maxDisappear:
                # 超出设定的消失次数，销毁tracker
                self.__deregister(ID)

        # self.__set_templates(gray_frame)

    def update_trackers(self, frame):
        """使用tracker追踪目标"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (ID, tracker) in self.trackers.items():
            (success, box) = tracker.update(frame)
            # print(success, box)
            if success:
                self.boxes[ID] = clip_box(convert_from_wh_box(box), frame.shape[0], frame.shape[1])
                self.tracked[ID] = True

                if self.use_CF:
                    if self.counts[ID] == 0:
                        # 旧的滤波器
                        H = self.template_numerator[ID] / self.template_denominator[ID]
                        # 追踪目标在frame中的位置
                        startX, startY, endX, endY = self.boxes[ID]
                        f = gray_frame[startY:endY, startX:endX]
                        f = cv2.resize(f, self.filter_size)
                        psr = correlation(f, H)

                        # 目标没有被遮挡才更新滤波器
                        if psr > self.updatePsr:
                            self.__update_template(ID, gray_frame)

                        if self.debug:
                            print('psr:{}'.format(psr))
                    else:
                        # 由于刚匹配到当前目标，无视psr，更新滤波器
                        self.__update_template(ID, gray_frame)
                        self.counts[ID] -= 1

                        if self.debug:
                            print('skip psr')
            else:
                # 追踪失败
                self.tracked[ID] = False
        # self.last_frame = frame.copy()
        return self.__appearing_boxes()

    def __appearing_boxes(self):
        boxes = {}
        for ID, tracked in self.tracked.items():
            if tracked:
                boxes[ID] = self.boxes[ID]
        return boxes

    def show_info(self, mes_title):
        print('--------------------{}--------------------'.format(mes_title))
        # print('trackers:{}'.format(self.trackers))
        print('tracked:{}'.format(self.tracked))
        print('boxes:{}'.format(self.boxes))
        print('disappear:{}'.format(self.disappear))

    def __update_template(self, ID, gray_frame, rate=0.125):
        # 更新滤波器模板
        startX, startY, endX, endY = self.boxes[ID]
        hi = gray_frame[startY:endY, startX:endX]
        hi = cv2.resize(hi, self.filter_size)
        numerator, denominator = correlation_filter(hi, self.fft_gauss_response)

        self.template_numerator[ID] = rate * numerator + (1 - rate) * self.template_numerator[ID]
        self.template_denominator[ID] = rate * denominator + (1 - rate) * self.template_denominator[ID]

    def __set_templates(self, gray_frame):
        # 对要追踪的id
        for ID in self.tracked.keys():
            # 对于追踪到的目标构建相关滤波模板
            # 没有追踪到的目标还需要依靠现有的滤波器来寻找目标
            if self.tracked[ID]:
                startX, startY, endX, endY = self.boxes[ID]
                hi = gray_frame[startY:endY, startX:endX]
                hi = cv2.resize(hi, self.filter_size)
                self.template_numerator[ID], self.template_denominator[ID] = correlation_filter(hi,
                                                                                                self.fft_gauss_response)
