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
        # 检测算法未检测到目标的次数
        self.disappear = {}
        # 目标是否被追踪到
        self.tracked = {}
        # 是否使用相关滤波器进行匹配
        self.use_CF = use_CF
        self.template = {}
        self.last_frame = None
        self.filter_size = kwargs.setdefault('filter_size', (128, 128))
        self.fft_gauss_response = np.fft.fft2(gauss_response(self.filter_size[0], self.filter_size[1]))
        # cv2.imshow('g', gauss_response(self.filter_size[0], self.filter_size[1]))

        self.maxDistance = kwargs.setdefault('maxDistance', 100)
        self.maxDisappear = kwargs.setdefault('maxDisappear', 3)
        self.areaThreshold = kwargs.setdefault('areaThreshold', 0)

        self.tracker_type = tracker_type

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
        self.nextTrackerID += 1

    def __renew_tracker(self, trackerID, frame, box):
        """重置tracker"""
        tracker = self.new_tracker()
        tracker.init(frame, convert_to_wh_box(box))
        self.trackers[trackerID] = tracker

        self.tracked[trackerID] = True
        self.disappear[trackerID] = 0
        self.boxes[trackerID] = box

    def __deregister(self, trackerID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.trackers[trackerID]
        del self.boxes[trackerID]
        del self.disappear[trackerID]
        del self.tracked[trackerID]

    def __compute_centroids(self, boxes):
        centroids = np.zeros((len(boxes), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(boxes):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            centroids[i] = (cX, cY)
        return centroids

    def init_manager(self, frame, boxes):
        """初始化manager,注册可能作为追踪目标的trackerID"""
        boxes = discard_boxes(boxes, self.areaThreshold, frame.shape[0], frame.shape[1])

        if len(self.trackers.keys()) == 0:
            for box in boxes:
                self.__register(frame, box)
            self.last_frame = frame.copy()

        if self.use_CF:
            self.__prepare_templates(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

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
            self.last_frame = frame.copy()
            return self.__appearing_boxes()

        if self.use_CF:
            self.__update_tracker_with_CF(frame, boxes)
        else:
            self.__update_tracker_without_CF(frame, boxes)

        # cv2.waitKey()

        self.last_frame = frame.copy()
        return self.__appearing_boxes()

    def __show_table(self, table, existing_IDs):
        h, w = table.shape
        for i in range(h):
            print('id:{:2}'.format(existing_IDs[i]), end=' ')
            for j in range(w):
                print('{:25}'.format(table[i][j]), end=' ')
            print()

    def discard_unselected_IDs(self, selected_IDs):
        for ID in list(self.trackers.keys()):
            if ID not in selected_IDs:
                self.__deregister(ID)

    def __update_tracker_without_CF(self, frame, boxes):
        existing_IDs = list(self.boxes.keys())
        existing_boxes = list(self.boxes.values())
        existing_centroids = self.__compute_centroids(existing_boxes)
        input_centroids = self.__compute_centroids(boxes)

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
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

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
        existing_centroids = self.__compute_centroids(existing_boxes)
        input_centroids = self.__compute_centroids(boxes)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算目前的跟踪目标的中点和检测目标的中点的距离矩阵
        D = dist.cdist(existing_centroids, input_centroids)
        rows = range(D.shape[0])
        cols = range(D.shape[1])

        psr_table = np.zeros_like(D)
        for row in rows:
            for col in cols:
                if D[row][col] > self.maxDistance:
                    # 超出了设定的距离
                    psr_table[row][col] = 0
                else:
                    # 未超出设定的最大距离，计算psr
                    # 得到检测到的目标对应的相关滤波器H
                    startX, startY, endX, endY = boxes[col]
                    hi = gray_frame[startY:endY, startX:endX]
                    hi = cv2.resize(hi, self.filter_size)
                    H = correlation_filter(hi, self.fft_gauss_response)
                    # cv2.imshow('tar', hi)

                    # 得到追踪的目标f
                    w, h = endX - startX, endY - startY
                    x, y = existing_centroids[row]
                    startX, startY, endX, endY = clip_box((x - w // 2, y - h // 2, x + w // 2, y + h // 2),
                                                          self.last_frame.shape[0], self.last_frame.shape[1])
                    f = gray_last_frame[startY:endY, startX:endX]
                    f = cv2.resize(f, self.filter_size)
                    # cv2.imshow('ori', f)

                    psr = correlation(f, H)
                    psr_table[row][col] = psr
                    # print(
                    #     'id:{}, psr:{}, cen:{}, tar_cen:{}'.format(existing_IDs[row], psr, existing_centroids[row],
                    #                                                input_centroids[col]))
                    # cv2.waitKey()

        print(psr_table)
        # cv2.waitKey()
        # D = (self.maxDistance - D) / self.maxDistance * 3
        # print(D)
        # final_table = D + psr_table

        # 根据每一行上的最大值对行索引进行排序
        rows = np.flip(psr_table.max(axis=1).argsort())
        # 得到每一行上最大值对应的列索引，并依据rows重新排序
        cols = psr_table.argmax(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if psr_table[row, col] == 0:
                continue

            self.__renew_tracker(existing_IDs[row], frame, boxes[col])

            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        # 没有对应tracker的 检测到的目标，新建tracker
        for col in unusedCols:
            self.__register(frame, boxes[col])
        # 没有对应检测到的目标 的tracker
        for row in unusedRows:
            ID = existing_IDs[row]
            self.disappear[ID] += 1
            if self.disappear[ID] >= self.maxDisappear:
                # 超出设定的消失次数，销毁tracker
                self.__deregister(ID)

    # def __init_tracker_with_CF(self, frame, boxes):
    #
    #     existing_IDs = list(self.boxes.keys())
    #     existing_boxes = list(self.boxes.values())
    #     existing_centroids = self.__compute_centroids(existing_boxes)
    #     input_centroids = self.__compute_centroids(boxes)
    #
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     gray_last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
    #
    #     # 计算目前的跟踪目标的中点和检测目标的中点的距离矩阵
    #     D = dist.cdist(existing_centroids, input_centroids)
    #     rows = range(D.shape[0])
    #     cols = range(D.shape[1])
    #
    #     psr_table = np.zeros_like(D)
    #     for row in rows:
    #         for col in cols:
    #             if D[row][col] > self.maxDistance:
    #                 # 超出了设定的距离
    #                 psr_table[row][col] = 0
    #             else:
    #                 # 未超出设定的最大距离，计算psr
    #                 # 得到检测到的目标对应的相关滤波器H
    #                 startX, startY, endX, endY = boxes[col]
    #                 hi = gray_frame[startY:endY, startX:endX]
    #                 hi = cv2.resize(hi, self.filter_size)
    #                 H = correlation_filter(hi, self.fft_gauss_response)
    #                 # cv2.imshow('tar', hi)
    #
    #                 # 得到追踪的目标f
    #                 w, h = endX - startX, endY - startY
    #                 x, y = existing_centroids[row]
    #                 startX, startY, endX, endY = clip_box((x - w // 2, y - h // 2, x + w // 2, y + h // 2),
    #                                                       self.last_frame.shape[0], self.last_frame.shape[1])
    #                 f = gray_last_frame[startY:endY, startX:endX]
    #                 f = cv2.resize(f, self.filter_size)
    #                 # cv2.imshow('ori', f)
    #
    #                 psr = correlation(f, H)
    #                 psr_table[row][col] = psr
    #                 # print(
    #                 #     'id:{}, psr:{}, cen:{}, tar_cen:{}'.format(existing_IDs[row], psr, existing_centroids[row],
    #                 #                                                input_centroids[col]))
    #                 # cv2.waitKey()
    #
    #     print(psr_table)
    #     # cv2.waitKey()
    #     # D = (self.maxDistance - D) / self.maxDistance * 3
    #     # print(D)
    #     # final_table = D + psr_table
    #
    #     # 根据每一行上的最大值对行索引进行排序
    #     rows = np.flip(psr_table.max(axis=1).argsort())
    #     # 得到每一行上最大值对应的列索引，并依据rows重新排序
    #     cols = psr_table.argmax(axis=1)[rows]
    #
    #     usedRows = set()
    #     usedCols = set()
    #
    #     for (row, col) in zip(rows, cols):
    #         if row in usedRows or col in usedCols:
    #             continue
    #         if psr_table[row, col] == 0:
    #             continue
    #
    #         self.__renew_tracker(existing_IDs[row], frame, boxes[col])
    #
    #         usedRows.add(row)
    #         usedCols.add(col)
    #
    #     unusedRows = set(range(0, D.shape[0])).difference(usedRows)
    #     unusedCols = set(range(0, D.shape[1])).difference(usedCols)
    #
    #     # 没有对应tracker的 检测到的目标，新建tracker
    #     for col in unusedCols:
    #         self.__register(frame, boxes[col])
    #     # 没有对应检测到的目标 的tracker
    #     for row in unusedRows:
    #         ID = existing_IDs[row]
    #         self.disappear[ID] += 1
    #         if self.disappear[ID] >= self.maxDisappear:
    #             # 超出设定的消失次数，销毁tracker
    #             self.__deregister(ID)

    def update_trackers(self, frame):
        for (ID, tracker) in self.trackers.items():
            (success, box) = tracker.update(frame)
            # print(success, box)
            if success:
                self.boxes[ID] = clip_box(convert_from_wh_box(box), self.last_frame.shape[0],
                                          self.last_frame.shape[1])
                self.tracked[ID] = True
            else:
                self.tracked[ID] = False
        self.last_frame = frame.copy()
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

    def __prepare_templates(self, gray_frame):
        # 对要追踪的id
        for id in self.boxes.keys():
            # 如果还没有构建相关滤波模板
            if id not in self.template.keys():
                startX, startY, endX, endY = self.boxes[id]
                hi = gray_frame[startY:endY, startX:endX]
                hi = cv2.resize(hi, self.filter_size)
                self.template[id] = correlation_filter(hi, self.fft_gauss_response)
