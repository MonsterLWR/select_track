import numpy as np
import tkinter


def compute_centroids(boxes):
    centroids = np.zeros((len(boxes), 2), dtype="int")
    for (i, (startX, startY, endX, endY)) in enumerate(boxes):
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        centroids[i] = (cX, cY)
    return centroids


def compute_area(box):
    startX, startY, endX, endY = box
    height = endY - startY
    width = endX - startX
    return height * width


def convert_to_wh_box(box):
    (startX, startY, endX, endY) = box
    w = endX - startX
    h = endY - startY
    return startX, startY, w, h


def convert_from_wh_box(box):
    startX, startY, w, h = box
    startX = int(startX)
    startY = int(startY)
    endX = int(startX + w)
    endY = int(startY + h)
    return startX, startY, endX, endY


def discard_boxes(boxes, area_threshhold, height, width):
    """删除面积过小的边框"""
    returned_boxes = _discard_border_boxes(boxes, height, width)
    returned_boxes = _discard_small_boxes(returned_boxes, area_threshhold)
    return returned_boxes


def _discard_small_boxes(boxes, area_threshhold):
    new_boxes = []
    for box in boxes:
        startX, startY, endX, endY = box
        area = (endY - startY) * (endX - startX)
        if area > area_threshhold:
            new_boxes.append((startX, startY, endX, endY))

    return new_boxes


def _discard_border_boxes(boxes, height, width):
    """去除检测到的边框超过视频边缘（height, width）的box"""
    new_boxes = []
    for box in boxes:
        startX, startY, endX, endY = box
        startY = np.clip(startY, 0, height)
        endY = np.clip(endY, 0, height)
        startX = np.clip(startX, 0, width)
        endX = np.clip(endX, 0, width)
        # if startY == 0 or startX == 0 or endX == width or endY == height:
        #     continue
        # else:
        new_boxes.append((startX, startY, endX, endY))
    return new_boxes


def clip_box(box, height, width):
    """将边框超出视频边缘（height, width）的box进行修正，使边框在视频边缘之内"""
    startX, startY, endX, endY = box
    startY = np.clip(startY, 0, height)
    endY = np.clip(endY, 0, height)
    startX = np.clip(startX, 0, width)
    endX = np.clip(endX, 0, width)

    if endX - startX == 0:
        if startX > 0:
            startX -= 1
        else:
            endX += 1
    if endY - startY == 0:
        if startY > 0:
            startY -= 1
        else:
            endY += 1

    return startX, startY, endX, endY


def pop_up_box():
    """
    使用tkinter弹出输入框输入数字, 具有确定输入和清除功能, 可在函数内直接调用num(文本框的值)使用
    """

    def inputint():
        nonlocal nums
        try:
            nums = [int(num) for num in var.get().strip().split(',')]
            # num = int(var.get().strip())
            root.quit()
            root.destroy()
            # quit = threading.Thread(target=root.quit)
            # quit.start()
        except:
            num = 'Not a valid integer.'

    def inputclear():
        nonlocal nums
        var.set('')
        nums = []

    # num = 0
    nums = []
    root = tkinter.Tk(className='请输入追踪目标的ID')  # 弹出框框名
    root.geometry('270x60')  # 设置弹出框的大小 w x h

    var = tkinter.StringVar()  # 这即是输入框中的内容
    var.set('Content of var')  # 通过var.get()/var.set() 来 获取/设置var的值
    entry1 = tkinter.Entry(root, textvariable=var)  # 设置"文本变量"为var
    entry1.pack()  # 将entry"打上去"
    btn1 = tkinter.Button(root, text='Input', command=inputint)  # 按下此按钮(Input), 触发inputint函数
    btn2 = tkinter.Button(root, text='Clear', command=inputclear)  # 按下此按钮(Clear), 触发inputclear函数

    # 按钮定位
    btn2.pack(side='right')
    btn1.pack(side='right')

    # 上述完成之后, 开始真正弹出弹出框
    root.mainloop()

    return nums
