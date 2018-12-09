import tensorflow as tf
import cv2
import numpy as np
from YOLO_v2.model_darknet19 import darknet
from YOLO_v2.decode import decode
from YOLO_v2.utils import preprocess_image, postprocess, draw_detection
from YOLO_v2.config import anchors


class YOLO_model:
    def __init__(self, sess, model_path, class_path, input_size=(416, 416)):
        self.sess = sess
        self.model_path = model_path

        with open(class_path) as f:
            self.class_names = []
            for l in f.readlines():
                l = l.strip()  # 去掉回车'\n'
                self.class_names.append(l)

        self.input_size = input_size
        # 【1】输入图片进入darknet19网络得到特征图，并进行解码得到：xmin xmax表示的边界框、置信度、类别概率
        self.tf_image = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
        model_output = darknet(self.tf_image)  # darknet19网络输出的特征图
        output_sizes = input_size[0] // 32, input_size[1] // 32  # 特征图尺寸是图片下采样32倍
        self.output_decoded = decode(model_output=model_output, output_sizes=output_sizes,
                                     num_class=len(self.class_names), anchors=anchors)  # 解码

        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def detect(self, image):
        image_shape = image.shape[:2]

        # copy、resize416*416、归一化、在第0维增加存放batchsize维度
        image_cp = preprocess_image(image, self.input_size)

        bboxes, obj_probs, class_probs = self.sess.run(self.output_decoded, feed_dict={self.tf_image: image_cp})

        # 【2】筛选解码后的回归边界框——NMS(post process后期处理)
        bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs, image_shape=image_shape)

        return bboxes, scores, np.array(self.class_names)[class_max_index]
        # img_detection = draw_detection(image, bboxes, scores, class_max_index, class_names)
        # cv2.imshow("detection_results", img_detection)
        # cv2.waitKey(0)


if __name__ == '__main__':
    with tf.Session() as sess:
        yolo = YOLO_model(sess, './yolo2_model/yolo2_coco.ckpt')
        yolo.detect(cv2.imread('./yolo2_data/timg.jpg'))
