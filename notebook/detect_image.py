import os, sys
import tensorflow as tf
from moxing.framework import file
import cv2
import time

# 执行参数 python notebook/detect_image.py --image notebook/test.jpg --min_score 0.2 --show_image true --show_box_label true --image_size 1024 --version V0007 --h5 train_mask_rcnn.h5
# 外部参数配置
tf.flags.DEFINE_string('image', 'dataset/test.jpg', 'dataset file.')
tf.flags.DEFINE_float('min_score', 0.2, 'int of show minimum score.')
tf.flags.DEFINE_boolean('show_image', True, 'bool of show picture beas64.')
tf.flags.DEFINE_boolean('show_box_label', False, 'bool of show identification border labels.')
tf.flags.DEFINE_integer('input_size', 1024, 'number of training uniform picture size.')
tf.flags.DEFINE_string('version', 'V0007', 'model version.')
tf.flags.DEFINE_string('h5', 'train_mask_rcnn.h5', 'saved model file.')
tf.flags.DEFINE_integer('num_gpus', 1, 'gpu nums.')

FLAGS = tf.flags.FLAGS

# 执行所在路径， V0xxx 表示模型版本号
source_path = os.path.join(os.getcwd(), "model/" + FLAGS.version + "/model")
sys.path.append(source_path)

from draw import model_load, display_instances, sliceImage, image_to_base64

# 选择输出绘制类识别80类，
COCO_CLASSES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# 模型权重文件
COCO_MODEL_PATH = os.path.join(source_path, FLAGS.h5)

created_at = str(round(time.time() * 1000))

# obs桶路径
obs_path = "obs://puddings/ma-mask-rcnn/notebook/out/image/" + created_at

# 输出目录
out_path = "notebook/out/image/" + created_at

# 输出目录存在需要删除里边的内容
if os.path.exists(out_path):
    file.remove(out_path, recursive=True)
os.makedirs(out_path)

if __name__ == "__main__":
    # 加载模型
    model = model_load(COCO_MODEL_PATH, FLAGS.input_size)

    # 读取图片
    image = cv2.imread(FLAGS.image)
    # 原图用于分割小图
    image_orig = image.copy() 

    prev_time = time.time()
    # 模型识别结果 rois, masks, class_ids, scores
    results = model.detect([image], verbose=0)[0]

    # 结果绘制到图
    image, recognizer = display_instances(
        image,
        results,
        COCO_CLASSES,
        FLAGS.min_score,
        FLAGS.show_image,
        FLAGS.show_box_label
    )

    # 绘制时间
    curr_time = time.time()
    exec_time = curr_time - prev_time
    print("识别耗时: %.2f ms" %(1000*exec_time))

    # print("识别结果：", recognizer)

    # 输入图片np uint8 尺寸/2
    # x, y = image.shape[0:2]
    # image = cv2.resize(image, (int(y / 2), int(x / 2)))

    # base图片编码
    # itb64 = image_to_base64(image)
    # print(itb64)
    
    # 绘制识别统计
    totalStr = ""
    for k in recognizer.keys():
        totalStr += '%s: %d    ' % (k, len(recognizer[k]))
    cv2.putText(image, totalStr, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

    # 绘制保存
    cv2.imwrite(out_path + "/output_result.jpg", image)
    cv2.imwrite(out_path + "/output_orig.jpg", image_orig)

    # 切割识别到的物体
    sliceImage(image_orig, recognizer, out_path)

    # 复制保存到桶
    print("输出目录：" + out_path)
    file.copy_parallel(out_path, obs_path)
    
    # 显示窗口
    # cv2.namedWindow('image_result', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('image_result', image)
    # 退出窗口
    # cv2.waitKey(0)
    # 任务完成后释放内容
    # cv2.destroyAllWindows()
    