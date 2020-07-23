from mrcnn.config import Config
import mrcnn.model as modellib
import numpy as np
import cv2
import random
import base64
import colorsys
import os

# 识别80类
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

# 预测配置
class InferenceConfig(Config):
    # 可辨识的名称
    NAME = "my_inference"

    # GPU的数量和每个GPU处理的图片数量，可以根据实际情况进行调整，参考为Nvidia Tesla P100
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 物体的分类个数，COCO中共有80种物体+背景
    NUM_CLASSES = 1 + 80  # background + 80 shapes

# 载入模型
def model_load(model_path="model/1/train_mask_rcnn.h5", input_size=1024):
    # 配置参数
    inference_config = InferenceConfig()
    # 图片尺寸统一处理为1024，可以根据实际情况再进一步调小
    inference_config.IMAGE_MIN_DIM = input_size
    inference_config.IMAGE_MAX_DIM = input_size
    inference_config.display()

    # 模型预测对象
    inference_model = modellib.MaskRCNN(mode="inference",
                                        config=inference_config,
                                        model_dir="logs")

    # 训练权重
    inference_model.load_weights(model_path, by_name=True)
    inference_model.keras_model._make_predict_function()

    return inference_model

# 图片np转base64，在cv中加色还原灰图
def image_to_base64(image_np):
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image = cv2.imencode('.jpg', image)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return 'data:image/jpeg;base64,{}'.format(image_code)

# 随机生成类别色
def random_colors(N):
    hsv_tuples = [(1.0 * x / N, 1., 1.) for x in range(N)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    return colors

# 颜色覆盖物体
def apply_mask(image, mask, color, alpha=0.4):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

# 显示绘制实例识别图像框标签
def display_instances(image, results, class_names, min_score=0.2, show_image=True, show_box_label=False):
    # 结果参数进行操作绘制
    boxes = results['rois']
    masks = results['masks']
    classes_scores = results['scores']
    class_ids = results['class_ids']
    image_h, image_w = image.shape[:2]
    colors = random_colors(81)  # 生成80类颜色组
    recognizer = {}             # 识别物统计
    
    print("识别目标：{0} , 图片高宽：{1} x {2}".format(boxes.shape[0], image_h, image_w))

    # 遍历绘制类别 len(bboxes)
    for i, box in enumerate(boxes):
        class_id = class_ids[i]             # 类别下标
        box_label = class_names[class_id]   # 类别标签名称
        box_color = colors[class_id]        # 类别所属颜色
        classes_score = classes_scores[i]   # 类别识别分数
        y1, x1, y2, x2 = box                # 边框四点坐标
       
        # 不在目标类的跳过
        if box_label not in class_names: continue

        # 检测分小于跳过
        if classes_score < min_score: continue
        
        # 是否识别并统计
        if box_label not in recognizer: recognizer[box_label] = []
        
        # 同类型追加
        label = recognizer[box_label]
        item = {}
        item['classes_score'] = str(round(classes_score,2))
        item['box_color'] = box_color
        item['box'] = np.array([y1, x1, y2, x2]).tolist()
        label.append(item)

        # 结果时带覆盖物输出
        if show_image:
            mask = masks[:, :, i]
            image = apply_mask(image, mask, box_color, 0.3)

        # 绘制边框和文字
        if show_image and show_box_label:
            box_size = 2        # 边框大小
            font_scale = 0.4    # 字体比例大小
            caption = '{} {:.2f}'.format(box_label, classes_score) if classes_score else box_label
            image = cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_size)
            # 填充文字区
            text_size = cv2.getTextSize(caption, 0, font_scale, thickness=box_size)[0]
            image = cv2.rectangle(image, (x1, y1), (x1 + text_size[0], y1 + text_size[1] + 8), box_color, -1)
            image = cv2.putText(
                image,
                caption,
                (x1, y1 + text_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (50, 50, 50),
                box_size//2,
                lineType=cv2.LINE_AA
            )

    # print(recognizer)    
    return image, recognizer
    
# 识别物体保存小图
def sliceImage(image_orig, recognizer, out_path):
    # 打开文件统计后遍历物体结果数据
    totalFile = open(out_path + "/totalCount.txt","w")
    for i, label in enumerate(recognizer):
        # 文件统计写入
        labelTotal = "{0}：{1} \n".format(label, len(recognizer[label]))
        totalFile.write(labelTotal)
        # 输出类别目录，不存在需要创建
        label_path = os.path.join(out_path, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        # 开始对类别物体进行切割
        for i, boxs in enumerate(recognizer[label]):
            y1, x1, y2, x2 = boxs['box']
            cv2.imwrite("{0}/{1}-{2}-{3}.jpg".format(label_path, label, i, boxs['classes_score']), image_orig[y1:y2, x1:x2])
    # 关闭文件统计        
    totalFile.close()
