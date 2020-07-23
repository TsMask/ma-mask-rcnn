from model_service.tfserving_model_service import TfServingBaseService
from draw import model_load, display_instances, image_to_base64
from skimage import io

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

# 模型权重
COCO_MODEL_PATH = "model/1/train_mask_rcnn.h5"

# 推理服务
class mask_rcnn_service(TfServingBaseService):
    count = 1               # 预测次数
    model_object = None     # 模型实例
    input_size = 1024       # 图片尺寸统一处理大小识别

    def _preprocess(self, data):
        preprocessed_data = {}
        
        # 遍历提交参数取值，image必传，配置默认值
        for k, v in data.items():
            if k == 'image':
                # 参数的默认值
                preprocessed_data['min_score'] = float(data['min_score']) if 'min_score' in data else 0.2
                preprocessed_data['show_image'] = int(data['show_image']) if 'show_image' in data else 1
                preprocessed_data['show_box_label'] = int(data['show_box_label']) if 'show_box_label' in data else 1
                preprocessed_data['input_size'] = int(data['input_size']) if 'input_size' in data else 1024
                
                # file_name, file_content 图片字典数据
                for _, file_content in v.items():
                    image = io.imread(file_content)
                    preprocessed_data[k] = image

        # 变更模型实例
        if(self.input_size != preprocessed_data['input_size']):
            print('--变更模型实例--')
            self.input_size = preprocessed_data['input_size']
            self.model_object = model_load(COCO_MODEL_PATH, self.input_size)

        # 加载模型实例
        if(self.model_object == None):
            print('--加载模型实例--')
            self.model_object = model_load(COCO_MODEL_PATH, self.input_size)
              
        return preprocessed_data

    def _postprocess(self, data):
        outputs = {}

        # 模型识别结果 rois, masks, class_ids, scores
        results = self.model_object.detect([data['image']], verbose=0)[0]

        # 结果绘制到图
        image, recognizer = display_instances(
            data['image'],
            results,
            COCO_CLASSES,
            data['min_score'],
            data['show_image'],
            data['show_box_label'],
        )

        # 预测次数+1
        print('预测次数：', self.count)
        self.count += 1

        # 输出数据，show_image是否输出识别处理的base64图片
        outputs['recognizer_data'] = recognizer
        if data['show_image']:
            itb64 = image_to_base64(image)
            outputs['predicted_image'] = itb64

        return outputs
