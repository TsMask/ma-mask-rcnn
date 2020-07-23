import os, sys
import tensorflow as tf
from moxing.framework import file
import cv2
import time

# 执行参数 python notebook/detect_video.py --video notebook/test.mp4 --min_score 0.2 --show_image true --show_box_label true --image_size 1024 --version V0007 --h5 train_mask_rcnn.h5
# 外部参数配置
tf.flags.DEFINE_string('video', 'dataset/test.mp4', 'dataset file.')
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

from draw import model_load, display_instances, sliceImage

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

# 输出目录
out_path = "notebook/out/video"

# 输出目录存在需要删除里边的内容
if os.path.exists(out_path):
    file.remove(out_path, recursive=True)
os.makedirs(out_path)

# 帧数，用于通过帧数取图
frameNum = 0

# 视频总帧统计物体数，存在重复
totalCount = {}

# obs桶路径
obs_path = "obs://puddings/ma-mask-rcnn/notebook/out/video"

# 保存统计总数并复制保存到桶
def outTotalObs(totalCount, out_path, obs_path):
    # 打开文件进行视频识别物总统计
    totalFile = open(out_path + "/totalCount.txt","w")
    # 文件统计写入
    for k in totalCount.keys():
        labelStr = "{0}：{1} \n".format(k, totalCount[k])
        totalFile.write(labelStr)
     # 关闭文件统计        
    totalFile.close()
    # 复制保存到桶
    file.copy_parallel(out_path, obs_path)

if __name__ == "__main__":
    # 加载模型
    model = model_load(COCO_MODEL_PATH, FLAGS.input_size)

    # 读取视频
    video = cv2.VideoCapture(FLAGS.video)

    # 输出保存视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_out = cv2.VideoWriter(out_path + "/outputVideo.mp4", fourcc, fps, size)

    # 视频是否可以打开，进行逐帧识别绘制
    while video.isOpened:
        # 视频读取图片帧
        retval, frame = video.read()
        if retval:
            frame_orig = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # 保存统计总数并复制保存到桶
            outTotalObs(totalCount, out_path, obs_path)
            # 读取失败、结束后释放所有内容
            video.release()
            video_out.release()
            print("没有图像！尝试使用其他视频")
            break

        print('识别帧：%d/%d' % (frameNum, video.get(7)))
        prev_time = time.time()
        # 模型识别结果 rois, masks, class_ids, scores
        results = model.detect([frame], verbose=0)[0]

        # 结果绘制到图
        image, recognizer = display_instances(
            frame,
            results,
            COCO_CLASSES,
            min_score=FLAGS.min_score,
            show_image=FLAGS.show_image,
            show_box_label=FLAGS.show_box_label
        )
        
        # 绘制时间
        curr_time = time.time()
        exec_time = curr_time - prev_time
        print("识别耗时: %.2f ms" %(1000*exec_time))

        # print("识别结果：", recognizer)

        # 遍历识别数据并绘制帧识别统计
        totalStr = ""
        for k in recognizer.keys():
            if k not in totalCount: totalCount[k] = 0
            num = len(recognizer[k]);
            totalCount[k] += num
            totalStr += '%s: %d    ' % (k, num)
            cv2.putText(image, totalStr, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

        # 视频输出保存
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_out.write(result)

        # 每300帧取图进行分割保存
        if(frameNum % 300 == 0):
            # 输出帧目录,不存目录需要创建
            slice_path = os.path.join(out_path, "imageSeg-" + str(frameNum))
            if not os.path.exists(slice_path):
                os.makedirs(slice_path)
            # 绘制帧保存
            cv2.imwrite(os.path.join(slice_path, "output_result.jpg"), result)
            cv2.imwrite(os.path.join(slice_path, "output_orig.jpg"), frame_orig)
            # 切割识别到的物体
            sliceImage(frame_orig, recognizer, slice_path)
        frameNum += 1

        # 绘制视频显示窗
        # cv2.namedWindow("video_reult", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("video_reult", result)
        # 退出窗口
        # if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    # 保存统计总数并复制保存到桶
    print("输出目录：" + out_path)
    outTotalObs(totalCount, out_path, obs_path)
    
    # 任务完成后释放所有内容
    video.release()
    video_out.release()
    # cv2.destroyAllWindows()
