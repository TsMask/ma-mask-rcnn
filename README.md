# 基于 ModelArts 平台模型部署预测 —— Mask R-CNN 模型

## 介绍

项目使用华为云 `ModelArts` AI 开发平台进行训练部署，采用 `Mask R-CNN` 算法模型进行识别预测。`Mask R-CNN` 源于 2018 年论文《Mask R-CNN》，是何恺明团队作品。`Mask R-CNN` 指的是在检测出图片中物体的同时为每一实例产生高质量的分割掩码（segmentation mask），算法基于 `Faster R-CNN`，在其中添加了实例掩码功能。

模型识别预测结果处理：

- 统一识别同种物体类别的颜色
- 自定义绘制实线边框和标签牌
- 统计识别结果的输出物体框坐标和颜色标识

>可以自己本机运行，需要安装 `tf1.13.1` 和 `pycocotools` 。    
>COCO数据集训练需要用到 `pycocotools` 这个第三方库   
>Windows电脑下安装出问题，建议Linux运行。    

## 目录结构

将目录文件夹上传至已创建的桶中，文件过多可能上传失败，建议分目录选择上传。

```text
ma-mask-rcnn

┌── infer                               模型推理
│   ├── mrcnn
│   ├── config.json
│   ├── customize_service.py
│   └── draw.py
├── log                                 模型训练其他日志
├── model                               模型训练版本输出
├── notebook                            开发环境模型调试
│   ├── detect_image.py
│   ├── detect_video.py
│   ├── run.ipynb
│   ├── test.jpg
│   └── test.mp4
├── train                               模型训练
│   ├── instance_segmentation.tar.gz
│   ├── mask_rcnn.py
│   └── pip-requirements.txt
├── web                                 在线服务API展示界面项目（自己本机运行）
│   ├── static
│   ├── templates
│   ├── main.py
│   └── requirements.txt
└── README.md
```

## 使用前提

拥有一个华为云账号

- EI 企业智能 —— ModelArts
- 存储 —— 对象存储服务 OBS

`ModelArts` 平台需要在全局配置中添加访问密钥才能使用的自动学习、数据管理、Notebook、训练作业、模型和服务可能需要使用对象存储功能，若没有添加访问密钥，则无法使用对象存储功能。

`对象存储服务 OBS` 创建一个桶进行文件的存储。

选择服务地区：**华北-北京四**

## 创建开发环境

在 `ModelArts` 平台中使用 _开发环境>Notebook_ 创建一个工作环境（Python3 | GPU）和选择对象存储服务（OBS）桶内的已创建或已上传的文件夹进行创建，之后可以启动已创建 `notebook` 进行在线的开发调试和具体的模型训练。使用一些关联文件需要同步到开发环境 `work` 文件夹内，注意同步文件大小存在指定大小内。

## 训练模型

在 `ModelArts` 平台中使用 _训练管理>训练作业_ 创建：

1. 算法来源为常用框架（TensorFlow | TF-1.13.1-python3.6）
2. 代码目录，选择已上传的 `train` 目录
3. 启动文件，选择 `train` 目录内的 `mask_rcnn.py` 文件
4. 数据来源为数据存储位置，选择已创建桶中已上传的 `train` 数据文件夹
5. 训练输出位置，选择已上传或创建名为 `model` 的文件夹
6. 作业日志路径，选择已上传或创建名为 `log` 的文件夹
7. 选择 **公共资源池>GPU** 训练更佳
8. 运行参数，参考下表：

|      名称       |  类型  |                    说明                    |
| :-------------: | :----: | :----------------------------------------: |
|    data_url     | string |      已创建桶中已上传的数据来源文件夹      |
|    train_url    | string | 已上传或已创建名为 `model` 的文件夹/V 版本 |
|    image_dim    | number |     训练统一图片大小 320/480/512/1024      |
|   epochs_step   | number |       每轮训练载入图片张数，默认 100       |
| validation_step | number |        每轮训练后验证张数，默认 20         |
|   max_epochs    | number |           训练迭代次数，默认 10            |
|    num_gpus     | number |         拥有 GPU 数，默认固定为 1          |

数据来源数据文件会复制到目录代码下。

## 模型推理

在完成上面的模型训练后，你可以在已上传或已创建名为 `model` 的文件夹/V 版本内看到模型文件，但是还不能直接部署，还需要编写平台的模型包规范。因为使用的是 `TF-1.13.1-python3.6` 环境训练，所以使用 `TensorFlow` 模型推理代码进行编写。

使用已经上传的 `infer` 文件夹中的推理文件：

- 方法一：使用开发环境 `notebook` 进行复制 `infer` 文件夹中的推理文件到对应版本模型的 `model` 文件夹内。
- 方法二：直接在 `对象存储服务 OBS` 桶内目录选择对应版本模型的 `model` 文件夹，选择上传对象上传 `infer` 文件夹内的推理文件。

完成上传后，就可以在 `ModelArts` 平台中使用 _模型管理>模型_ 导入：

1. 元模型来源，选择 **从训练中选择** 或 **从对象存储服务（OBS）中选择**
2. 根据元模型来源选择对应的模型版本，**选择训练作业** 或 **选择元模型**
3. 部署类型，选择在线服务即可

推理模型参数如下：

**输入参数**

|      名称      |  类型  |                         说明                         |
| :------------: | :----: | :--------------------------------------------------: |
|     image      |  file  |                     上传图片文件                     |
|   min_score    | number |                最小显示分数，默认 0.2                |
|   show_image   | number |      是否输出带覆盖物结果 base64 输出图，默认 1      |
| show_box_label | number |            结果图是否显示框和文字，默认 1            |
|   image_size   | number | 图片尺寸统一处理大小识别 320/480/512/1024，默认 1024 |

**输出参数**

|      名称       |  类型  |                       说明                        |
| :-------------: | :----: | :-----------------------------------------------: |
| predicted_image | string | 识别绘制图片 base64，根据输入参数 show_image 输出 |
| recognizer_data | object |             识别物体框和颜色标识数据              |

## 模型部署

在完成上面的模型推理后，就可以在 `ModelArts` 平台中使用 _部署上线>在线服务_ 部署。选择模型及配置是你导入训练推理编写后得到的模型和对应的模型版本号，选好 **CPU 规格**下一步直接提交，等待服务启动完成。

服务启动后，可以直接使用服务提供的预测进行图片的预测查看。

## 在线服务 API 展示界面项目

在自己的机器上安装依赖后运行，使用的预测识别服务来自上面的模型部署服务启动。

1. 使用前提是部署模型的在线服务并启动预测。
2. 修改界面项目的 `main.py` 里修改预测的 API 接口地址和获取 Token 的地区用户名和密码。

界面项目对在线服务预测接口进行预测调用实现功能：

- 显示预测结果图
- 对预测图片进行物体分割小图
- 对预测结果物体进行统计
- 对视频媒体文件进行取帧预测统计

## 建议阅读

`ModelArts` 平台：

- [模型包规范介绍](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0091.html)
- [ModelArts 平台常见问题](https://support.huaweicloud.com/modelarts_faq/modelarts_05_0014.html)
- [MoXing 开发指南](https://support.huaweicloud.com/moxing-devg-modelarts/modelarts_11_0001.html)

`Mask R-CNN` 算法模型：

- [Mask R-CNN 论文](http://cn.arxiv.org/pdf/1703.06870v3)
- [Mask R-CNN GitHub 仓库](https://github.com/matterport/Mask_RCNN)
- [Mask R-CNN 训练自己的数据集](https://blog.csdn.net/l297969586/article/details/79140840)
- [Mask R-CNN 算法及其实现详解](https://blog.csdn.net/remanented/article/details/79564045)
