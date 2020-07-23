# 模型管理 - 推理文件夹

[模型包规范介绍](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0091.html)

- 模型包里面必须包含 `model` 文件夹，`model` 文件夹下面放置模型文件，模型配置文件，模型推理代码。
- 模型配置文件必需存在，文件名固定为 `config.json` , 有且只有一个，模型配置文件编写请参见[模型配置文件编写说明](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0092.html)。
- 模型推理代码文件是可选的。推荐采用相对导入方式（Python import）导入自定义包。如果需要此文件，则文件名固定为 `customize_service.py` , 此文件有且只能有一个，模型推理代码编写请参见[模型推理代码编写说明](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0093.html)。

## 推理文件上传

```text
你创建并上传的到OBS桶内的文件夹
│
├── infer                           模型推理文件
│   ├── mrcnn                       MaskRonn 模型识别所需文件
│   ├── config.json                 模型配置文件
│   ├── customize_service.py        模型推理代码文件
│   └── draw.py                     Mask R-CNN 识别所需文件
...
```

将该文件夹内的文件上传至对应训练输出的模型版本文件夹里的 `model` 文件夹下。

## 推理模型参数

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
