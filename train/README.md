# 训练作业 - 模型训练文件夹

- train -- 模型启动文件所在 OBS 文件夹

  - `model.py` -- 模型启动文件。
  - `pip-requirements.txt` -- 定义的配置文件，用于指定依赖包的包名及版本号。

需要数据集 `train` 目录内 `instance_segmentation.tar.gz`（将目录内文件复制到环境里model/1下）

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
|    data_url     | string |  已创建桶中已上传的 `train` 数据文件夹   |
|    train_url    | string | 已上传或已创建名为 `model` 的文件夹/V 版本 |
|    image_dim    | number |     训练统一图片大小 320/480/512/1024      |
|   epochs_step   | number |       每轮训练载入图片张数，默认 100       |
| validation_step | number |        每轮训练后验证张数，默认 20         |
|   max_epochs    | number |           训练迭代次数建议5，默认 10            |
|    num_gpus     | number |         拥有 GPU 数，默认固定为 1          |

**instance_segmentation.tar.gz 下载**

```ipynb
from modelarts.session import Session
session = Session()

if session.region_name == 'cn-north-1':
    bucket_path="modelarts-labs/end2end/mask_rcnn/instance_segmentation.tar.gz"
    
elif session.region_name == 'cn-north-4':
    bucket_path="modelarts-labs-bj4/end2end/mask_rcnn/instance_segmentation.tar.gz"
else:
    print("请更换地区到北京一或北京四")
    
session.download_data(bucket_path=bucket_path,
                      path='./instance_segmentation.tar.gz')
# 使用tar命令解压资源包
!tar zxf ./instance_segmentation.tar.gz
```