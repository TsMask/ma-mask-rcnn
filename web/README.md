# 基于 ModelArts 平台部署 - 布丁几个

使用 `Mask R-CNN` 模型进行图片文件识别预测。

页面界面样式素材来自：哔哩哔哩 BML2020，宅现场云应援！

1. 先启动 `ModelArts` 平台在线服务
2. 在该项目 `main.py` 里修改预测的 API 接口地址和获取 Token 的地区用户名和密码。

## 项目依赖

```shell
# 安装所需依赖
pip install -r requirements.txt

# 四个必装的依赖
# pip install fastapi
# pip install uvicorn
# pip install python-multipart
# pip install aiofiles

```

## 项目启动

```shell
# 开发调试
uvicorn main:app --reload

# 生成部署
uvicorn main:app --host 0.0.0.0 --port 80
```
