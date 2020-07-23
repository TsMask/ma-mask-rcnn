from fastapi import FastAPI, Form, File
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import requests
import json

app = FastAPI()
# 挂载模板文件夹
tmp = Jinja2Templates(directory='templates')
# 挂载静态文件夹
app.mount('/static', StaticFiles(directory='static'), name='static')
# 令牌一天有效
token = None

# 首页
@app.get('/')
async def get_index(request: Request):
    return tmp.TemplateResponse('index.html', {'request': request, 'title': '基于 ModelArts 平台部署 - Mask R-CNN'})

# 预测
@app.post("/infers")
async def post_infers(request: Request,
                      image: bytes = File(...),
                      min_score: float = Form(...),
                      show_image: int = Form(...),
                      show_box_label: int = Form(...),
                      input_size: int = Form(...)):

    url = "https://772cc4556c7047dc9cfcb05814d147c9.apig.cn-north-4.huaweicloudapis.com/v1/infers/301e1052-e85f-415a-a656-9dde9479ffb8"
    data = {
        "min_score": min_score,
        "show_image": show_image,
        "show_box_label": show_box_label,
        "input_size": input_size,
    }
    files = [('image', image)]
    headers = {"X-Auth-Token": token}
    response = requests.request("POST", url, headers=headers, data=data, files=files)

    # 识别失败 识别成功
    if(response.status_code in [500, 200]):
        return response.json()

    return {"Token": token != None, "code": response.status_code}

# 获取Token
@app.post("/tokens")
async def post_tokens(request: Request):
    url = "https://iam.cn-north-4.myhuaweicloud.com/v3/auth/tokens"
    headers = {"Content-Type": "application/json"}
    data = {
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                        "domain": {
                            "name": "华为云账号"
                        },
                        "name": "华为云账号",
                        "password": "登录密码",
                    },
                },
            },
            "scope": {
                "domain": {
                    "name": "cn-north-4"
                },
                "project": {
                    "name": "cn-north-4"
                },
            },
        },
    }
    # 序列字符化
    payload = json.dumps(data)
    response = requests.request("POST", url, headers=headers, data=payload)

    if(response.status_code == 201):
        global token
        token = response.headers['X-Subject-Token']

    return {"code": response.status_code}
