window.onload = function () {
  // 连接状态
  const isConn = sessionStorage.getItem("token");
  // 连接按钮
  const tokenBtnEl = document.getElementById("tokenBtnEl");
  // 处理类型状态
  let isImage = true;
  // 处理类型按钮
  const typeBtnEl = document.getElementById("typeBtnEl");
  // 显识别图状态
  let resultImage = true;
  // 显识别图
  const resultBtnEl = document.getElementById("resultBtnEl");
  // 分割小图状态
  let sliceImage = true;
  // 分割小图
  const sliceBtnEl = document.getElementById("sliceBtnEl");
  // 文件选择按钮
  const fileInputEl = document.getElementById("fileInputEl");
  // 图片显示元素
  const imgEl = document.getElementById("imgEl");
  // 视频显示元素
  const videoEl = document.getElementById("videoEl");
  // 掩膜图结果元素
  const canvasEl = document.getElementById("canvasEl");
  const ctx = canvasEl.getContext("2d");
  // 识别输出总框节点
  const recognizerBoxEl = document.getElementById("recognizerBoxEl");
  // 识别输出统计数
  const infoBoxEl = document.getElementById("infoBoxEl");
  // 识别预测进行状态 图片视频进行闪烁
  const imageInter = document.getElementsByClassName("camera-dot-pink")[0];
  const videoInter = document.getElementsByClassName("camera-dot-aoi")[0];
  // 定时器
  let hasTimeout = null;
  // 统计数据
  let countData = {};
  // 识别类英>中文
  const COCO_CLASSES = {'BG':"背景", 'person':"人", 'bicycle':"自行车", 'car':"汽车", 'motorcycle':"摩托车", 'airplane':"飞机", 'bus':"公共汽车",
                'train':"火车", 'truck':"卡车", 'boat':"船", 'traffic light':"红绿灯", 'fire hydrant':"消防栓", 'stop sign':"停车标志",
                'parking meter':"停车收费表", 'bench':"长凳", 'bird':"鸟", 'cat':"猫", 'dog':"狗", 'horse':"马", 'sheep':"羊", 'cow':"牛",
                'elephant':"大象", 'bear':"熊", 'zebra':"斑马", 'giraffe':"长颈鹿", 'backpack':"背包", 'umbrella':"雨伞", 'handbag':"手提包",
                'tie':"领带", 'suitcase':"手提箱", 'frisbee':"飞盘", 'skis':"滑雪板", 'snowboard':"大象", 'sports ball':"运动球", 'kite':"风筝",
                'baseball bat':"棒球棒", 'baseball glove':"棒球手套", 'skateboard':"滑板", 'surfboard':"冲浪板", 'tennis racket':"网球拍",
                'bottle':"瓶子", 'wine glass':"酒杯", 'cup':"杯子", 'fork':"叉子", 'knife':"刀", 'spoon':"勺子", 'bowl':"碗", 'banana':"香蕉", 'apple':"苹果",
                'sandwich':"三明治", 'orange':"橘子", 'broccoli':"花椰菜", 'carrot':"胡萝卜", 'hot dog':"热狗", 'pizza':"披萨", 'donut':"甜甜圈", 'cake':"蛋糕", 'chair':"椅子",
                'couch':"沙发", 'potted plant':"盆栽", 'bed':"床", 'dining table':"餐桌", 'toilet':"厕所", 'tv':"电视", 'laptop':"笔记本", 'mouse':"鼠标", 'remote':"遥控器",
                'keyboard':"键盘", 'cell phone':"手机", 'microwave':"微波炉", 'oven':"烤箱", 'toaster':"吐司炉", 'sink':"水槽", 'refrigerator':"冰箱", 'book':"书", 'clock':"时钟",
                'vase':"花瓶", 'scissors':"剪刀", 'teddy bear':"泰迪熊", 'hair drier':"吹风机", 'toothbrush':"牙刷"}
  // 预测接口url
  let url_infers = "http://127.0.0.1:8000/infers";
  // token接口url
  let url_tokens = "http://127.0.0.1:8000/tokens";

  console.info("界面样式素材来自：哔哩哔哩 BML2020，宅现场云应援！");
  console.info("编写者：TsMask");
  console.log("平台服务连接状态：", isConn);

  // 根据连接状态预设
  tokenBtnEl.style.backgroundColor = isConn ? "#6dc781" : "#ff5c7c";
  tokenBtnEl.style.cursor = isConn ? "no-drop" : "pointer";
  tokenBtnEl.disabled = !!isConn;
  fileInputEl.style.cursor = !isConn ? "no-drop" : "pointer";
  fileInputEl.disabled = !isConn;
  typeBtnEl.style.backgroundColor = "#7f375a";
  resultBtnEl.style.backgroundColor = "#6dc781";
  sliceBtnEl.style.backgroundColor = "#6dc781";
  imgEl.style.display = "block";
  videoEl.style.display = "none";
  imageInter.style.display = "none";
  videoInter.style.display = "none";

  // 处理类型更换
  typeBtnEl.addEventListener("click", function (e) {
    videoEl.pause();
    clearTimeout(hasTimeout);
    // 改变选择类型
    e.target.style.backgroundColor = isImage ? "#257c82" : "#7f375a";
    e.target.textContent = isImage ? "视频" : "图片";
    imgEl.style.display = isImage ? "none" : "block";
    videoEl.style.display = isImage ? "block" : "none";
    fileInputEl.accept = isImage ? "video/mp4" : "image/jpeg";
    isImage = !isImage;
    // 置空原来数据
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    recognizerBoxEl.innerHTML = "";
    infoBoxEl.innerHTML = "";
    countData = {};
  });

  // 初始获取token标记
  tokenBtnEl.addEventListener("click", function (e) {
    fetch(url_tokens, { method: "POST" })
      .then(function (response) {
        response.json().then(function (res) {
          if (res.code == 201) {
            sessionStorage.setItem("token", true);
            e.target.style.backgroundColor = "#6dc781";
            e.target.style.cursor = "no-drop";
            e.target.disabled = true;
            fileInputEl.style.cursor = "pointer";
            fileInputEl.disabled = false;
          } else {
            alert("连接失败：" + res.code);
          }
        });
      })
      .catch((error) => console.log("error", error));
  });

  // 更换文件刷新识别
  fileInputEl.addEventListener("change", function (e) {
    if (!e.target.files.length) return;
    const file = e.target.files[0];
    // 置空原来数据
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    recognizerBoxEl.innerHTML = "";
    infoBoxEl.innerHTML = "";
    countData = {};
    if (isImage) {
      // 原图显示
      imgEl.onload = function () {
        canvasEl.width = imgEl.width;
        canvasEl.height = imgEl.height;
      };
      fileToBase64(file, function (base64) {
        imgEl.src = base64;
      });
      imageInter.style.display = "block";
      interFetch(file, +resultImage);
    } else {
      videoEl.src = URL.createObjectURL(file);
      setTimeout(function () {
        videoEl.play();
      }, 500);
    }
  });

  // 是否分割小图更换
  sliceBtnEl.addEventListener("click", function (e) {
    sliceImage = !sliceImage;
    e.target.style.backgroundColor = sliceImage ? "#6dc781" : "#ff5c7c";
  });

  // 是否显识别图更换
  resultBtnEl.addEventListener("click", function (e) {
    resultImage = !resultImage;
    e.target.style.backgroundColor = resultImage ? "#6dc781" : "#ff5c7c";
    canvasEl.style.display = resultImage ? "block" : "none";
  });

  // 视频播放监听触发
  videoEl.addEventListener("play", function () {
    videoInter.style.display = "block";
    canvasEl.width = videoEl.videoWidth / 2;
    canvasEl.height = videoEl.videoHeight / 2;
    runVideoInter();
  });

  // 视频取帧预测
  function runVideoInter() {
    console.log("视频暂停状态：", videoEl.paused);
    // 视频暂停退出回调
    if (videoEl.paused) {
      clearTimeout(hasTimeout);
      videoInter.style.display = "none";
      return;
    }
    // 创建画布
    const canvasVideoTempEl = document.createElement("canvas");
    canvasVideoTempEl.width = videoEl.videoWidth / 2;
    canvasVideoTempEl.height = videoEl.videoHeight / 2;
    // 画布绘制
    var ctx = canvasVideoTempEl.getContext("2d");
    ctx.drawImage(
      videoEl,
      0,
      0,
      canvasVideoTempEl.width,
      canvasVideoTempEl.height
    );
    // 画布转数据预测 console.log(canvasEl.toDataURL("image/jpeg"));
    canvasVideoTempEl.toBlob(
      function (blob) {
        blob.name = "video.jpg";
        interFetch(blob, +resultImage);
      },
      "image/jpeg",
      0.7
    );
    // 开始定时器
    hasTimeout = setTimeout(() => runVideoInter(), 1000 * 8);
  }

  /**
   * 预测识别
   * @param {File} file
   */
  function interFetch(file, result_base64) {
    let formdata = new FormData();
    formdata.append("min_score", 0.2); // 最小显示分数
    formdata.append("show_image", result_base64); // 输出结果图
    formdata.append("show_box_label", result_base64); // 输出的结果图带物体框和标识
    formdata.append("input_size", 1024); // 图片尺寸统一处理大小，512/7s,1024/16s
    formdata.append("image", file, file.name);
    // 发送
    fetch(url_infers, { method: "POST", body: formdata })
      .then(function (response) {
        response.json().then(function (res) {
          if (res.erno || res.code == 400) return interFetch(file, result_base64);
          if (res.code == 404) return alert("预测接口已关闭，请启动！");
          if (isImage) imageInter.style.display = "none";
          // 是否绘制掩膜图
          if (result_base64) {
            const imgTempEl = new Image();
            imgTempEl.onload = function () {
              ctx.drawImage(imgTempEl, 0, 0, canvasEl.width, canvasEl.height);
            };
            imgTempEl.src = res.predicted_image;
          }
          // 是否分割小图
          sliceImage
            ? sliceImageBox(file, res.recognizer_data)
            : outCount(res.recognizer_data);
        });
      })
      .catch((error) => console.log("error", error));
  }

  /**
   * 分割图片小图
   * @param {File} image_file 图片文件
   * @param {Object} recognizer_data 识别数据
   */
  function sliceImageBox(image_file, recognizer_data) {
    // 创建临时原图片节点
    const imgOrgTempEl = document.createElement("img");
    imgOrgTempEl.onload = function () {
      // 将识别结果处理后添加分类到输出总框
      for (const item in recognizer_data) {
        const element = recognizer_data[item];
        // 统计信息节点
        if (!countData.hasOwnProperty(item)) {
          countData[item] = {};
          countData[item]["name"] = COCO_CLASSES[item];
          countData[item]["num"] = 0;
        }
        countData[item]["value"] = `= ${+countData[item]["num"]} +${element.length}`;
        countData[item]["num"] += element.length;
        // 类别内添加图片节点
        element.forEach((ele) => {
          // 边框和颜色
          let b = ele.box;
          let c = ele.box_color;
          countData[item]["color"] = "#" + ((1 << 24) + (c[0] << 16) + (c[1] << 8) + c[2]).toString(16).slice(1);
          // 创建临时图片画布
          const canvasTempEl = document.createElement("canvas");
          canvasTempEl.width = 200;
          canvasTempEl.height = 300;
          let ctxs = canvasTempEl.getContext("2d");
          // 截取绘制选定框图
          ctxs.drawImage(
            imgOrgTempEl,
            b[1],
            b[0],
            b[3] - b[1],
            b[2] - b[0],
            0,
            0,
            canvasTempEl.width,
            canvasTempEl.height
          );
          // 文字背景条
          ctxs.lineWidth = 50;
          ctxs.strokeStyle = countData[item]["color"];;
          ctxs.moveTo(0, 0);
          ctxs.lineTo(60 + COCO_CLASSES[item].length * 30, 0);
          ctxs.stroke();
          // 绘制类型和分数
          ctxs.font = "italic normal bold 18px sans-serif";
          ctxs.fillStyle = "#00000";
          ctxs.fillText(COCO_CLASSES[item], 10, 20);
          ctxs.fillText(ele.classes_score, COCO_CLASSES[item].length * 30, 20);
          // 图片节点
          let imgTempEl = document.createElement("img");
          imgTempEl.src = canvasTempEl.toDataURL();
          imgTempEl.alt = COCO_CLASSES[item];
          imgTempEl.title = `${COCO_CLASSES[item]} ${item} ${ele.classes_score}`;
          imgTempEl.style.borderColor = countData[item]["color"];
          recognizerBoxEl.appendChild(imgTempEl);
        });
      }
      // 计算滚动时间
      recognizerBoxEl.style.animationDuration = Math.ceil(recognizerBoxEl.offsetWidth / 860) * 50 + "s";
      // 输出统计节点
      outCountNode(countData);
    };
    // 图片文件转base64
    fileToBase64(image_file, function (base64) {
      imgOrgTempEl.src = base64;
    });
  }

  /**
   * 统计识别分类数
   * @param {Object} recognizer_data 识别数据
   */
  function outCount(recognizer_data) {
    // 将识别结果处理后添加分类到输出总框
    for (const item in recognizer_data) {
      const element = recognizer_data[item];
      // 统计信息节点
      if (!countData.hasOwnProperty(item)) {
        countData[item] = {};
        countData[item]["name"] = COCO_CLASSES[item];
        countData[item]["num"] = 0;
      }
      countData[item]["value"] = `= ${+countData[item]["num"]} +${element.length}`;
      countData[item]["num"] += element.length;
      let c = element[0].box_color;
      countData[item]["color"] = "#" + ((1 << 24) + (c[0] << 16) + (c[1] << 8) + c[2]).toString(16).slice(1);
    }
    // 输出统计节点
    outCountNode(countData);
  }

  /**
   * 输出统计节点
   * @param {Object} countData 统计数据
   */
  function outCountNode(countData) {
    // 置空原来数据
    infoBoxEl.innerHTML = "";
    // 遍历统计输出添加节点
    for (const item in countData) {
      const element = countData[item];
      // 统计数据再次添加
      const infoTitleDivTempEl = document.createElement("div");
      infoTitleDivTempEl.classList.add("title");
      infoTitleDivTempEl.textContent = `${element.name}：`;
      const infoValueDivTempEl = document.createElement("div");
      const infoNumDivTempEl = document.createElement("div");
      infoValueDivTempEl.classList.add("value");
      infoNumDivTempEl.classList.add("num");
      infoValueDivTempEl.textContent = element.value;
      infoNumDivTempEl.textContent = element.num;
      const infoDivTempEl = document.createElement("div");
      infoDivTempEl.classList.add("item");
      infoDivTempEl.style.color = element.color;
      infoDivTempEl.appendChild(infoTitleDivTempEl);
      infoDivTempEl.appendChild(infoNumDivTempEl);
      infoDivTempEl.appendChild(infoValueDivTempEl);
      infoBoxEl.appendChild(infoDivTempEl);
    }
  }
};

/**
 * 图片文件对象转base64
 * @param {File} image_file
 * @param {String} callback
 */
function fileToBase64(image_file, callback) {
  const reader = new FileReader();
  reader.addEventListener("load", function () {
    return callback(reader.result);
  });
  reader.readAsDataURL(image_file);
}
