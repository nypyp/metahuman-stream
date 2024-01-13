A streaming digital human based on the Ernerf model， realize audio video synchronous dialogue. It can basically achieve commercial effects.  
基于ernerf模型的流式数字人，实现音视频同步对话。基本可以达到商用效果

[![Watch the video]](/assets/demo.mp4)

## 1. Installation

Tested on Ubuntu 20.04, Python3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda activate nerfstream
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```
linux cuda环境搭建可以参考这篇文章 https://zhuanlan.zhihu.com/p/674972886

### 1.2 安装rtmpstream库  
参照 https://github.com/lipku/python_rtmpstream


## 2. Run

### 2.1 运行rtmpserver (srs)
```
docker run --rm -it -p 1935:1935 -p 1985:1985 -p 8080:8080 registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5
```

### 2.2 启动数字人：

```python
python app.py
```

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
```

运行成功后，用vlc访问rtmp://serverip/live/livestream

### 2.3 网页端数字人播报输入文字
安装并启动nginx
```
apt install nginx
nginx
```
修改echo.html中websocket和视频播放地址，将serverip替换成实际服务器ip  
然后将echo.html和mpegts-1.7.3.min.js拷到/var/www/html下


用浏览器打开http://serverip/echo.html, 在文本框输入任意文字，提交。数字人播报该段文字
  
## 3. Docker Run  
不需要第1步的安装，直接运行。
```
docker run --gpus all -it --network=host --rm  registry.cn-hangzhou.aliyuncs.com/lipku/nerfstream:v1.3
```
srs和nginx的运行同2.1和2.3

## 4. Data flow
![](/assets/dataflow.png)

## 5. 数字人模型文件
可以替换成自己训练的模型(https://github.com/Fictionarry/ER-NeRF)
```python
.
├── data
│   ├── data_kf.json			
│   ├── pretrained
│   └── └── ngp_kg.pth

```

## 6. 性能分析
1. 帧率  
在Tesla T4显卡上测试整体fps为18左右，如果去掉音视频编码推流，帧率在20左右。用4090显卡应该能达到25帧，欢迎有显卡资源的同学提供数据。  
优化：新开一个线程运行音视频编码推流  
2. 延时  
整体延时5s多  
（1）tts延时2s左右，目前用的edgetts，需要将每句话转完后一次性输入，可以优化tts改成流式输入  
（2）wav2vec延时1s多，需要缓存50帧音频做计算，可以通过-m设置context_size来减少延时  
（3）srs转发延时，设置srs服务器减少缓冲延时

## 7. TODO
- 添加chatgpt实现数字人对话
- 声音克隆
- 数字人静音时用一段视频代替

如果本项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目。  
Email: lipku@foxmail.com
