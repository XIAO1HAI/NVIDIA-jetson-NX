# NVIDIA Jetson NX 开发学习笔记

**注：**【提前声明】由于也是第一次开发NX，所以也是在探索中前行，所以在成功搭建之前都是在摸石头过河，可能有些过程图没有很及时的记录，部分都是后来用手机补的记录过程，所以再次声明一下，希望可以帮助到你！

## **创建虚拟环境**--==!important==

官网没有建议创建虚拟环境的方法，参考很多博客和文章：

1、有直接和平时电脑上安装anaconda一样进行创建环境；

2、也有说因为NX是ARM架构，直接安装anaconda会出现一些潜在的问题，所以推荐安装miniforge代替（和anaconda命令操作一模一样）

【两种方法我都试了，1中直接安装anaconda的.sh文件也可以正常使用，但有潜在问题，没有遇到暂不评论】

【2中的miniforge创建环境、指定python版本等都很方便，唯一缺点就是无论你用conda安装依赖包和pip安装，最终的安装的包都在usr/local/../dist-packages下面，你对应虚拟环境下面pip list 和 conda list下没有你刚刚安装的包的，list之后没有就会导致环境依赖包无法正常使用，在虚拟环境重新安装会提示你-该包已经存在（Requirement already satisfied:）】

>鉴于以上两种原因，决定选用python3自带的创建venv的命令，如下：
>
>```python
>#首先查看自带的python版本
>python
># 使用python3自带的虚拟环境创建
>python3 -m venv yourvenvname
>#激活虚拟环境
>source yourvenvname/bin/activate
>```
>
>**注：**python3自带的虚拟环境创建，创建之后的python版本和base环境下一样，暂时没有办法更改(可以通过安装多版本的python，并设置优先级进行解决)
>
>执行上面命令之后，可以正常安装使用所需要的依赖包，通过pip list 可以进行查看依赖包是否安装在当前环境下,有包的话就成功安装依赖包。

参考文章：

[1]https://docs.python.org/3/tutorial/stdlib.html

[2]http://t.zoukankan.com/yangzhuzhu-p-12981956.html

### **多版本Python安装**

我的NX里面自带的是python3.6，因为任务需要python3.8，所以安装以下多版本的python

>首先，直接用root用户权限直接安装所需要的第二个python版本，这里我需要python3.8，就直接安装python3.8（**注：**安装完第二个版本的python3.8之后，新python3.8环境中是所有依赖包都没有的，pip都是最老的版本，和另外python3.6中的sudo apt-get install都是单独存在，更不用说pip install的依赖包）
>
>```python
>#安装所需第二个python版本python3.8
>sudo apt-get install python3.8
>#安装好之后，查看python各个版本的所在位置
>which python3  #输出的是所在目录  我的是
>/usr/bin/python3
>which python3.6    #输出的是所在目录  我的是
>/usr/bin/python3.6
>which python3.8  #输出的是所在目录  我的是 
>/usr/bin/python3.8
>```
>
>然后，通过update-alternatives命令用来维护系统命令的符号链接，可以将多个文件链接到同一个符号文件上，并进行管理。(简单来说就是可以设置多个版本的优先级来进行不同Python版本之间的切换)
>
>![image-20220419120415061](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419120415061.png)
>
>```python
>#设置以后输入python就可以直接进入python3的环境中，具体是3.6还是3.8需要看你设置的python3.6和python3.8后边的数字，数字越大就代表优先级越高（这里设置的python3.6的优先级为2，python3.8的优先级为1）
>#第一个目录下为python 代表要替换的python所在目录 后边紧跟的python为python环境的名称
>#如果为python3就是要替换python3这个名称
>sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
>sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
>```
>
>具体配置参数含义如下图：
>
>![image-20220419115855136](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419115855136.png)
>
>```python
>#设置完成之后，可以通过命令查看设置的优先级
>sudo update-alternatives --config python
>```
>
>具体如图：
>
>![image-20220419120454092](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419120454092.png)
>
>测试-->
>
>在终端输入python，正常启动python环境，为3.6版本，在==sudo update-alternatives --config python==命令后可以选择具体的环境，选择2就如上图所示，选择了3.8的环境，如图：
>
>![image-20220419144501277](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419144501277.png)
>
>>**注：**可能会遇到的问题，切换到3.8环境后，使用==python3 -m venv yourvenvname==发现报错了，按照报错原因执行==sudo apt-get install python3-venv==仍然无法解决，而且在base环境也无法pip list ,最终查阅查考文章解决。(如果没有遇到问题可跳过)
>
>>python3 -m venv yourvenvname报错
>
>>![image-20220419145451286](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419145451286.png)
>
>>解决过程：
>
>>![image-20220419150306064](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419150306064.png)
>
>>```python
>>#解决python -m venv创建虚拟环境报错
>>#需要安装对应版本的python3-venv
>>sudo apt-get install python3.8-venv
>>```
>
>>![img](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\20201014094940492.png#pic_center)
>
>>pip list报错，参考[3]的前几种方法依然无法pip list,最终选择了强制重新安装pip，成功解决
>
>>![image-20220419145131248](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419145131248.png)
>
>>解决过程：
>
>>![image-20220419150211379](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419150211379.png)
>
>>```python
>>#解决安装多版本python后pip list报错
>>curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
>>python3 get-pip.py --force-reinstall
>>#成功可以pip list
>>```
>
>最终，成功安装多版本python并且可以创建python3.8的虚拟环境
>
>![image-20220419150411901](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419150411901.png)
>
>![image-20220419150703354](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419150703354.png)
>
>参考文章：
>
>[1]http://www.360doc.com/content/21/1103/14/15266844_1002579390.shtml
>
>[2]https://blog.csdn.net/LabDNirvana/article/details/109066152
>
>[3]https://blog.csdn.net/whatday/article/details/106480390

环境搭建后，进入环境中均可以升级一下pip版本，初始环境中的pip都是9.0

```python
#升级pip3
python -m pip install --upgrade pip
```

### **pytorch环境搭建**

#### **Python36**

**注：**硬件：jetpack4.6 软件：python=3.6，torch=1.6.0，torchvision=0.7.0

>1、确保已经**安装好Jetpack**在您的NX板子上(正常可以进入Ubuntu系统桌面即为安装)--我的jetpack版本为4.6
>
>>1.1、可以通过安装jetson-stats进行查看NX板子的实时运行情况(sudo 用管理员root身份运行，不一定每一次都需要加)
>
>>```python
>>sudo jetson_clocks
>># 安装 jetson-stats
>>sudo -H pip install jetson-stats
>># 查看状态(可以实时查看板子CPU、GPU的运行情况，以及jetpack版本信息)
>>sudo jtop
>>```
>
>2、**安装pytorch环境依赖包**
>
>```python
>#更新apt-get
>sudo apt-get -y update
>#安装依赖包以及升级pip3
>sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip
>```
>
>3、**安装torch**
>
>下载[pytorch](https://so.csdn.net/so/search?q=pytorch&spm=1001.2101.3001.7020)编译好的==.whl文件==，使用Nvidia官方预编译whl文件．选择==自己需要的版本==下载
>
>​	下载链接：[PyTorch for Jetson - version 1.10 now available - Jetson Nano - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)
>
>![image-20220415191454544](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220415191454544.png)
>
>.whl文件下载好之后，cd 命令进入下载的目录并执行：
>
>```python
>#安装torch的.whl文件之前，先安装Cython
>sudo pip3 install Cython
>#根据自己需要的版本，这里我安装的是torch1.6.0
>sudo pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
>```
>
>4、**安装torchvision**
>
>进入其官方GitHub：[GitHub - pytorch/vision: Datasets, Transforms and Models specific to Computer Vision](https://github.com/pytorch/vision/tree/master)
>找到自己需要的版本下载下来，torch和torchvision的版本需要相互对应，比如博主最终安装的是Pytorch1.6.0，所以torchvision版本选择0.7.0.
>
>![image-20220415191920277](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220415191920277.png)
>
>下载torchvision压缩包文件查找步骤如下：
>
>![image-20220415191042231](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220415191042231.png)
>
>![image-20220415191339990](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220415191339990.png)
>
>![image-20220415191329605](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220415191329605.png)
>
>下载好.zip压缩包之后，进入下载目录进行解压，同时进入terminal终端
>
>```python
># cd 你的下载目录下的解压文件夹Downloads/vision-0.7.0对应你的目录
>cd Downloads/vision-0.7.0
>#执行这一步命令，没有报错证明成功安装torchvision(如果有报错参考第5步)  正常安装过程图如图：
>sudo python3 setup.py install
>```
>
>正常执行sudo python3 setup.py install的过程图：
>
>![image-20220419121300956](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419121300956.png)
>
>5、安装torchvision--在setup.py install的时候会有一些报错，根据报错原因(一般就是缺少一些依赖包如pillow之类的，也可能是其他，我的是pillow)，直接执行安装缺少包的命令
>
>![image-20220419120934250](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419120934250.png)
>
>有可能出现的错误，参考网址：https://blog.csdn.net/JulyLH/article/details/123140461
>
>```python
>#安装报错缺少的包,后边不加==就会默认安装最新版本
>pip3 install pillow
>```
>
>6、验证是否安装成功
>
>```python
>#列出目前pip所安装的包，有torch和torchvision包为成功安装
>pip list
>#验证    进入python命令行
>python
>#import没有报错就是安装成功啦！
>>>>import torch
>
>```
>
>至此，Jetson NX中的pytorch环境搭建完毕！安装成功啦！撒花！！！
>
>参考文章：
>
>[1] https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
>
>[2] https://blog.csdn.net/JulyLH/article/details/123140461
>
>[3] https://zhuanlan.zhihu.com/p/461620760
>
>[4]https://download.pytorch.org/whl/torch_stable.html(忽略这个地址，这个地址是PC端的历史torch版本)
>
>**注：**[1]是官方教程，部分不建议采用，torch官网直接给的python38对应的最新版本，不一定适合自己的版本

#### **Python38**

**注：**硬件：jetpack4.6 软件：python=3.8

>如果安装了上面的多版本python，在python3.8创建的虚拟环境中直接==pip install torch==和==pip install torchvision==即可,它会直接从下载源中找到对应jetpack版本和python版本的aarch64的torch,如图：
>
>![image-20220419162233925](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419162233925.png)
>
>![image-20220419162307939](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419162307939.png)

### **tensorflow环境搭建**

**注：**硬件：jetpack4.6   软件：python=3.6，tensorflow=2.5.0+nv21.8

>1、确保已经**安装好Jetpack**在您的NX板子上(正常可以进入Ubuntu系统桌面即为安装)--我的jetpack版本为4.6
>
>>1.1、可以通过安装jetson-stats进行查看NX板子的实时运行情况(sudo 用管理员root身份运行，不一定每一次都需要加)
>
>```python
>sudo jetson_clocks
># 安装 jetson-stats
>sudo -H pip install jetson-stats
># 查看状态(可以实时查看板子CPU、GPU的运行情况，以及jetpack版本信息)
>sudo jtop
>```
>
>2、**安装tensorflow环境系统依赖包**
>
>```python
>#更新apt-get
>sudo apt-get update
>#安装相关依赖包
>sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
>#安装pip3
>sudo apt-get install python3-pip
>sudo pip3 install -U pip testresources setuptools==49.6.0 
>```
>
>3、**安装tensorflow的python依赖包**
>
>**注1：**numpy的版本一定要根据自己的python版本进行安装(我的为python36安装以下版本)，一定要对应好版本，我踩的坑有一部分来自于版本（图为遇到的多次错误）
>
>![image-20220419120813781](H:\zhuomian\AI Edge Computing\Paper--md\NX学习笔记.assets\image-20220419120813781.png)
>
>**注2：**keras_applications1.0.8-这个包可能会在安装时找不到，官网已经镜像源都找不到，可以直接在python中文网https://www.cnpython.com/pypi/keras-applications/download下载好==.whl文件==直接安装（百度直接搜这个包就可以，在python中文网就有）
>
>h5py、gast的版如果是python36或者37 就安装以下版本：
>
>```
>sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
>```
>
>python38版本及以上numpy、h5py、gast安装以下版本：
>
>```python
>sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig h5py==3.1.0
>#官网安装h5py，可借鉴，也可以直接第一步命令安装
># sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
>```
>
>4、**安装tensorflow**
>
>从20.02 TensorFlow版本(tensorflow2.1.0和1.15.2)开始，软件包名称已从tensorflow-gpu 至 tensorflow，自己对应的具体版本可以通过官方文档https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel查询
>
>```python
>#使用第一步安装的jetson-stats，用命令jtop查看jetpack版本,我的版本是jetpack4.6
>sudo jtop
>#安装对应你jetpack版本的tensorflow最好，也可以安装低版本(尽量不要安装高于自己jetpack版本的tensorflow)
># 使用以下命令安装TensorFlow点3命令。该命令将安装与JetPack 4.6兼容的最新版本的TensorFlow
>sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
>
># TensorFlow版本2是最近发布的，与TensorFlow 1.x并不完全向后兼容。如果您希望使用TensorFlow 1.x软件包，可以通过将TensorFlow版本指定为小于2来安装它
>sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 ‘tensorflow<2’
>
># 如果要安装特定版本的JetPack支持的TensorFlow的最新版本，请发出以下命令
>sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow
>```
>
>其中**JP_VERSION**为使用的JetPack版本，例如 42 对于`JetPack 4.2.2`或，33 对于`JetPack 3.3.1`等
>
>5、验证是否安装成功
>
>```python
>#首选查看安装的package是否含有tensorflow等已经安装的包,如果有为正常，没预测需要排查原因
>pip list
>#有的情况，list之后有包，但在python中import之后依旧报错“Illegal instruction (core dumped)”使用参考文章[3]解决
>#进入python中
>python
>#import之后没有报错即为成功
>>>>import tensorflow
>>>>
>```
>
>
>
>参考文章：
>
>[1] https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
>
>[2] https://blog.csdn.net/qq_39567427/article/details/109158918
>
>[3] https://blog.csdn.net/Blang233/article/details/116272746
>
>**注：**[1]是官方教程，部分不建议采用，最新版本，不一定适配自己的版本

## **CUDA10.0和CUDNN7安装**==（通用）==

安装pytorch和tensorflow环境之后，进入任意一个环境或者base环境都可以进行安装cuda和cudnn

CUDA10.2和CUDNN8一样的安装方法，其他版本需要看自己的硬件是否支持。

>首先，进入==/etc/apt==目录下将下载源文件==sources.list==更改或者添加如下下载源
>
>```python
>#进入/etc/apt目录
>cd /etc/apt
>#ls查看当前目录下的文件，有sources.list
>ls
>#用vim编辑工具添加下载源
>vi sources.list
>#进入vim编辑器之后，按键盘上的insert键进行添加如下内容
>deb https://repo.download.nvidia.com/jetson/common r32 main
>deb https://repo.download.nvidia.com/jetson/t210 r32 main
>#编辑完按esc退出编辑模式，使用:wq保存退出（英文冒号）
>```
>
>然后，更新apt-get，进行cuda和cudnn安装
>
>```python
>#更新apt-get
>sudo apt-get update
>#安装cuda 10.0 和 cudnn 7
>sudo apt install cuda-toolkit-10-0
>sudo apt-get install libcudnn7
>```
>
>最后，添加环境变量
>
>```python
>vim ~/.bashrc
>#添加环境变量--将如下三行内容加入
>export CUDA_HOME=/usr/local/cuda-10.0
>export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
>export PATH=/usr/local/cuda-10.0/bin:$PATH
>#激活生效
>source ~/.bashrc
>```
>
>输入命令==nvcc -V==查看cuda信息
>
>```python
>nvcc -V
>```
>
>验证是否可以调用cuda
>
>>1、pytorch环境下
>>
>>```python
>>#进入python
>>python
>>>>>import torch
>>>>>print(torch.cuda.is_available())
>>#输出TRUE即为成功
>>TRUE
>>```
>>
>>2、tensorflow环境下
>>
>>```
>>#进入python
>>python
>>>>>import tensorflow as tf
>>>>>print(tf.test.is_gpu_available())
>>#输出TRUE即为成功调用GPU
>>TRUE
>>```
>
>至此，CUDA10.0和CUDNN7安装已经成功啦！

## TensorRT和Pycuda安装

>```python 
>#安装pycuda
>pip install pycuda
>```
>
>



