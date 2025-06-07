# hust_code
包含两个工作代码
##  instrument_3d
instrument_3d文件夹包含手术器械三维关键点位姿的数据生成脚本、三维位姿耦合脚本以及blender模型环境文件

这里深度估计和位姿估计的代码用开源的DepthAnything和YOLO实现

## TROI_Segformer
TROI_Segformer是改进的Segformer模型，训练过程需要同时提供图像的胆囊和TROI的掩膜文件

模型修改部分基本在nets/segformer.py中，额外的CAM代码提供注意力热图