## object-detection-slim    

### 1简介  
自制数据集，使用slim框架和object_detection框架对数据集进行训练，得出一个物体检测模型，并对结果进行验证    
### 2数据集  
数据集包含5个分类，共150张图片，每张图片都做了标注，标注数据格式与voc数据集相同，训练图片样式如下：     
![](train94.jpg '训练图片')   
数据集中物品分类如下：  
- computer  
- monitor  
- scuttlebutt  
- water dispenser  
- drawer chest   
  
数据集中包含文件如下：  
- images（原始图片尺寸1152x864）  
- xmls（使用LabelImg工具标注，每个xml文件对应一个图片文件，每个xml文件里面包含图片中多个物体的位置和种类信息）   
- labels_items（包含所有分类的ID和名称） 
  
LabelImg工具地址：https://github.com/tzutalin/labelImg.git   
注：LabelImg工具对中文的支持不是很好，label尽量使用英文。   

### 3预训练模型  
[mobilenet模型ssd检测框架](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  

### 4代码实现  
1）使用create_pet_tf_record.py将原始数据和label关联，按照格式转换生成TFRecord格式数据集（train.record和val.record）  
2）设置配置文件，加载预训练模型进行finetune，使用数据集进行模型训练和验证  
3）导出训练好的模型，对测试图片进行验证和输出   

### 5验证结果   
![](test.jpg '测试图片')   
![](output.png '结果验证')   

**参考论文**   
[《SSD: Single Shot MultiBox Detector》](https://arxiv.org/abs/1512.02325)    
[《MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications》](https://arxiv.org/abs/1704.04861)     

**模型代码**    
[slim和object_detection框架](https://github.com/tensorflow/models/tree/r1.5/research)    