# TANet_smoke
Texture-aware network for smoke density estimation.

For training: modify the ```tools/smoke_ori.yaml``` file according to your request before running ```/tools/train_0205.py```.

For testing: run the file ```tools/test2.py```.

The updated models, metrics and other outputs are recorded in ```tools/run/smoke/backbone_name```, where the well-trained model ```(train_epoch_131.pth)``` and results demonstrated in our paper locate.


The original training data comes from the paper "A Wave-Shaped Deep Neural Network for Smoke Density Estimation (IEEE TIP 2020)" (i.e., W-Net), whose datasets and code are available [here](http://staff.ustc.edu.cn/~yfn/index.html). However, we employ randomly synthesis strategy represented by the equation below rather than directly adopting the trainig set provided by W-Net while training our TANet. 

$\boldsymbol{I}(x,y) = \boldsymbol{S}(x,y)\alpha + \boldsymbol{B}(x,y)\alpha$

In the equation, $\boldsymbol{I}(x,y), \boldsymbol{S}(x,y) and \boldsymbol{B}(x,y)$ stand for image pixel, smoke pixel and background pixel at coordinate $(x,y)$. $\alpha$indicates the smoke density value at this point. The smoke and background set can be accessed through [smokeLink](https://mega.nz/file/o4dQlRID#ilTHUkMamK4kEkk8Zygz-jIFWQ1-G8MzCIluMkY1RW0) and [bgLink](https://mega.nz/file/BoFB0DTS#eoc16GDy6o02gqlA7XJOPfCSO7K1bClnr6918tBUtbc).


If this repo does some help, please kindly cite our paper, and further details will be presented in our journal version.
```
@INPROCEEDINGS{10008826,
  author={Xia, Xue and Zhan, Kun and Peng, Yajing and Fang, Yuming},
  booktitle={2022 IEEE International Conference on Visual Communications and Image Processing (VCIP)}, 
  title={Texture-aware Network for Smoke Density Estimation}, 
  year={2022},
  volume={},
  number={},
  pages={1-5},
```


This work was inspired by [SPNet](https://ieeexplore.ieee.org/document/9157204), and its repo lies [here](https://github.com/houqb/SPNet).
```
@inproceedings{hou2020strip,
  title={{Strip Pooling}: Rethinking Spatial Pooling for Scene Parsing},
  author={Hou, Qibin and Zhang, Li and Cheng, Ming-Ming and Feng, Jiashi},
  booktitle={CVPR},
  year={2020}
}
```


----------------------- update on 2025.5.19, journal version ------------------------
1. comprehensive literature review
2. local conv-based and long-range conv-based TA module comparison
3. frequency module embbeded for texture detail preservation
4. extensive experiments
   
```
@article{tanet2025,
  title={Texture‐Aware Network for Enhancing Inner SmokeRepresentation in Visual Smoke Density Estimation,
  author={Xia, Xue and Peng, Yajing and Li, Zichen and Shi, Jinting and Fang, Yuming},
  booktitle={IET Computer Vision},
  year={2025},
  url={https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/cvi2.70023},
}
```
