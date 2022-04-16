

# TensorRT  SOLOv2

​		This project is based on SOLOv2 instance segmentation, and the model is deployed with tensorRT, aiming to attract jade.

### Demo

<img src="./image/result1.jpg" alt="result1" style="zoom:120%;" />

![result3](./image/result3.jpg)

<img src="./image/result2.jpg" alt="result2" style="zoom:80%;" />



### Model

Download link:  https://pan.baidu.com/s/1lX6n44CtXlqdIV0NiS8jXA

Code: fw8f

### Example

**1、go to the build directory.**

```powershell
SOLOv2-TensorRT$ cd build
```

**2、run command.**

```powershell
 ./convertModel ./coco_20200412_permute.onnx -g ./seg_coco_permute.bin
```

**3、run command.**

```powershell
SOLOv2-TensorRT/build$ make clean
```

**4、Compile the project.**

```powershell
SOLOv2-TensorRT/build$ make
[ 33%] Building CXX object CMakeFiles/SOLOv2-TensorRT.dir/src/segmentation_trt.cpp.o
[100%] Linking CXX executable SOLOv2-TensorRT
[100%] Built target SOLOv2-TensorRT
```

**5、run the demo program.**

```powershell
SOLOv2-TensorRT/build$ ./SOLOv2-TensorRT -image_path ../image/1.jpg -save_path demo1.jpg
curr opencv version 3.4.10
num of classes: 80
[04/15/2022-16:03:45] [W] [TRT] TensorRT was linked against cuDNN 8.0.5 but loaded cuDNN 8.0.2
[04/15/2022-16:03:46] [W] [TRT] TensorRT was linked against cuBLAS/cuBLAS LT 11.2.0 but loaded cuBLAS/cuBLAS LT 11.1.0
[04/15/2022-16:03:46] [W] [TRT] TensorRT was linked against cuDNN 8.0.5 but loaded cuDNN 8.0.2
[04/15/2022-16:03:46] [W] [TRT] TensorRT was linked against cuBLAS/cuBLAS LT 11.2.0 but loaded cuBLAS/cuBLAS LT 11.1.0
 0 = 0.851572 at 215 411
 0 = 0.763859 at 98 401
 31 = 0.460805 at 215 528
0.773580 seconds
sh: 1: pause: not found
```

