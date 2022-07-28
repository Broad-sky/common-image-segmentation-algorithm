//!Time : 2021年03月04日 14时42分
//!GitHub : https://github.com/Broad-sky
//!Author : Shengping Shen
//!File : segmentation_trt.h
//!About : Practice is perfect in one, formed in thought destoryed

#include<opencv2/opencv.hpp>
#include<iostream>
#include"segmentation_trt.h"
//..\include\

int main(int argc, char **argv)
{
	if (argc == 5 && std::string(argv[1]) == "-image_path" && std::string(argv[3]) == "-save_path")
	{
		const char *image_path = argv[2]; // input image path
		const char *save_path = argv[4];  // save result image path

		segTRTSigma* seg = new segTRTSigma();
		if (seg->initialModel())  // init model
		{
			cv::Mat srcimg = cv::imread(image_path);
			std::vector<segObject> segobj;
			if (!srcimg.empty())
			{
				clock_t start, finish;
				start = clock();
				segobj = seg->runModel(srcimg, false); // run model
				cv::imwrite(save_path, segobj[0].segMat);
				finish = clock();
				double subtime = (float)(finish - start) / CLOCKS_PER_SEC;
				printf("%f seconds\n", subtime);
			}
		}
		else
		{
			std::cerr << "initial model error" << std::endl;
		}
	}
	else
	{
		std::cerr << "-->arguments not right!" << std::endl;
		std::cerr << "--> ./SOLOv2-TensorRT -image_path ./1.jpg -save_path ./demo1.jpg" << std::endl;
		return -1;
	}
}