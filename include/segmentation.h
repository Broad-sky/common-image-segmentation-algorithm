//!Time : 2021年03月04日 14时17分
//!GitHub : https://github.com/Broad-sky
//!Author : Sigma
//!File : segmentation_trt.h
//!About : Practice is perfect in one, formed in thought destoryed

#ifndef _SEGMENTATION_H_
#define _SEGMENTATION_H_
#include<vector>
#include<string>
#include"opencv2/opencv.hpp"

struct segParams
{
	int32_t batchSize;
	int32_t dlaCore;
	bool int8;
	bool fp16;
	bool seria;
	std::vector<std::string> dataDirs;
	std::vector<std::string> inputTensorNames;
	std::vector<std::string> outputTensorNames;

};

struct segTRTParams : public segParams
{
	std::string trtFileName;
};

struct segOnnxParams : public segParams
{
	std::string onnxFileName;
};

struct InferDeleter
{
	template<typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}

};

struct Object {
	int fea_w;
	int fea_h;
	cv::Rect_<float> rect;
	int cx;
	int cy;
	int label;
	float prob;
	cv::Mat mask;
	cv::Mat seg_preds;
};

struct segObject {
	std::string message;
	int target_num;
	cv::Mat segMat;
	cv::Mat gammMat;
	int ret;
};

#endif //!_SEGMENTATION_H_