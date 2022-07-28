//!Time : 2021年03月04日 14时23分
//!GitHub : https://github.com/Broad-sky
//!Author : Shengping Shen
//!File : segmentation_trt.h
//!About : Practice is perfect in one, formed in thought destoryed

#ifndef _SEGMENTATION_TRT_H_
#define _SEGMENTATION_TRT_H_
#include <iostream>
#include<memory>
#include<NvInfer.h>
#include "NvInferPlugin.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include<fstream>
#include<math.h>
#include "cuda_runtime_api.h"
#include"logging.h"
#include <cudnn.h>
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


class segTRTSigma
{
	template <typename T>
	using segUniquePtr = std::unique_ptr<T, InferDeleter>;

public:
	segTRTSigma();
	~segTRTSigma();

	bool initialModel();
	std::vector<segObject> runModel(cv::Mat& srcimg_gray, bool is_gamma);

private:
	segTRTParams initial_params();
	bool build_model();
	size_t do_inference(cv::Mat & srcimg_gray, std::vector<segObject> & segobj, bool is_gamma);

private:
	segTRTParams sParams;
	std::shared_ptr< nvinfer1::ICudaEngine> sEngine;
	std::shared_ptr< nvinfer1::IExecutionContext> context;

	float * imnormalize(cv::Mat &srcimg);
	size_t get_seg(cv::Mat& srcimg, float* feature_pred_output,
		float* kernel_pred1_output, float* kernel_pred2_output, float* kernel_pred3_output, 
		float* kernel_pred4_output, float* kernel_pred5_output,
		float* cate_pred1_output, float* cate_pred2_output, float* cate_pred3_output,
		float* cate_pred4_output, float* cate_pred5_output, std::vector<segObject> &segobj);
	
	std::vector<Object> nms_sorted_seg(const std::vector<Object>& objects, std::vector<int>& temp_picked, std::string kernel, float sigma);
	void dynamic_conv(float * kernel_pred, float * feature_pred, std::vector<std::vector<Object>> &objects
		, std::vector<Object> &all_picked_kernel);
	void get_class_names();
	void show_result_seg(const cv::Mat &bgr, const std::vector<Object> &objects, std::vector<segObject> & segobj);
private:
	float nms_thresh;
	size_t keep_top_k;

	int target_size_w;
	int target_size_h;

	int ins_w;
	int ins_h;

	int nms_pre;
	float score_thr;
	float mask_thr;
	float update_thr;
	std::string kernel;
	float sigma;
	int max_per_img;
	int num_class;

	std::string model_path;
	std::string save_path;


	std::vector<std::string> class_names;
	std::string names_path;
};

#endif // !_SEGMENTATION_TRT_H_


