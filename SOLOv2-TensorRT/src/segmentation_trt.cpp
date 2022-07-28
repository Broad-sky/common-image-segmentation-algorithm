//!Time : 2021年03月04日 14时38分
//!GitHub : https://github.com/Broad-sky
//!Author : Shengping Shen
//!File : segmentation_trt.h
//!About : Practice is perfect in one, formed in thought destoryed

#include"segmentation_trt.h"

#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static Logger gLogger;

segTRTParams segTRTSigma::initial_params()
{
	std::cout<<"curr opencv version " << CVAUX_STR(CV_MAJOR_VERSION) << "." << CVAUX_STR(CV_MINOR_VERSION) 
		<<"." << CVAUX_STR(CV_SUBMINOR_VERSION) << std::endl;//2
	nms_thresh = 0.6;
	target_size_w = 992;
	target_size_h = 1024;
	nms_pre = 500;
	score_thr = 0.1;
	mask_thr = 0.5;
	kernel = "gaussian";
	sigma = 2.0;
	max_per_img = 100;
	num_class = 80;
	ins_w = int(target_size_w / 4);
	ins_h = int(target_size_h / 4);
	std::vector<std::string> dataDirs;
	segTRTParams params;
	params.dataDirs = dataDirs;
	model_path = "seg_coco_permute.bin";
	names_path = "./custom.names";
	save_path = "./demo.jpg";
	params.inputTensorNames.push_back("dummy_input");	
	params.inputTensorNames.push_back("dummy_coord32");	
	params.inputTensorNames.push_back("dummy_coord16");	
	params.inputTensorNames.push_back("dummy_coord8");	
	params.outputTensorNames.push_back("feature_pred");
	params.outputTensorNames.push_back("kernel_pred1");
	params.outputTensorNames.push_back("kernel_pred2");
	params.outputTensorNames.push_back("kernel_pred3");
	params.outputTensorNames.push_back("kernel_pred4");
	params.outputTensorNames.push_back("kernel_pred5");
	params.outputTensorNames.push_back("cate_pred1");	
	params.outputTensorNames.push_back("cate_pred2");	
	params.outputTensorNames.push_back("cate_pred3");	
	params.outputTensorNames.push_back("cate_pred4");	
	params.outputTensorNames.push_back("cate_pred5");	
	params.dlaCore = -1;
	params.int8 = false;
	params.fp16 = false;
	params.batchSize = 1;
	params.seria = false;
	return params;
}

void segTRTSigma::get_class_names()
{
	std::fstream fst;
	fst.open(names_path, std::ifstream::in);
	if (fst.is_open())
	{
		std::string buff{""};
		while (std::getline(fst, buff))
		{
			class_names.push_back(buff);
		}
		for (auto vec: class_names )
		{
			std::cout<<vec;
			/* code */
		}
		
	}
	num_class = class_names.size();
	std::cout << "num of classes: " << num_class << std::endl;
}

segTRTSigma::segTRTSigma()
{
	cudaSetDevice(0);
	sParams = initial_params();
	get_class_names();
}

static inline float sigmoid(float x) {
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float intersection_area(const Object &a, const Object &b, int img_w, int img_h) {
	float area = 0.f;
	for (int y = 0; y < img_h; y = y + 4) {
		for (int x = 0; x < img_w; x = x + 4) {
			const uchar *mp1 = a.mask.ptr(y);
			const uchar *mp2 = b.mask.ptr(y);
			if (mp1[x] == 255 && mp2[x] == 255) area += 1.f;
		}
	}
	return area;
}

static inline float area(const Object &a, int img_w, int img_h) {
	float area = 0.f;
	for (int y = 0; y < img_h; y = y + 4) {
		for (int x = 0; x < img_w; x = x + 4) {
			const uchar *mp = a.mask.ptr(y);
			if (mp[x] == 255) area += 1.f;
		}
	}
	return area;
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
	int i = left;
	int j = right;
	float p = objects[(left + right) / 2].prob;

	while (i <= j)
	{
		while (objects[i].prob > p)
			i++;

		while (objects[j].prob < p)
			j--;

		if (i <= j)
		{
			// swap
			std::swap(objects[i], objects[j]);

			i++;
			j--;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (left < j) qsort_descent_inplace(objects, left, j);
		}
#pragma omp section
		{
			if (i < right) qsort_descent_inplace(objects, i, right);
		}
	}
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
	if (objects.empty())
		return;

	qsort_descent_inplace(objects, 0, objects.size() - 1);
}


std::vector<Object> segTRTSigma::nms_sorted_seg(const std::vector<Object>& objects, std::vector<int>& nms_picked, std::string kernel ,
	float sigma)
{
	nms_picked.clear();
	const int n = objects.size();
	std::vector<float> areas(n);

	for (int i = 0; i < n; i++)
	{
		areas[i] = area(objects[i], ins_w, ins_h);
	}
	for (int i = 0; i < n; i++)
	{
		const Object& a = objects[i];
		int keep = 1;
		for (int j = 0; j < (int)nms_picked.size(); j++)
		{
			const Object& b = objects[nms_picked[j]];
			// intersection over union
			float inter_area = intersection_area(a, b, ins_w, ins_h);
			float union_area = areas[i] + areas[nms_picked[j]] - inter_area;
			float cur_iou = inter_area / union_area;
			if (cur_iou > 0.5)
			{
				keep = 0;
			}
		}
		if (keep)
		{
			nms_picked.push_back(i);
		}
	}
	return objects;
}

void calloc_error()
{
	fprintf(stderr, "Calloc error - possibly out of CPU RAM \n");
	exit(EXIT_FAILURE);
}

void *xcalloc(size_t nmemb, size_t size) {
	void *ptr = calloc(nmemb, size);
	if (!ptr) {
		calloc_error();
	}
	memset(ptr, 0, nmemb * size);
	return ptr;
}

void gemm_nt(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			PUT_IN_REGISTER float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA*A[i*lda + k] * B[j*ldb + k];
			}
			C[i*ldc + j] += sum;
		}
	}
}

void gemm_tn(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			PUT_IN_REGISTER float A_PART = ALPHA * A[k * lda + i];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART*B[k*ldb + j];
			}
		}
	}
}

void gemm_tt(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			PUT_IN_REGISTER float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA*A[i + k*lda] * B[k + j*ldb];
			}
			C[i*ldc + j] += sum;
		}
	}
}

void gemm_nn(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA*A[i*lda + k];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART*B[k*ldb + j];
			}
		}
	}
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	if (BETA != 1) {
		int i, j;
		for (i = 0; i < M; ++i) {
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] *= BETA;
			}
		}
	}
	int t;
#pragma omp parallel for
	for (t = 0; t < M; ++t) {
		if (!TA && !TB)
			gemm_nn(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
		else if (TA && !TB)
			gemm_tn(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
		else if (!TA && TB)
			gemm_nt(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
		else
			gemm_tt(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
	}
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
	int i;
	if (INCX == 1 && ALPHA == 0) {
		memset(X, 0, N * sizeof(float));
	}
	else {
		for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
	}
}


static void filter_ins(const float *cate_pred, std::vector<Object> &picked, std::vector<Object> &all_picked,
	int num_class, float cate_thresh, int fea_w, int fea_h)
{
	int count_ins = 0;
	int num_anchors = fea_w*fea_h;

	for (int i = 0; i < fea_h; i++)
	{
		for (int j = 0; j < fea_w; j++)
		{
			float global_prob = 0;
			int global_cls_idx = 1;
			for (int cls_idx = 0; cls_idx < num_class; cls_idx++)
			{
				if (cate_pred == nullptr)
				{
					std::cout << "cate_pred is null" << std::endl;
				}
				float prob_x = cate_pred[i*fea_w*num_class + num_class * j + cls_idx];
				if (prob_x > global_prob)
				{
					global_prob = prob_x;
					global_cls_idx = cls_idx;
				}
			}
			if (global_prob < cate_thresh)
			{
				continue;
			}
			else
			{
				Object obj;
				obj.fea_w = fea_w;
				obj.fea_h = fea_h;
				obj.rect.x = (int)j;
				obj.rect.y = (int)i;
				obj.prob = global_prob;
				obj.label = global_cls_idx;
				picked.push_back(obj);
				all_picked.push_back(obj);
			}
		}
	}
}

void segTRTSigma::dynamic_conv(float * kernel_pred, float * feature_pred, std::vector<std::vector<Object>> &objects
	, std::vector<Object> &all_picked_kernel)
{
	int c_out = all_picked_kernel.size();
	int output_size{ 0 };
	int stride = 0;
	if (c_out > 0) {
		float * a = kernel_pred;
		float * b = feature_pred;
		int n = ins_h * ins_w;
		output_size = c_out * n;
		float * output = (float*)xcalloc(output_size, sizeof(float));
		fill_cpu(output_size, 0, output, 1);
		int m = c_out;
		int k = 256;
		float * c = output;

// #ifdef GPU
		cublasStatus_t status;

		float *d_w, *d_f, *d_o;
		cudaMalloc((void**)&d_w, sizeof(float)*m*k);
		cudaMalloc((void**)&d_f, sizeof(float)*k*n);
		cudaMalloc((void**)&d_o, sizeof(float)*m*n);

		cublasHandle_t handle;
		cublasCreate(&handle);
		cudaMemcpy(d_w, a, sizeof(float)*m*k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_f, b, sizeof(float)*k*n, cudaMemcpyHostToDevice);

		float alpha = 1, beta = 0;
		cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			n,          
			m,          
			k,          
			&alpha,     
			d_f,        
			n,          
			d_w,        
			k,          
			&beta,      
			d_o,        
			n           
		);

		cudaMemcpy(c, d_o, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
// #elif
// 		gemm(0, 0, m, n, k, 1, a, k, b, n, 0, c, n);   // [9， 256] x [256, 200*200] => [9, 200, 200], 即 mk x kn => mn
// #endif //GPU

		int count_out_channel1 = 0;
		//40, 36, 24, 16, 12
		for (auto iter = all_picked_kernel.cbegin(); iter != all_picked_kernel.cend(); iter++)
		{
			if ((*iter).fea_w==40 || (*iter).fea_w == 36)
			{
				stride = 8;
			}
			if ((*iter).fea_w == 24)
			{
				stride = 16;
			}
			if ((*iter).fea_w == 16 || (*iter).fea_w == 12)
			{
				stride = 32;
			}

			Object obj;
			// mask float uchar
			obj.seg_preds = cv::Mat(ins_h, ins_w, CV_32FC1);
			obj.mask = cv::Mat(ins_h, ins_w, CV_8UC1);
			float seg_scores = 0.f;
			int sum_masks = 0;

			obj.seg_preds = cv::Scalar(0.f);
			obj.mask = cv::Scalar(0);
			for (int ih = 0; ih < ins_h; ih++)
			{
				float *temp_seg_preds = obj.seg_preds.ptr<float>(ih);
				uchar *temp_seg_masks = obj.mask.ptr<uchar>(ih);
				for (int iw = 0; iw < ins_w; iw++)
				{
					float seg_pred = sigmoid((output + count_out_channel1*ins_w*ins_h)[ih*ins_w + iw]);
					if (seg_pred > mask_thr)
					{
						temp_seg_preds[iw] = seg_pred;
						temp_seg_masks[iw] = 255;
						seg_scores += seg_pred;
						sum_masks++;
					}
				}
			}
			count_out_channel1++;
			if (sum_masks < stride) {
				std::cout << "now count_mask<stride! " << "first current ins area: " << sum_masks
					<< " ,stride is: " << stride << " ,mask_thr: " << mask_thr << std::endl;
				continue;
			}

			seg_scores = seg_scores / (float(sum_masks));
			float cate_scores = seg_scores * (*iter).prob;

			obj.prob = cate_scores;
			obj.label = (*iter).label;
			obj.rect.x = (*iter).rect.x;
			obj.rect.y = (*iter).rect.y;
			objects[(*iter).label].push_back(obj);
		}
	}
	std::vector<Object>().swap(all_picked_kernel);
}

void segTRTSigma::show_result_seg(const cv::Mat &bgr, const std::vector<Object> &objects, std::vector<segObject> &segobj) {
	unsigned char colors[80][3] = {
		{ 255, 0, 255 },
		{ 153, 122, 23 },
		{ 85, 162, 199 },
		{ 1, 114, 210 },
		{ 255, 226, 0 },
		{ 0, 18, 255 },
		{ 255, 151, 0 },
		{ 170, 0, 255 },
		{ 0, 255, 56 },
		{ 255, 0, 75 },
		{ 0, 75, 255 },
		{ 0, 255, 169 },
		{ 255, 0, 207 },
		{ 75, 255, 0 },
		{ 207, 0, 255 },
		{ 37, 0, 255 },
		{ 0, 207, 255 },
		{ 94, 0, 255 },
		{ 0, 255, 113 },
		{ 255, 18, 0 },
		{ 255, 0, 56 },
		{ 18, 0, 255 },
		{ 0, 255, 226 },
		{ 170, 255, 0 },
		{ 255, 0, 245 },
		{ 151, 255, 0 },
		{ 132, 255, 0 },
		{ 75, 0, 255 },
		{ 151, 0, 255 },
		{ 0, 151, 255 },
		{ 132, 0, 255 },
		{ 0, 255, 245 },
		{ 255, 132, 0 },
		{ 226, 0, 255 },
		{ 255, 37, 0 },
		{ 207, 255, 0 },
		{ 0, 255, 207 },
		{ 94, 255, 0 },
		{ 0, 226, 255 },
		{ 56, 255, 0 },
		{ 255, 94, 0 },
		{ 255, 113, 0 },
		{ 0, 132, 255 },
		{ 255, 0, 132 },
		{ 255, 170, 0 },
		{ 255, 0, 188 },
		{ 113, 255, 0 },
		{ 245, 0, 255 },
		{ 113, 0, 255 },
		{ 255, 188, 0 },
		{ 0, 113, 255 },
		{ 255, 0, 0 },
		{ 0, 56, 255 },
		{ 255, 0, 113 },
		{ 0, 255, 188 },
		{ 255, 0, 94 },
		{ 255, 0, 18 },
		{ 18, 255, 0 },
		{ 0, 255, 132 },
		{ 0, 188, 255 },
		{ 0, 245, 255 },
		{ 0, 169, 255 },
		{ 37, 255, 0 },
		{ 255, 0, 151 },
		{ 188, 0, 255 },
		{ 0, 255, 37 },
		{ 0, 255, 0 },
		{ 255, 0, 170 },
		{ 255, 0, 37 },
		{ 255, 75, 0 },
		{ 0, 0, 255 },
		{ 255, 207, 0 },
		{ 255, 0, 226 },
		{ 255, 245, 0 },
		{ 188, 255, 0 },
		{ 0, 255, 18 },
		{ 0, 255, 75 },
		{ 0, 255, 151 },
		{ 255, 56, 0 },
		{ 245, 255, 0 }
	};
	cv::Mat image = bgr.clone();
	cv::Mat bgr1 = bgr.clone();
	std::string message("");
	for (size_t i = 0; i < objects.size(); i++) {
		const Object &obj = objects[i];
		if (obj.prob < 0.3)
			continue;
		int img_h = bgr.rows;
		int img_w = bgr.cols;
		cv::Mat seg_preds2;   // rescale
		cv::resize(obj.seg_preds, seg_preds2, cv::Size(img_w, img_h), cv::INTER_LINEAR);

		cv::Mat final_mask = cv::Mat(img_h, img_w, CV_8UC1);
		float sum_mask_y = 0.0f;
		float sum_mask_x = 0.0f;
		int area = 0;
		{
			final_mask = cv::Scalar(0);
			for (int y = 0; y < img_h; y++)
			{
				float *seg_preds_p = seg_preds2.ptr<float>(y);
				uchar *mask_p = final_mask.ptr<uchar>(y);
				for (int x = 0; x < img_w; x++)
				{
					if (seg_preds_p[x]>0.5){
						mask_p[x] = 255;
						sum_mask_y += float(y);
						sum_mask_x += float(x);
						area++;
					}
					else{
						mask_p[x] = 0;
					}
				}
			}
		}
		if (area < 81) continue;
		int cx = int(sum_mask_x / area);
		int cy = int(sum_mask_y / area);
		const  unsigned char *color = colors[i%80];
		std::string text = class_names[obj.label];
		std::string text_sub =text.substr(0, text.size()-1);
		std::cout<<text_sub<< std::string(" ")  << std::to_string(obj.prob)  << std::endl;
		int x = cx;
		int y = cy;
		for (int y = 0; y < bgr.rows; y++) {
			const uchar *mp = final_mask.ptr(y);
			uchar *bgr1_p = bgr1.ptr(y);
			uchar *image_p = image.ptr(y);
			for (int x = 0; x < bgr1.cols; x++) {
				if (mp[x] == 255) {
					image_p[0] = cv::saturate_cast<uchar>(bgr1_p[0] * 0.5 + color[0] * 0.5);
					image_p[1] = cv::saturate_cast<uchar>(bgr1_p[1] * 0.5 + color[1] * 0.5);
					image_p[2] = cv::saturate_cast<uchar>(bgr1_p[2] * 0.5 + color[2] * 0.5);
				}
				image_p += 3;
				bgr1_p += 3;
			}
		}
		cv::putText(image, text_sub, cv::Point(x, y),
			cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
		std::string temp_name = text_sub;
		message = message + temp_name + std::to_string(obj.prob);
	}
	segobj[0].message = message;
	segobj[0].segMat = image;
	segobj[0].target_num = int(objects.size());
}

static float * create_picked_kernel(const float * kernel_pred, std::vector<Object> &kernel_picked) {
	int c_out = kernel_picked.size();
	int count_out_channel = 0;
	int kernel_size = c_out * 256;
	float * weights = new float[kernel_size];
	for (auto iter = kernel_picked.cbegin(); iter != kernel_picked.cend(); iter++)
	{
		for (int i = 0; i < (*iter).fea_h; i++)
		{
			for (int j = 0; j < (*iter).fea_w; j++)
			{
				if ((*iter).rect.x == j && (*iter).rect.y == i)
				{
					memcpy(weights + count_out_channel * 256, kernel_pred + (i*(*iter).fea_w + j) * 256, 256 * sizeof(float));
				}
			}
		}
		count_out_channel++;
	}
	return weights;
}


static void combine_array(float src[], float src1[], float src2[], float src3[], float src4[], float des[], 
	int src_num, int src_num1, int src_num2, int src_num3, int src_num4, int des_num){

	int s1 = src_num + src_num1;
	int s2 = src_num + src_num1 + src_num2;
	int s3 = src_num + src_num1 + src_num2 + src_num3;
	int s4 = src_num + src_num1 + src_num2 + src_num3 + src_num4;

	for (int i = 0; i < des_num; i++)
	{
		if (i<src_num)
		{
			des[i] = src[i];
		}

		if (src_num<= i&&i<s1)
		{
			des[i] = src1[i-src_num];
		}

		if (s1<=i&&i<s2)
		{
			des[i] = src2[i - src_num - src_num1];
		}

		if (s2<= i&&i<s3)
		{
			des[i] = src3[i - src_num - src_num1 - src_num2];
		}

		if (s3<= i&&i<s4)
		{
			des[i] = src4[i - src_num - src_num1 - src_num2 - src_num3];
		}
	}
}


size_t segTRTSigma::get_seg(cv::Mat& srcimg, float* feature_pred_output, float* kernel_pred1_output,
	float* kernel_pred2_output, float* kernel_pred3_output, float* kernel_pred4_output, 
	float* kernel_pred5_output, float* cate_pred1_output, float* cate_pred2_output, 
	float* cate_pred3_output, float* cate_pred4_output, float* cate_pred5_output, std::vector<segObject> &segobj)
{
	int fea_sacles[5] = { 40,36,24,16,12 };
	std::vector<Object> filtered_inses, filtered_ins1, filtered_ins2, filtered_ins3, filtered_ins4, filtered_ins5;
	filter_ins(cate_pred1_output, filtered_ins1, filtered_inses, num_class, score_thr, fea_sacles[0], fea_sacles[0]);
	filter_ins(cate_pred2_output, filtered_ins2, filtered_inses, num_class, score_thr, fea_sacles[1], fea_sacles[1]);
	filter_ins(cate_pred3_output, filtered_ins3, filtered_inses, num_class, score_thr, fea_sacles[2], fea_sacles[2]);
	filter_ins(cate_pred4_output, filtered_ins4, filtered_inses, num_class, score_thr, fea_sacles[3], fea_sacles[3]);
	filter_ins(cate_pred5_output, filtered_ins5, filtered_inses, num_class, score_thr, fea_sacles[4], fea_sacles[4]);
	std::vector<std::vector<Object>> temp_objects;
	temp_objects.resize(num_class);

	float *picked_kernel_pred1 = nullptr;
	float *picked_kernel_pred2 = nullptr;
	float *picked_kernel_pred3 = nullptr;
	float *picked_kernel_pred4 = nullptr;
	float *picked_kernel_pred5 = nullptr;

	picked_kernel_pred1 = create_picked_kernel(kernel_pred1_output, filtered_ins1);
	picked_kernel_pred2 = create_picked_kernel(kernel_pred2_output, filtered_ins2);
	picked_kernel_pred3 = create_picked_kernel(kernel_pred3_output, filtered_ins3);
	picked_kernel_pred4 = create_picked_kernel(kernel_pred4_output, filtered_ins4);
	picked_kernel_pred5 = create_picked_kernel(kernel_pred5_output, filtered_ins5);

	int dynamic_out_channels = (int)filtered_inses.size();
	int total_num_weights = dynamic_out_channels * 256;
	float * kernel_preds = new float[total_num_weights];
	int filtered_ins1_size = filtered_ins1.size();
	int filtered_ins2_size = filtered_ins2.size();
	int filtered_ins3_size = filtered_ins3.size();
	int filtered_ins4_size = filtered_ins4.size();
	int filtered_ins5_size = filtered_ins5.size();

	combine_array(picked_kernel_pred1, picked_kernel_pred2, picked_kernel_pred3, picked_kernel_pred4, picked_kernel_pred5, kernel_preds,
		filtered_ins1_size * 256, filtered_ins2_size * 256, filtered_ins3_size * 256, filtered_ins4_size * 256, filtered_ins5_size * 256,
		total_num_weights);
	dynamic_conv(kernel_preds, feature_pred_output, temp_objects, filtered_inses);

	delete []picked_kernel_pred1;
	delete []picked_kernel_pred2;
	delete []picked_kernel_pred3;
	delete []picked_kernel_pred4;
	delete []picked_kernel_pred5;
	delete[]kernel_preds;
	picked_kernel_pred1 = nullptr;
	picked_kernel_pred2 = nullptr;
	picked_kernel_pred3 = nullptr;
	picked_kernel_pred4 = nullptr;
	picked_kernel_pred5 = nullptr;
	kernel_preds = nullptr;

	std::vector<Object> final_objects;
	for (int i = 0; i < (int)temp_objects.size(); i++)
	{
		std::vector<Object> one_object = temp_objects[i];
		qsort_descent_inplace(one_object);
		std::vector<int> nms_picked;
		std::vector<Object> nms_proposals;
		if (one_object.size()>0)
		{
			nms_sorted_seg(one_object, nms_picked,"gaussian", 2.0);
			for (int j = 0; j < (int)nms_picked.size(); j++)
			{
				int z = nms_picked[j];
				final_objects.push_back(one_object[z]);
			}
		}
	}
	show_result_seg(srcimg, final_objects, segobj);
	return 0;
 }

float * segTRTSigma::imnormalize(cv::Mat& img)
{
	const float means[3] = { 123.675f, 116.28f, 103.53f };
	const float stds[3] = { 58.395f, 57.12f, 57.375f };
	float * blob = new float[img.total() * 3];
	int channels = 3;
	int img_h = target_size_h;
	int img_w = target_size_w;
	for (size_t c = 0; c < channels; c++)
	{
		for (size_t h = 0; h < img_h; h++)
		{
			for (size_t w = 0; w < img_w; w++)
			{
				blob[c * img_w * img_h + h * img_w + w] =
					(((float)img.at<cv::Vec3b>(h, w)[c]) - means[c]) / stds[c];
			}
		}
	}
	return blob;
}


bool segTRTSigma::build_model()
{
	initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	if (!runtime)
	{
		return false;
	}
	char* trtDeSerBuffer{nullptr};
	const std::string engine_file_path(model_path);
	std::ifstream ifs;
	int serLength;
	ifs.open(engine_file_path.c_str(), std::ios::in | std::ios::binary);      // open input file
	if (ifs.is_open())
	{
		ifs.seekg(0, std::ios::end);    // go to the end
		serLength = ifs.tellg();           // report location (this is the length)
		ifs.seekg(0, std::ios::beg);    // go back to the beginning
		trtDeSerBuffer = new char[serLength];    // allocate memory for a buffer of appropriate dimension  
		ifs.read(trtDeSerBuffer, serLength);       // read the whole file into the buffer  
		ifs.close();
	}
	else
	{
		return false;
	}
	sEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtDeSerBuffer, serLength, nullptr), InferDeleter());
	if (!sEngine)
	{
		std::cout << "load engine failed!" << std::endl;
		return false;
	}
	delete[] trtDeSerBuffer;

	context = std::shared_ptr<nvinfer1::IExecutionContext>(sEngine->createExecutionContext(),InferDeleter());
	if (!context)
	{
		return false;
	}
	return true;
}


float * prepare_coord_input(int target_size_w,int target_size_h, int stride)
{
	int pw = int(target_size_w / stride);
	int ph = int(target_size_h / stride);
	float step_w = 2.f / (pw - 1);
	float step_h = 2.f / (ph - 1);
	float * channel_coord3 = new float[pw*ph * 2];
	for (int h = 0; h < ph; h++)
	{
		for (int w = 0; w < pw; w++)
		{
			channel_coord3[h*pw + w] = -1.f + step_w*(float)w;
			channel_coord3[pw*ph + h*pw + w] = -1.f + step_h*(float)h;
		}
	}
	return channel_coord3;
}


size_t segTRTSigma::do_inference(cv::Mat& srcimg, std::vector<segObject> & segobj, bool is_gamma)
{
	if (srcimg.empty())
	{
		return 102;
	}
	cv::Mat img_resize, src_img;
	cv::resize(srcimg, img_resize, cv::Size(target_size_w, target_size_h));
	cv::cvtColor(img_resize, src_img, cv::COLOR_BGR2RGB);
	segObject segt;
	segt.gammMat = srcimg;
	segobj.push_back(segt);
	float* blob{ nullptr };
	blob = imnormalize(src_img);
	if (blob == nullptr)
	{
		return 102;
	}
	float * channel_coord323 = nullptr;
	channel_coord323 = prepare_coord_input(target_size_w, target_size_h, 32);
	float * channel_coord163 = nullptr;
	channel_coord163 = prepare_coord_input(target_size_w, target_size_h, 16);
	float * channel_coord83 = nullptr;
	channel_coord83 = prepare_coord_input(target_size_w, target_size_h, 8);
	assert(sParams.inputTensorNames.size() == 2);
	int mBatchSize = sEngine->getMaxBatchSize();
	int dummy_inputIndex = sEngine->getBindingIndex(sParams.inputTensorNames[0].c_str());
	assert(sEngine->getBindingDataType(dummy_inputIndex) == nvinfer1::DataType::kFLOAT);
	int dummy_coord32_Index = sEngine->getBindingIndex(sParams.inputTensorNames[1].c_str());
	assert(sEngine->getBindingDataType(dummy_coord32_Index) == nvinfer1::DataType::kFLOAT);
	int dummy_coord16_Index = sEngine->getBindingIndex(sParams.inputTensorNames[2].c_str());
	assert(sEngine->getBindingDataType(dummy_coord16_Index) == nvinfer1::DataType::kFLOAT);
	int dummy_coord8_Index = sEngine->getBindingIndex(sParams.inputTensorNames[3].c_str());
	assert(sEngine->getBindingDataType(dummy_coord8_Index) == nvinfer1::DataType::kFLOAT);
	const int feature_pred_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[0].c_str());
	assert(engine.getBindingDataType(feature_pred_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int kernel_pred1_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[1].c_str());
	assert(engine.getBindingDataType(kernel_pred1_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int kernel_pred2_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[2].c_str());
	assert(engine.getBindingDataType(kernel_pred2_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int kernel_pred3_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[3].c_str());
	assert(engine.getBindingDataType(kernel_pred3_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int kernel_pred4_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[4].c_str());
	assert(engine.getBindingDataType(kernel_pred3_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int kernel_pred5_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[5].c_str());
	assert(engine.getBindingDataType(kernel_pred5_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int cate_pred1_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[6].c_str());
	assert(engine.getBindingDataType(cate_pred1_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int cate_pred2_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[7].c_str());
	assert(engine.getBindingDataType(cate_pred2_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int cate_pred3_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[8].c_str());
	assert(engine.getBindingDataType(cate_pred3_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int cate_pred4_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[9].c_str());
	assert(engine.getBindingDataType(cate_pred4_outputIndex) == nvinfer1::DataType::kFLOAT);
	const int cate_pred5_outputIndex = sEngine->getBindingIndex(sParams.outputTensorNames[10].c_str());
	assert(engine.getBindingDataType(cate_pred5_outputIndex) == nvinfer1::DataType::kFLOAT);
	int dummy_input_size = 1;
	auto dummy_input_dims = sEngine->getBindingDimensions(dummy_inputIndex);
	for (int i = 0; i < dummy_input_dims.nbDims; i++)
	{
		dummy_input_size *= dummy_input_dims.d[i];
	}
	int dummy_coord32_input_size = 1;
	auto dummy_coord32_input_dims = sEngine->getBindingDimensions(dummy_coord32_Index);
	for (int i = 0; i < dummy_coord32_input_dims.nbDims; i++)
	{
		dummy_coord32_input_size *= dummy_coord32_input_dims.d[i];
	}
	int dummy_coord16_input_size = 1;
	auto dummy_coord16_input_dims = sEngine->getBindingDimensions(dummy_coord16_Index);
	for (int i = 0; i < dummy_coord16_input_dims.nbDims; i++)
	{
		dummy_coord16_input_size *= dummy_coord16_input_dims.d[i];
	}
	int dummy_coord8_input_size = 1;
	auto dummy_coord8_input_dims = sEngine->getBindingDimensions(dummy_coord8_Index);
	for (int i = 0; i < dummy_coord8_input_dims.nbDims; i++)
	{
		dummy_coord8_input_size *= dummy_coord8_input_dims.d[i];
	}
	int feature_pred_size = 1;
	auto feature_pred_dims = sEngine->getBindingDimensions(feature_pred_outputIndex);
	for (int i = 0; i < feature_pred_dims.nbDims; i++)
	{
		feature_pred_size *= feature_pred_dims.d[i];
	}
	int kernel_pred1_size = 1;
	auto kernel_pred1_dims = sEngine->getBindingDimensions(kernel_pred1_outputIndex);
	for (int i = 0; i < kernel_pred1_dims.nbDims; i++)
	{
		kernel_pred1_size *= kernel_pred1_dims.d[i];
	}
	int kernel_pred2_size = 1;
	auto kernel_pred2_dims = sEngine->getBindingDimensions(kernel_pred2_outputIndex);
	for (int i = 0; i < kernel_pred2_dims.nbDims; i++)
	{
		kernel_pred2_size *= kernel_pred2_dims.d[i];
	}
	int kernel_pred3_size = 1;
	auto kernel_pred3_dims = sEngine->getBindingDimensions(kernel_pred3_outputIndex);
	for (int i = 0; i < kernel_pred3_dims.nbDims; i++)
	{
		kernel_pred3_size *= kernel_pred3_dims.d[i];
	}
	int kernel_pred4_size = 1;
	auto kernel_pred4_dims = sEngine->getBindingDimensions(kernel_pred4_outputIndex);
	for (int i = 0; i < kernel_pred4_dims.nbDims; i++)
	{
		kernel_pred4_size *= kernel_pred4_dims.d[i];
	}
	int kernel_pred5_size = 1;
	auto kernel_pred5_dims = sEngine->getBindingDimensions(kernel_pred5_outputIndex);
	for (int i = 0; i < kernel_pred5_dims.nbDims; i++)
	{
		kernel_pred5_size *= kernel_pred5_dims.d[i];
	}
	int cate_pred1_size = 1;
	auto cate_pred1_dims = sEngine->getBindingDimensions(cate_pred1_outputIndex);
	for (int i = 0; i < cate_pred1_dims.nbDims; i++)
	{
		cate_pred1_size *= cate_pred1_dims.d[i];
	}
	int cate_pred2_size = 1;
	auto cate_pred2_dims = sEngine->getBindingDimensions(cate_pred2_outputIndex);
	for (int i = 0; i < cate_pred2_dims.nbDims; i++)
	{
		cate_pred2_size *= cate_pred2_dims.d[i];
	}
	int cate_pred3_size = 1;
	auto cate_pred3_dims = sEngine->getBindingDimensions(cate_pred3_outputIndex);
	for (int i = 0; i < cate_pred3_dims.nbDims; i++)
	{
		cate_pred3_size *= cate_pred3_dims.d[i];
	}
	int cate_pred4_size = 1;
	auto cate_pred4_dims = sEngine->getBindingDimensions(cate_pred4_outputIndex);
	for (int i = 0; i < cate_pred4_dims.nbDims; i++)
	{
		cate_pred4_size *= cate_pred4_dims.d[i];
	}
	int cate_pred5_size = 1;
	auto cate_pred5_dims = sEngine->getBindingDimensions(cate_pred5_outputIndex);
	for (int i = 0; i < cate_pred5_dims.nbDims; i++)
	{
		cate_pred5_size *= cate_pred5_dims.d[i];
	}
	float * feature_pred_output = new float[feature_pred_size];
	float * kernel_pred1_output = new float[kernel_pred1_size];
	float * kernel_pred2_output = new float[kernel_pred2_size];
	float * kernel_pred3_output = new float[kernel_pred3_size];
	float * kernel_pred4_output = new float[kernel_pred4_size];
	float * kernel_pred5_output = new float[kernel_pred5_size];
	float * cate_pred1_output = new float[cate_pred1_size];
	float * cate_pred2_output = new float[cate_pred2_size];
	float * cate_pred3_output = new float[cate_pred3_size];
	float * cate_pred4_output = new float[cate_pred4_size];
	float * cate_pred5_output = new float[cate_pred5_size];

	void* buffers[15];
	assert(3 * target_size_w * target_size_h == dummy_input_size);
	CHECK(cudaMalloc(&buffers[dummy_inputIndex], dummy_input_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[dummy_coord32_Index], dummy_coord32_input_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[dummy_coord16_Index], dummy_coord16_input_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[dummy_coord8_Index], dummy_coord8_input_size * sizeof(float)));

	CHECK(cudaMalloc(&buffers[feature_pred_outputIndex], feature_pred_size * sizeof(float)));

	CHECK(cudaMalloc(&buffers[kernel_pred1_outputIndex], kernel_pred1_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[kernel_pred2_outputIndex], kernel_pred2_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[kernel_pred3_outputIndex], kernel_pred3_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[kernel_pred4_outputIndex], kernel_pred4_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[kernel_pred5_outputIndex], kernel_pred5_size * sizeof(float)));

	CHECK(cudaMalloc(&buffers[cate_pred1_outputIndex], cate_pred1_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[cate_pred2_outputIndex], cate_pred2_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[cate_pred3_outputIndex], cate_pred3_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[cate_pred4_outputIndex], cate_pred4_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[cate_pred5_outputIndex], cate_pred5_size * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	CHECK(cudaMemcpyAsync(buffers[dummy_inputIndex], blob, dummy_input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[dummy_coord32_Index], channel_coord323, dummy_coord32_input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[dummy_coord16_Index], channel_coord163, dummy_coord16_input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[dummy_coord8_Index], channel_coord83, dummy_coord8_input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

	bool status = context->enqueue(mBatchSize,buffers, stream,nullptr);
	if (!status)
	{
		std::cout << "execute ifer error! " << std::endl;
		return 101;
	}
	CHECK(cudaMemcpyAsync(feature_pred_output, buffers[feature_pred_outputIndex], feature_pred_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(kernel_pred1_output, buffers[kernel_pred1_outputIndex], kernel_pred1_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(kernel_pred2_output, buffers[kernel_pred2_outputIndex], kernel_pred2_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(kernel_pred3_output, buffers[kernel_pred3_outputIndex], kernel_pred3_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(kernel_pred4_output, buffers[kernel_pred4_outputIndex], kernel_pred4_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(kernel_pred5_output, buffers[kernel_pred5_outputIndex], kernel_pred5_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

	CHECK(cudaMemcpyAsync(cate_pred1_output, buffers[cate_pred1_outputIndex], cate_pred1_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(cate_pred2_output, buffers[cate_pred2_outputIndex], cate_pred2_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(cate_pred3_output, buffers[cate_pred3_outputIndex], cate_pred3_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(cate_pred4_output, buffers[cate_pred4_outputIndex], cate_pred4_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(cate_pred5_output, buffers[cate_pred5_outputIndex], cate_pred5_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[dummy_inputIndex]));
	CHECK(cudaFree(buffers[dummy_coord32_Index]));
	CHECK(cudaFree(buffers[dummy_coord16_Index]));
	CHECK(cudaFree(buffers[dummy_coord8_Index]));

	CHECK(cudaFree(buffers[feature_pred_outputIndex]));

	CHECK(cudaFree(buffers[kernel_pred1_outputIndex]));
	CHECK(cudaFree(buffers[kernel_pred2_outputIndex]));
	CHECK(cudaFree(buffers[kernel_pred3_outputIndex]));
	CHECK(cudaFree(buffers[kernel_pred4_outputIndex]));
	CHECK(cudaFree(buffers[kernel_pred5_outputIndex]));

	CHECK(cudaFree(buffers[cate_pred1_outputIndex]));
	CHECK(cudaFree(buffers[cate_pred2_outputIndex]));
	CHECK(cudaFree(buffers[cate_pred3_outputIndex]));
	CHECK(cudaFree(buffers[cate_pred4_outputIndex]));
	CHECK(cudaFree(buffers[cate_pred5_outputIndex]));

	delete[]channel_coord323;
	delete[]channel_coord163;
	delete[]channel_coord83;
	channel_coord323 = nullptr;
	channel_coord163 = nullptr;
	channel_coord83 = nullptr;

	size_t ret = get_seg(srcimg, feature_pred_output,
		kernel_pred1_output, kernel_pred2_output, kernel_pred3_output, kernel_pred4_output, kernel_pred5_output,
		cate_pred1_output,cate_pred2_output, cate_pred3_output, cate_pred4_output, cate_pred5_output, segobj);
	delete[] blob;
	delete[] feature_pred_output;

	delete[] kernel_pred1_output;
	delete[] kernel_pred2_output;
	delete[] kernel_pred3_output;
	delete[] kernel_pred4_output;
	delete[] kernel_pred5_output;

	delete[] cate_pred1_output;
	delete[] cate_pred2_output;
	delete[] cate_pred3_output;
	delete[] cate_pred4_output;
	delete[] cate_pred5_output;
	blob = nullptr;
	feature_pred_output = nullptr;
	kernel_pred1_output = nullptr;
	kernel_pred2_output = nullptr;
	kernel_pred3_output = nullptr;
	kernel_pred4_output = nullptr;
	kernel_pred5_output = nullptr;
	cate_pred1_output = nullptr;
	cate_pred2_output = nullptr;
	cate_pred3_output = nullptr;
	cate_pred4_output = nullptr;
	cate_pred5_output = nullptr;
	return ret;
}

bool segTRTSigma::initialModel()
{
	bool ret = build_model();
	return ret;
}

std::vector<segObject> segTRTSigma::runModel(cv::Mat& srcimg_gray, bool is_gamma)
{
	std::vector<segObject> segobj;
	size_t ret = do_inference(srcimg_gray, segobj, is_gamma);
	return segobj;
}

segTRTSigma::~segTRTSigma()
{
}