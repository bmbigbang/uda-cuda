//Udacity HW 4
//Radix Sorting

#include <thrust/host_vector.h>
#include <iostream>
#include "timer.h"
#include <string>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


__global__ void radox(unsigned int* const d_inputVals, unsigned int* const d_inputPos, 
	                  unsigned int* d_outputVals, unsigned int* const d_outputPos,
	                  const size_t numElems) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
}

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
	const dim3 blockSize(1000, 1, 1);
	const dim3 gridSize((numElems / 1001), 1, 1);
	std::cout << numElems;
	radox <<< gridSize, blockSize >>> (d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);

	cudaDeviceSynchronize(); cudaGetLastError();
}

//Udacity HW4 Driver

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_runtime.h"

//The caller becomes responsible for the returned pointer. This
//is done in the interest of keeping this code as simple as possible.
//In production code this is a bad idea - we should use RAII
//to ensure the memory is freed.  DO NOT COPY THIS AND USE IN PRODUCTION
//CODE!!!
void loadImageHDR(const std::string &filename,
	float **imagePtr,
	size_t *numRows, size_t *numCols)
{
	cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	if (image.channels() != 3) {
		std::cerr << "Image must be color!" << std::endl;
		exit(1);
	}

	if (!image.isContinuous()) {
		std::cerr << "Image isn't continuous!" << std::endl;
		exit(1);
	}

	*imagePtr = new float[image.rows * image.cols * image.channels()];

	float *cvPtr = image.ptr<float>(0);
	for (size_t i = 0; i < image.rows * image.cols * image.channels(); ++i)
		(*imagePtr)[i] = cvPtr[i];

	*numRows = image.rows;
	*numCols = image.cols;
}

void loadImageRGBA(const std::string &filename,
	uchar4 **imagePtr,
	size_t *numRows, size_t *numCols)
{
	cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	if (image.channels() != 3) {
		std::cerr << "Image must be color!" << std::endl;
		exit(1);
	}

	if (!image.isContinuous()) {
		std::cerr << "Image isn't continuous!" << std::endl;
		exit(1);
	}

	cv::Mat imageRGBA;
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	*imagePtr = new uchar4[image.rows * image.cols];

	unsigned char *cvPtr = imageRGBA.ptr<unsigned char>(0);
	for (size_t i = 0; i < image.rows * image.cols; ++i) {
		(*imagePtr)[i].x = cvPtr[4 * i + 0];
		(*imagePtr)[i].y = cvPtr[4 * i + 1];
		(*imagePtr)[i].z = cvPtr[4 * i + 2];
		(*imagePtr)[i].w = cvPtr[4 * i + 3];
	}

	*numRows = image.rows;
	*numCols = image.cols;
}

void saveImageRGBA(const uchar4* const image,
	const size_t numRows, const size_t numCols,
	const std::string &output_file)
{
	int sizes[2];
	sizes[0] = numRows;
	sizes[1] = numCols;
	cv::Mat imageRGBA(2, sizes, CV_8UC4, (void *)image);
	cv::Mat imageOutputBGR;
	cv::cvtColor(imageRGBA, imageOutputBGR, CV_RGBA2BGR);
	//output the image
	cv::imwrite(output_file.c_str(), imageOutputBGR);
}

//output an exr file
//assumed to already be BGR
void saveImageHDR(const float* const image,
	const size_t numRows, const size_t numCols,
	const std::string &output_file)
{
	int sizes[2];
	sizes[0] = numRows;
	sizes[1] = numCols;

	cv::Mat imageHDR(2, sizes, CV_32FC3, (void *)image);

	imageHDR = imageHDR * 255;

	cv::imwrite(output_file.c_str(), imageHDR);
}
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

#include "loadSaveImage.h"
#include <stdio.h>

//simple cross correlation kernel copied from Mike's IPython Notebook
__global__ void naive_normalized_cross_correlation(
	float*         d_response,
	unsigned char* d_original,
	unsigned char* d_template,
	int            num_pixels_y,
	int            num_pixels_x,
	int            template_half_height,
	int            template_height,
	int            template_half_width,
	int            template_width,
	int            template_size,
	float          template_mean
)
{
	int  ny = num_pixels_y;
	int  nx = num_pixels_x;
	int  knx = template_width;
	int2 image_index_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
	int  image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

	if (image_index_2d.x < nx && image_index_2d.y < ny)
	{
		//
		// compute image mean
		//
		float image_sum = 0.0f;

		for (int y = -template_half_height; y <= template_half_height; y++)
		{
			for (int x = -template_half_width; x <= template_half_width; x++)
			{
				int2 image_offset_index_2d = make_int2(image_index_2d.x + x, image_index_2d.y + y);
				int2 image_offset_index_2d_clamped = make_int2(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
				int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

				unsigned char image_offset_value = d_original[image_offset_index_1d_clamped];

				image_sum += (float)image_offset_value;
			}
		}

		float image_mean = image_sum / (float)template_size;

		//
		// compute sums
		//
		float sum_of_image_template_diff_products = 0.0f;
		float sum_of_squared_image_diffs = 0.0f;
		float sum_of_squared_template_diffs = 0.0f;

		for (int y = -template_half_height; y <= template_half_height; y++)
		{
			for (int x = -template_half_width; x <= template_half_width; x++)
			{
				int2 image_offset_index_2d = make_int2(image_index_2d.x + x, image_index_2d.y + y);
				int2 image_offset_index_2d_clamped = make_int2(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
				int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

				unsigned char image_offset_value = d_original[image_offset_index_1d_clamped];
				float         image_diff = (float)image_offset_value - image_mean;

				int2 template_index_2d = make_int2(x + template_half_width, y + template_half_height);
				int  template_index_1d = (knx * template_index_2d.y) + template_index_2d.x;

				unsigned char template_value = d_template[template_index_1d];
				float         template_diff = template_value - template_mean;

				float image_template_diff_product = image_offset_value   * template_diff;
				float squared_image_diff = image_diff           * image_diff;
				float squared_template_diff = template_diff        * template_diff;

				sum_of_image_template_diff_products += image_template_diff_product;
				sum_of_squared_image_diffs += squared_image_diff;
				sum_of_squared_template_diffs += squared_template_diff;
			}
		}


		//
		// compute final result
		//
		float result_value = 0.0f;

		if (sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0)
		{
			result_value = sum_of_image_template_diff_products / sqrt(sum_of_squared_image_diffs * sum_of_squared_template_diffs);
		}

		d_response[image_index_1d] = result_value;
	}
}


__global__ void remove_redness_from_coordinates(
	const unsigned int*  d_coordinates,
	unsigned char* d_r,
	unsigned char* d_b,
	unsigned char* d_g,
	unsigned char* d_r_output,
	int    num_coordinates,
	int    num_pixels_y,
	int    num_pixels_x,
	int    template_half_height,
	int    template_half_width
)
{
	int ny = num_pixels_y;
	int nx = num_pixels_x;
	int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

	int imgSize = num_pixels_x * num_pixels_y;

	if (global_index_1d < num_coordinates)
	{
		unsigned int image_index_1d = d_coordinates[imgSize - global_index_1d - 1];
		ushort2 image_index_2d = make_ushort2(image_index_1d % num_pixels_x, image_index_1d / num_pixels_x);

		for (int y = image_index_2d.y - template_half_height; y <= image_index_2d.y + template_half_height; y++)
		{
			for (int x = image_index_2d.x - template_half_width; x <= image_index_2d.x + template_half_width; x++)
			{
				int2 image_offset_index_2d = make_int2(x, y);
				int2 image_offset_index_2d_clamped = make_int2(min(nx - 1, max(0, image_offset_index_2d.x)), min(ny - 1, max(0, image_offset_index_2d.y)));
				int  image_offset_index_1d_clamped = (nx * image_offset_index_2d_clamped.y) + image_offset_index_2d_clamped.x;

				unsigned char g_value = d_g[image_offset_index_1d_clamped];
				unsigned char b_value = d_b[image_offset_index_1d_clamped];

				unsigned int gb_average = (g_value + b_value) / 2;

				d_r_output[image_offset_index_1d_clamped] = (unsigned char)gb_average;
			}
		}

	}
}




struct splitChannels : thrust::unary_function<uchar4, thrust::tuple<unsigned char, unsigned char, unsigned char> > {
	__host__ __device__
		thrust::tuple<unsigned char, unsigned char, unsigned char> operator()(uchar4 pixel) {
		return thrust::make_tuple(pixel.x, pixel.y, pixel.z);
	}
};

struct combineChannels : thrust::unary_function<thrust::tuple<unsigned char, unsigned char, unsigned char>, uchar4> {
	__host__ __device__
		uchar4 operator()(thrust::tuple<unsigned char, unsigned char, unsigned char> t) {
		return make_uchar4(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), 255);
	}
};

struct combineResponses : thrust::unary_function<float, thrust::tuple<float, float, float> > {
	__host__ __device__
		float operator()(thrust::tuple<float, float, float> t) {
		return thrust::get<0>(t) * thrust::get<1>(t) * thrust::get<2>(t);
	}
};

//we need to save the input so we can remove the redeye for the output
static thrust::device_vector<unsigned char> d_red;
static thrust::device_vector<unsigned char> d_blue;
static thrust::device_vector<unsigned char> d_green;

static size_t numRowsImg;
static size_t numColsImg;
static size_t templateHalfWidth;
static size_t templateHalfHeight;

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
void preProcess(unsigned int **inputVals,
	unsigned int **inputPos,
	unsigned int **outputVals,
	unsigned int **outputPos,
	size_t &numElem,
	const std::string& filename,
	const std::string& templateFilename) {
	//make sure the context initializes ok
	cudaFree(0);

	uchar4 *inImg;
	uchar4 *eyeTemplate;

	size_t numRowsTemplate, numColsTemplate;

	loadImageRGBA(filename, &inImg, &numRowsImg, &numColsImg);
	loadImageRGBA(templateFilename, &eyeTemplate, &numRowsTemplate, &numColsTemplate);

	templateHalfWidth = (numColsTemplate - 1) / 2;
	templateHalfHeight = (numRowsTemplate - 1) / 2;

	//we need to split each image into its separate channels
	//use thrust to demonstrate basic uses

	numElem = numRowsImg * numColsImg;
	size_t templateSize = numRowsTemplate * numColsTemplate;

	thrust::device_vector<uchar4> d_Img(inImg, inImg + numRowsImg * numColsImg);
	thrust::device_vector<uchar4> d_Template(eyeTemplate, eyeTemplate + numRowsTemplate * numColsTemplate);

	d_red.resize(numElem);
	d_blue.resize(numElem);
	d_green.resize(numElem);

	thrust::device_vector<unsigned char> d_red_template(templateSize);
	thrust::device_vector<unsigned char> d_blue_template(templateSize);
	thrust::device_vector<unsigned char> d_green_template(templateSize);

	//split the image
	thrust::transform(d_Img.begin(), d_Img.end(), thrust::make_zip_iterator(
		thrust::make_tuple(d_red.begin(),
			d_blue.begin(),
			d_green.begin())),
		splitChannels());

	//split the template
	thrust::transform(d_Template.begin(), d_Template.end(),
		thrust::make_zip_iterator(thrust::make_tuple(d_red_template.begin(),
			d_blue_template.begin(),
			d_green_template.begin())),
		splitChannels());


	thrust::device_vector<float> d_red_response(numElem);
	thrust::device_vector<float> d_blue_response(numElem);
	thrust::device_vector<float> d_green_response(numElem);

	//need to compute the mean for each template channel
	unsigned int r_sum = thrust::reduce(d_red_template.begin(), d_red_template.end(), 0);
	unsigned int b_sum = thrust::reduce(d_blue_template.begin(), d_blue_template.end(), 0);
	unsigned int g_sum = thrust::reduce(d_green_template.begin(), d_green_template.end(), 0);

	float r_mean = (double)r_sum / templateSize;
	float b_mean = (double)b_sum / templateSize;
	float g_mean = (double)g_sum / templateSize;

	const dim3 blockSize(32, 8, 1);
	const dim3 gridSize((numColsImg + blockSize.x - 1) / blockSize.x, (numRowsImg + blockSize.y - 1) / blockSize.y, 1);

	//now compute the cross-correlations for each channel

	naive_normalized_cross_correlation << <gridSize, blockSize >> >(thrust::raw_pointer_cast(d_red_response.data()),
		thrust::raw_pointer_cast(d_red.data()),
		thrust::raw_pointer_cast(d_red_template.data()),
		numRowsImg, numColsImg,
		templateHalfHeight, numRowsTemplate,
		templateHalfWidth, numColsTemplate,
		numRowsTemplate * numColsTemplate, r_mean);
	cudaDeviceSynchronize(); cudaGetLastError();

	naive_normalized_cross_correlation << <gridSize, blockSize >> >(thrust::raw_pointer_cast(d_blue_response.data()),
		thrust::raw_pointer_cast(d_blue.data()),
		thrust::raw_pointer_cast(d_blue_template.data()),
		numRowsImg, numColsImg,
		templateHalfHeight, numRowsTemplate,
		templateHalfWidth, numColsTemplate,
		numRowsTemplate * numColsTemplate, b_mean);
	cudaDeviceSynchronize(); cudaGetLastError();

	naive_normalized_cross_correlation << <gridSize, blockSize >> >(thrust::raw_pointer_cast(d_green_response.data()),
		thrust::raw_pointer_cast(d_green.data()),
		thrust::raw_pointer_cast(d_green_template.data()),
		numRowsImg, numColsImg,
		templateHalfHeight, numRowsTemplate,
		templateHalfWidth, numColsTemplate,
		numRowsTemplate * numColsTemplate, g_mean);

	cudaDeviceSynchronize(); cudaGetLastError();

	//generate combined response - multiply all channels together


	thrust::device_vector<float> d_combined_response(numElem);

	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
		d_red_response.begin(),
		d_blue_response.begin(),
		d_green_response.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(
			d_red_response.end(),
			d_blue_response.end(),
			d_green_response.end())),
		d_combined_response.begin(),
		combineResponses());

	//find max/min of response

	typedef thrust::device_vector<float>::iterator floatIt;
	thrust::pair<floatIt, floatIt> minmax = thrust::minmax_element(d_combined_response.begin(), d_combined_response.end());

	float bias = *minmax.first;

	//we need to make all the numbers positive so that the students can sort them without any bit twiddling
	thrust::transform(d_combined_response.begin(), d_combined_response.end(), thrust::make_constant_iterator(-bias),
		d_combined_response.begin(), thrust::plus<float>());

	//now we need to create the 1-D coordinates that will be attached to the keys
	thrust::device_vector<unsigned int> coords(numElem);
	thrust::sequence(coords.begin(), coords.end()); //[0, ..., numElem - 1]

													//allocate memory for output and copy since our device vectors will go out of scope
													//and be deleted
	cudaMalloc(inputVals, sizeof(unsigned int) * numElem);
	cudaMalloc(inputPos, sizeof(unsigned int) * numElem);
	cudaMalloc(outputVals, sizeof(unsigned int) * numElem);
	cudaMalloc(outputPos, sizeof(unsigned int) * numElem);

	cudaMemcpy(*inputVals, thrust::raw_pointer_cast(d_combined_response.data()), sizeof(unsigned int) * numElem, cudaMemcpyDeviceToDevice);
	cudaMemcpy(*inputPos, thrust::raw_pointer_cast(coords.data()), sizeof(unsigned int) * numElem, cudaMemcpyDeviceToDevice);
	cudaMemset(*outputVals, 0, sizeof(unsigned int) * numElem);
	cudaMemset(*outputPos, 0, sizeof(unsigned int) * numElem);
}

void postProcess(const unsigned int* const outputVals,
	const unsigned int* const outputPos,
	const size_t numElems,
	const std::string& output_file) {

	thrust::device_vector<unsigned char> d_output_red = d_red;

	const dim3 blockSize(256, 1, 1);
	const dim3 gridSize((40 + blockSize.x - 1) / blockSize.x, 1, 1);

	remove_redness_from_coordinates << <gridSize, blockSize >> >(outputPos,
		thrust::raw_pointer_cast(d_red.data()),
		thrust::raw_pointer_cast(d_blue.data()),
		thrust::raw_pointer_cast(d_green.data()),
		thrust::raw_pointer_cast(d_output_red.data()),
		40,
		numRowsImg, numColsImg,
		9, 9);


	cudaDeviceSynchronize(); cudaGetLastError();

	//combine the new red channel with original blue and green for output
	thrust::device_vector<uchar4> d_outputImg(numElems);

	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
		d_output_red.begin(),
		d_blue.begin(),
		d_green.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(
			d_output_red.end(),
			d_blue.end(),
			d_green.end())),
		d_outputImg.begin(),
		combineChannels());

	thrust::host_vector<uchar4> h_Img = d_outputImg;

	saveImageRGBA(&h_Img[0], numRowsImg, numColsImg, output_file);

	//Clear the global vectors otherwise something goes wrong trying to free them
	d_red.clear(); d_red.shrink_to_fit();
	d_blue.clear(); d_blue.shrink_to_fit();
	d_green.clear(); d_green.shrink_to_fit();
}



void preProcess(unsigned int **inputVals,
	unsigned int **inputPos,
	unsigned int **outputVals,
	unsigned int **outputPos,
	size_t &numElems,
	const std::string& filename,
	const std::string& template_file);

void postProcess(const unsigned int* const outputVals,
	const unsigned int* const outputPos,
	const size_t numElems,
	const std::string& output_file);

void your_sort(unsigned int* const inputVals,
	unsigned int* const inputPos,
	unsigned int* const outputVals,
	unsigned int* const outputPos,
	const size_t numElems);

int main(int argc, char **argv) {
	unsigned int *inputVals;
	unsigned int *inputPos;
	unsigned int *outputVals;
	unsigned int *outputPos;

	size_t numElems;

	std::string input_file;
	std::string template_file;
	std::string output_file;
	std::string reference_file;
	double perPixelError = 0.0;
	double globalError = 0.0;
	bool useEpsCheck = false;

	switch (argc)
	{
	case 3:
		input_file = std::string(argv[1]);
		template_file = std::string(argv[2]);
		output_file = "HW4_output.png";
		break;
	case 4:
		input_file = std::string(argv[1]);
		template_file = std::string(argv[2]);
		output_file = std::string(argv[3]);
		break;
	default:
		std::cerr << "Usage: ./HW4 input_file template_file [output_filename]" << std::endl;
		exit(1);
	}
	//load the image and give us our input and output pointers
	preProcess(&inputVals, &inputPos, &outputVals, &outputPos, numElems, input_file, template_file);

	GpuTimer timer;
	timer.Start();

	//call the students' code
	your_sort(inputVals, inputPos, outputVals, outputPos, numElems);

	timer.Stop();
	cudaDeviceSynchronize(); cudaGetLastError();
	printf("\n");
	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

	if (err < 0) {
		//Couldn't print! Probably the student closed stdout - bad news
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}

	//check results and output the red-eye corrected image
	postProcess(outputVals, outputPos, numElems, output_file);

	// check code moved from HW4.cu
	/****************************************************************************
	* You can use the code below to help with debugging, but make sure to       *
	* comment it out again before submitting your assignment for grading,       *
	* otherwise this code will take too much time and make it seem like your    *
	* GPU implementation isn't fast enough.                                     *
	*                                                                           *
	* This code MUST RUN BEFORE YOUR CODE in case you accidentally change       *
	* the input values when implementing your radix sort.                       *
	*                                                                           *
	* This code performs the reference radix sort on the host and compares your *
	* sorted values to the reference.                                           *
	*                                                                           *
	* Thrust containers are used for copying memory from the GPU                *
	* ************************************************************************* */
	thrust::device_ptr<unsigned int> d_inputVals(inputVals);
	thrust::device_ptr<unsigned int> d_inputPos(inputPos);

	thrust::host_vector<unsigned int> h_inputVals(d_inputVals,
		d_inputVals + numElems);
	thrust::host_vector<unsigned int> h_inputPos(d_inputPos,
		d_inputPos + numElems);

	thrust::host_vector<unsigned int> h_outputVals(numElems);
	thrust::host_vector<unsigned int> h_outputPos(numElems);


	//postProcess(valsPtr, posPtr, numElems, reference_file);

	//compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

	thrust::device_ptr<unsigned int> d_outputVals(outputVals);
	thrust::device_ptr<unsigned int> d_outputPos(outputPos);

	thrust::host_vector<unsigned int> h_yourOutputVals(d_outputVals,
		d_outputVals + numElems);
	thrust::host_vector<unsigned int> h_yourOutputPos(d_outputPos,
		d_outputPos + numElems);


	cudaFree(inputVals);
	cudaFree(inputPos);
	cudaFree(outputVals);
	cudaFree(outputPos);

	return 0;
}
