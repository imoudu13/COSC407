#include "EasyBMP.h"
#include "EasyBMP.cpp"
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>	// for uchar4 struct

#define MIN(x,y) (  (y) ^ (((x) ^ (y)) & -((x) < (y))) )
#define MAX(x,y) (  (x) ^ (((x) ^ (y)) & -((x) < (y))) )

//****************************************************************************************************************
// PARALLEL FUNCTIONS
//****************************************************************************************************************
	/*
	TODO: 	Provide CUDA implementation for parallelizing the two SERIAL functions: convolution_8bits and convolution_32Bits
			Make sure to check for errors from CUDA API calls and from Kernel Launch.
			Also, time your parallel code and compute the speed-up.
	*/


//****************************************************************************************************************
// SERIAL FUNCTIONS
//****************************************************************************************************************

//	This function applies the convolution kernel (denoted by filter) to every pixel of the input image (image_in)
//	constraints: image_in and image_out have 8-bit pixels (e.g., grayscale images, only one color channel, etc)

void convolution_8bits(const unsigned char* const image_in, unsigned char* const image_out, const int height, const int width, const float *filter, const int filter_width){
	//only filters with width = odd_number are allowed
	if (filter_width % 2 == 0){
		//you don't have to print anything when running on kernel.
		fprintf(stderr,"Filters with even width are not supported yet. Program terminated!\n");
		exit(1);
	}
	//Apply the filter to every image pixel (col, row)
	for (int row = 0; row < height; ++row)
		for (int col = 0; col < width; ++col) {
			float sum = 0.0f;
			//Having the filter centered at image pixel (col,row), multiply every filter pixel by the corresponding image pixel, and find the sum
			for (int row_f = -filter_width / 2; row_f <= filter_width / 2; ++row_f)
				for (int col_f = -filter_width / 2; col_f <= filter_width / 2; ++col_f) {
					//get the value of the image pixel for the current filter pixel. If out of boundary, use boundary pixels
					int row_i = MIN(MAX(row + row_f, 0), (height - 1));
					int col_i = MIN(MAX(col + col_f, 0), (width - 1));
					float pxl_image = image_in[row_i * width + col_i];
					//get the value for the current filter
					float pxl_filter = filter[(row_f + filter_width / 2) * filter_width + col_f + filter_width / 2];
					//multiply image_pixel by filter_pixel and add to final value of the image pixel
					sum += pxl_image * pxl_filter;
				}
			//final value of image pixel(col, row) = the sum of every filter pixel multiplied by the corresponding image pixels around image pixel(col,row)
			image_out[row * width + col] = sum;
		}
}

//	This function applies the convolution kernel (denoted by filter) to every pixel of the input image (image_in)
//	Constraints:- Both image_in and image_out are in RGBA format (32-bit pixels as uchar4)
//				- Filter is a square matrix (float) and its width is odd number. The sum of all its values is 1 (normalized)

void convolution_32bits( const uchar4* const image_in, uchar4 *const image_out, int height, int width, const float* const filter, const int filter_width){
	//break the input image (uchar4 matrix) into 4 channels (four char matrices): Red, Green, Blue, and Alpha
	unsigned char *R_in = new unsigned char[width * height];
	unsigned char *G_in = new unsigned char[width * height];
	unsigned char *B_in = new unsigned char[width * height];
	unsigned char *A_in = new unsigned char[width * height];
	for (int i = 0; i < width * height; ++i) {	//break each pixel in input image
		uchar4 pxl = image_in[i];
		R_in[i] = pxl.x;
		G_in[i] = pxl.y;
		B_in[i] = pxl.z;
		A_in[i] = pxl.w;
	}

	//perform 8-bit convolution for each 8-bit image channel
	unsigned char *R_out = new unsigned char[width * height];
	convolution_8bits(R_in, R_out, height, width, filter, filter_width);

	unsigned char *G_out = new unsigned char[width * height];
	convolution_8bits(G_in, G_out, height, width, filter, filter_width);

	unsigned char *B_out = new unsigned char[width * height];
	convolution_8bits(B_in, B_out, height, width, filter, filter_width);

	unsigned char *A_out = new unsigned char[width * height];
	convolution_8bits(A_in, A_out, height, width, filter, filter_width);

	//merge the four channels into one output image of type uchar4
	for (size_t i = 0; i < height * width; ++i)
		image_out[i] = make_uchar4(R_out[i], G_out[i], B_out[i], A_out[i]);
}

//**************************************************************
//No need to parallelize any of the functions below this comment
//**************************************************************

//This function reads a BMP image using the EasyBMP library and returns a 1D array representing the RGBA values of the image pixels
//image_out->x is Red, image_out->y is Green, image_out->z is Blue, image_out->w is Alpha
//how to use:	1- in the calling function, declare these variables:	uchar4* img = NULL;	int width = 0, height = 0;
//				2- then call this function								readBMP(filename, &img, &width, &height);
void readBMP(const char* FileName, uchar4 **image_out, int* width, int* height){
	BMP img;
	img.ReadFromFile(FileName);
	*width = img.TellWidth();
	*height = img.TellHeight();
	uchar4 *const img_uchar4 = (uchar4*)malloc(*width * *height * sizeof(int));
	// save each pixel to image_out as uchar4 in row-major format
	for (int row = 0; row <*height; row++)
		for (int col = 0; col < *width; col++)
			img_uchar4[col + row * *width] = make_uchar4(img(col, row)->Red, img(col, row)->Green, img(col, row)->Blue, img(col, row)->Alpha);	//use row-major
	*image_out = img_uchar4;
}

//This function writes a BMP image using the EasyBMP library
//how to use: in the calling function, call		writeBMP(destination_filename, source_image_array, width, height);
void writeBMP(const char* FileName, uchar4 *image, int width, int height){
	BMP output;
	output.SetSize(width, height);
	output.SetBitDepth(24);
	// save each pixel to the output image
	for (int row = 0; row < height; row++){		//for each row
		for (int col = 0; col <  width; col++){	//for each col
			uchar4 rgba = image[col + row * width];
			output(col, row)->Red = rgba.x;
			output(col, row)->Green = rgba.y;
			output(col, row)->Blue = rgba.z;
			output(col, row)->Alpha = rgba.w;
		}
	}
	output.WriteToFile(FileName);

}

//Normalize image filter (sum of all values should be 1)
// the filter is a 2D float array
void normalizeFilter(float* filter, int width){
	//find the sum
	float sum = 0;
	for (int i = 0; i < width*width; i++)
		sum += filter[i];
	//normalize
	for (int i = 0; i < width*width; i++)
		filter[i] /= sum;
}

//this Function reads the convolution-filter image
//Contrasting: Filter is 32 bit RGPA image. The filter must be sqaure. Filter width must be an odd number
float* readFilter(const char* filter_image_name, int* filter_width){
	int filterHeight;	//for testing that height = width
	//read filter image as 32 bit RGPA bitmap and check the constraints (square, odd width)
	uchar4* filterImageUchar;
	readBMP(filter_image_name, &filterImageUchar, filter_width, &filterHeight);
	if (*filter_width != filterHeight || *filter_width % 2 == 0){
		fprintf(stderr, "Non-square filters or filters with even width are not supported yet. Program terminated!\n");
		exit(1);
	}
	//convert every pixel to a float number representing its grayscale intensity. Formula used is 0.21 R + 0.72 G + 0.07 B
	float* filter = (float*)malloc(*filter_width * *filter_width * sizeof(float));
	for (int i = 0; i < *filter_width * *filter_width; i++){
		uchar4 element = filterImageUchar[i];
		filter[i] = 0.21 * element.x + 0.72 * element.y + 0.07 * element.z;
	}
	//Normalization makes sure that the sum of all values in the filter is 1
	normalizeFilter(filter, *filter_width);
	//return result
	return filter;
}

void checkForGPU() {
    // This code attempts to check if a GPU has been allocated
    // Colab notebooks technically have access to NVCC and will compile and
    // execute CPU/Host code, however, GPU/Device code will silently fail.
    // To prevent such situations, this code will warn the user.
    int count;
    cudaGetDeviceCount(&count);
    if (count <= 0 || count > 100) {
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("->WARNING<-: NO GPU DETECTED ON THIS COLLABORATE INSTANCE.\n");
        printf("IF YOU ARE ATTEMPTING TO RUN GPU-BASED CUDA CODE, YOU SHOULD CHANGE THE RUNTIME TYPE!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
}

void serial(){
	int filter_width;
	const char* filter_image_name = "./A7/filter_blur_21.bmp";	//filter width = 21 pixels
	const char* image_in_name = "./A7/okanagan.bmp";
	const char* image_out_name = "okanagan_blur.bmp";

	//load filter
	float* filter = readFilter(filter_image_name, &filter_width);
	printf("Filter loaded...\n");

	//load input image
	int width, height;
	uchar4* image_in;
	readBMP(image_in_name, &image_in, &width, &height);	//image_in will have all pixel information, each pixel as uchar4
	printf("Input image loaded...\n");

	//apply convolution filter to input image
	uchar4* image_out = (uchar4*)malloc(width*height*sizeof(uchar4));	//reserve space in the memory for the output image
	printf("Applying the convolution filter...\n");
	int t = clock();
	convolution_32bits(image_in, image_out, height, width, filter, filter_width);	//filter applied to image_in, results saved in image_out
	t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
	printf("Convolution filter applied. Time taken: %d.%d seconds\n", t / 1000, t % 1000);

	//save results to output image
	writeBMP(image_out_name, image_out, width, height);
	printf("Output image saved.\nProgram finished!\n");
}


// PARALLEL PORTION

// this function does the error checking
#define CUDA_CHECK(call) \
do { \
    cudaError_t _err = call; \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

__global__ void convolution_8bits_kernel(const unsigned char* d_in, unsigned char* d_out, int height, int width, const float* d_filter, int filter_width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

	// make sure it stays within bounds
    if (col >= width || row >= height) {
		return;
	}

    int half = filter_width / 2;
    float sum = 0.0f;

	// apply the filter to every pixel just like serial
    for (int row_f = -half; row_f <= half; ++row_f) {
        for (int col_f = -half; col_f <= half; ++col_f) {
            int r = MIN(MAX(row + row_f, 0), height - 1);
            int c = MIN(MAX(col + col_f, 0), width - 1);
            float pxl_image  = (float)d_in[r * width + c];
            float pxl_filter = d_filter[(row_f + half) * filter_width + (col_f + half)];
            sum += pxl_image * pxl_filter;
        }
    }
    d_out[row * width + col] = (unsigned char)(sum);
}

void convolution_8bits_parallel(const unsigned char* image_in, unsigned char* image_out, int height, int width, const float* filter, int filter_width) {
    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;
    float* d_filter = nullptr;
    size_t img_size = width * height * sizeof(unsigned char);
    size_t filter_size = filter_width * filter_width * sizeof(float);

	// allocate memory on the gpu for the above variables
    CUDA_CHECK(cudaMalloc(&d_in, img_size));
    CUDA_CHECK(cudaMalloc(&d_out, img_size));
    CUDA_CHECK(cudaMalloc(&d_filter, filter_size));
    
	// copy the memory to the gpu
	CUDA_CHECK(cudaMemcpy(d_in, image_in, img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice));
    
	// create the grid
	dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
	// launching the kernel
	convolution_8bits_kernel<<<grid, block>>>(d_in, d_out, height, width, d_filter, filter_width);
    
	// cathc any errors
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

	// sync
    CUDA_CHECK(cudaDeviceSynchronize());
    
	// copy memory back to cpu
	CUDA_CHECK(cudaMemcpy(image_out, d_out, img_size, cudaMemcpyDeviceToHost));
    
	// free memory on the gpu
	CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_filter));
}

// this function is the same as the serial except at the end deallocated memory that was created for the R, G, B, and A arrays
void convolution_32bits_parallel(const uchar4* image_in, uchar4* image_out, int height, int width, const float* filter, int filter_width) {
    unsigned char *R_in = new unsigned char[width * height];
    unsigned char *G_in = new unsigned char[width * height];
    unsigned char *B_in = new unsigned char[width * height];
    unsigned char *A_in = new unsigned char[width * height];

    for (int i = 0; i < width * height; ++i) {
        R_in[i] = image_in[i].x;
        G_in[i] = image_in[i].y;
        B_in[i] = image_in[i].z;
        A_in[i] = image_in[i].w;
    }

    unsigned char *R_out = new unsigned char[width * height];
    unsigned char *G_out = new unsigned char[width * height];
    unsigned char *B_out = new unsigned char[width * height];
    unsigned char *A_out = new unsigned char[width * height];
    convolution_8bits_parallel(R_in, R_out, height, width, filter, filter_width);
    convolution_8bits_parallel(G_in, G_out, height, width, filter, filter_width);
    convolution_8bits_parallel(B_in, B_out, height, width, filter, filter_width);
    convolution_8bits_parallel(A_in, A_out, height, width, filter, filter_width);

    for (int i = 0; i < width * height; ++i) {
        image_out[i] = make_uchar4(R_out[i], G_out[i], B_out[i], A_out[i]);
    }

	// deallocate memory that was saved for arrays initially
    delete[] R_in;
    delete[] G_in;
    delete[] B_in;
    delete[] A_in;
    delete[] R_out;
    delete[] G_out;
    delete[] B_out;
    delete[] A_out;
}

void parallel() {
    const char* filter_image_name = "./A7/filter_blur_21.bmp";
    const char* image_in_name = "./A7/okanagan.bmp";
    const char* image_out_name = "okanagan_blur_parallel.bmp";
    int filter_width;
    float* filter = readFilter(filter_image_name, &filter_width);
    printf("Filter loaded (width=%d)\n", filter_width);
    int width, height;
    uchar4* image_in  = nullptr;
    readBMP(image_in_name, &image_in, &width, &height);
    printf("Input image loaded (%dx%d)\n", width, height);
    uchar4* image_out = (uchar4*)malloc(width * height * sizeof(uchar4));
    int start = clock();
    convolution_32bits_parallel(image_in, image_out, height, width, filter, filter_width);
    int duration_ms = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    printf("Parallel convolution time: %d.%03d seconds\n", duration_ms / 1000, duration_ms % 1000);
    writeBMP(image_out_name, image_out, width, height);
    printf("Parallel output image saved: %s\n", image_out_name);
    free(image_in);
    free(image_out);
    free(filter);
}

//MAIN: testing convolution with a blur filter
int main(){
  checkForGPU();
	serial();
	parallel();
}