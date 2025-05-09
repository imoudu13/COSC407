#include <stdio.h>
#include "EasyBMP.h"
#include "EasyBMP.cpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Complex number definition
struct Complex {	// typedef is not required for C++
	float x; 		// real part is represented on x-axis in output image
	float y; 		// imaginary part is represented by y-axis in output image
};

//Function declarations
void compute_julia(const char*, int, int);
void save_image(uchar4*, const char*, int, int);
Complex add(Complex, Complex);
Complex mul(Complex, Complex);
float mag(Complex);

//function declarations for CUDA implementation
void compute_julia_cuda(const char* filename, int width, int height); // GPU version
__device__ Complex add_device(Complex c1, Complex c2);
__device__ Complex mul_device(Complex c1, Complex c2);
__device__ float  mag_device(Complex c);

//main function
int main(void) {
	char* name = "test.bmp";
	// compute_julia(name, 3000, 3000);	//width x height
    compute_julia_cuda(name, 3000, 3000);
	printf("Finished creating %s.\n", name);
	return 0;
}

__device__ Complex add_device(Complex c1, Complex c2) {
    Complex r;
    r.x = c1.x + c2.x;
    r.y = c1.y + c2.y;
    return r;
}
__device__ Complex mul_device(Complex c1, Complex c2) {
    Complex r;
    r.x = c1.x * c2.x - c1.y * c2.y;
    r.y = c1.x * c2.y + c2.x * c1.y;
    return r;
}
__device__ float mag_device(Complex c) {
    return sqrtf(c.x * c.x + c.y * c.y);
}

// kernel: one thread per pixel
__global__ void julia_kernel(uchar4* d_pixels,
                             int width, int height,
                             int max_iterations, int infinity,
                             float x_min, float y_min,
                             float x_incr, float y_incr,
                             Complex c)
{
    // 2D thread location:
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // do not go out of bounds:
    if (col >= width || row >= height) return;

    Complex z;
    z.x = x_min + col * x_incr;
    z.y = y_min + row * y_incr;

    int n = 0;
    do {
        z = add_device(mul_device(z, z), c);
    } while (mag_device(z) < infinity && n++ < max_iterations);

    if (n == max_iterations) {
        d_pixels[col + row * width] = { 0, 0, 0, 0 };
    } else {
        unsigned char hue = (unsigned char)(255 * sqrtf((float)n / max_iterations));
        d_pixels[col + row * width] = { hue, hue, hue, 255 };
    }
}

// parallel computation
void compute_julia_cuda(const char* filename, int width, int height)
{
    // 1. allocate memory on the host
    uchar4* h_pixels = (uchar4*)malloc(width * height * sizeof(uchar4));

    // 2. allocate memory on the device
    uchar4* d_pixels;
    cudaMalloc(&d_pixels, width * height * sizeof(uchar4));

    int max_iterations = 400;
    int infinity = 20;
    Complex c = { 0.285f, 0.01f };

    float w = 4.f;
    float h_f = w * height / width;
    float x_min  = -w / 2.f;
    float y_min  = -h_f / 2.f;
    float x_incr = w / width;
    float y_incr = h_f / height;

    // 3. configure our kernel launch parameters:
    dim3 blockSize(16, 16);
    dim3 gridSize( (width + blockSize.x - 1) / blockSize.x,
                   (height + blockSize.y - 1) / blockSize.y );

    // 4. launch kernel
    julia_kernel<<<gridSize, blockSize>>>(d_pixels, width, height, max_iterations, infinity, x_min, y_min, x_incr, y_incr, c);

    cudaDeviceSynchronize(); // wait for GPU to finish

    // 5. copy the results back to host
    cudaMemcpy(h_pixels, d_pixels, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // 6. free device memory
    cudaFree(d_pixels);

    // 7.save image on host (same function as before)
    save_image(h_pixels, filename, width, height);

    // 8. free host memory
    free(h_pixels);
}

// serial implementation of Julia set
void compute_julia(const char* filename, int width, int height) {
	//create output image
	uchar4 *pixels = (uchar4*)malloc(width * height * sizeof(uchar4));	//uchar4 is a CUDA type representing a vector of 4 chars

	//PROBLEM SETTINGS (marked by '******')
	// **** Accuracy ****: lower values give less accuracy but faster performance
	int max_iterations = 400;
	int infinity = 20;													//used to check if z goes towards infinity

	// ***** Shape ****: other values produce different patterns. See https://en.wikipedia.org/wiki/Julia_set
	Complex c = { 0.285, 0.01 }; 										//the constant in z = z^2 + c

	// ***** Size ****: higher w means smaller size
	float w = 4;
	float h = w * height / width;										//preserve aspect ratio

	// LIMITS for each pixel
	float x_min = -w / 2, y_min = -h / 2;
	float x_incr = w / width, y_incr = h / height;
	
	//****************************************************
	//REQ: Parallelize the following for loop using CUDA 
	//****************************************************
	for (int row = 0; row < height; row++) {						// For each pixel in image, compute pixel color
		for (int col = 0; col < width; col++) {
			Complex z;
			z.x = x_min + col * x_incr;
			z.y = y_min + row * y_incr;

			//iteratively compute z = z^2 + c and check if z goes to infinity
			int n = 0;
			do{
				z = add(mul(z, z), c);								// z = z^2 + c
			} while (mag(z) < infinity && n++ < max_iterations);	// keep looping until z->infinity or we reach max_iterations
			
			// color each pixel based on above loop
			if (n == max_iterations) {								// if we reach max_iterations before z reaches infinity, pixel is black 
				pixels[col + row * width] = { 0,0,0,0 };
			} else {												// if z reaches infinity, pixel color is based on how long it takes z to go to infinity
				unsigned char hue = (unsigned char)(255 * sqrt((float)n / max_iterations));
				pixels[col + row * width] = { hue,hue,hue,255 };
			}
		}
	}
	
	//Write output image to a file (DO NOT parallelize this function)
	save_image(pixels, filename, width, height);

	//free memory
	free(pixels);
}

void save_image(uchar4* pixels, const char* filename, int width, int height) {
	BMP output;
	output.SetSize(width, height);
	output.SetBitDepth(24);
	// save each pixel to output image
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			uchar4 color = pixels[col + row * width];
			output(col, row)->Red = color.x;
			output(col, row)->Green = color.y;
			output(col, row)->Blue = color.z;
		}
	}
	output.WriteToFile(filename);
}

Complex add(Complex c1, Complex c2) {
	return{ c1.x + c2.x, c1.y + c2.y };
}

Complex mul(Complex c1, Complex c2) {
	return{ c1.x * c2.x - c1.y * c2.y, c1.x * c2.y + c2.x * c1.y };
}

float mag(Complex c) {
	return (float)sqrt((double)(c.x * c.x + c.y * c.y));
}