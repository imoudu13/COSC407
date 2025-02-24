#include "qdbmp.h"
#include <stdio.h>
#include <omp.h>

typedef enum {desaturate, negative} ImgProcessing;

/* Creates a negative or desaturated image of the input bitmap file */
int main() {
    const char* inFile = "okanagan.bmp";
    const char* outFile = "okanagan_processed.bmp";
    const ImgProcessing processingType = desaturate; // or negative

    UCHAR r, g, b;
    UINT width, height;
    BMP* bmp;

    /* Read an image file */
    bmp = BMP_ReadFile(inFile);
    BMP_CHECK_ERROR(stdout, -1);

    /* Get image's dimensions */
    width = BMP_GetWidth(bmp);
    height = BMP_GetHeight(bmp);

    double t = omp_get_wtime();

    /* Parallel region */
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

		printf("number of threads: %d\n", thread_id);
        /* Manually divide work */
        #pragma omp parallel for
        for (UINT x = thread_id; x < width; x += num_threads) {
            for (UINT y = 0; y < height; ++y) {
                BMP_GetPixelRGB(bmp, x, y, &r, &g, &b);

                if (processingType == negative) {
                    BMP_SetPixelRGB(bmp, x, y, 255 - r, 255 - g, 255 - b);
                } else if (processingType == desaturate) {
                    UCHAR gray = r * 0.3 + g * 0.59 + b * 0.11;
                    BMP_SetPixelRGB(bmp, x, y, gray, gray, gray);
                }
            }
        }
    }

    /* Calculate and print processing time */
    t = 1000 * (omp_get_wtime() - t);
    printf("Finished image processing in %.1f ms.\n", t);

    /* Save result */
    BMP_WriteFile(bmp, outFile);
    BMP_CHECK_ERROR(stdout, -2);

    /* Free all memory allocated for the image */
    BMP_Free(bmp);

    return 0;
}
