#include "zed/Mat.hpp"
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <algorithm>
#include "npp.h"
#include "device_functions.h"
#include <stdint.h>

// CUDA Function :: fill an image with a checkerboard pattern
void cuCreateCheckerboard(sl::zed::Mat &image);

// CUDA Function :: keep the current pixel if its depth is below a threshold
void cuCroppImageByDepth(float* array, sl::zed::Mat &depth, sl::zed::Mat &imageLeft, sl::zed::Mat &imageCut, sl::zed::Mat &mask, float threshold);

// CUDA Function :: find the maximal square size
void cuFindMaximalSquare(float* array, int *d_size, float *dist);
//void cuFindMaximalSquare(float* array, int *d_size);

void cuFindMinDist(float* d_size, float* dist);

void cuFindPos(float* d_dist,int* d_xPos,int* d_yPos, float* out);

