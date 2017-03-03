#include "kernel.cuh"
#include "stdio.h"



/*________________________________________________________* 
*														  *
*   		CUDA KERNELS AND ASSOCIATED FUNCTIONS		  *
*														  *
*_________________________________________________________*/

// device Kernel (can only be call by a kernel) :: define the modulo operation
inline __device__ int modulo(int val, int c){
	return (val & (c - 1));
}

// Kernel :: fill an image with a chekcerboard pattern
__global__ void _checkerboard(uchar4 *image, int step, uchar4 color1, uchar4 color2, unsigned int width, unsigned int height, unsigned int imStep)
{
	// get the position of the current pixel
	int x_local = blockIdx.x * blockDim.x + threadIdx.x;
	int y_local = blockIdx.y * blockDim.y + threadIdx.y;

	// exit if the pixel is out of the size of the image
	if (x_local >= width || y_local >= height) return;
	
	// fill the image, alternate the colors
	if (modulo(x_local, step) < (step/2))
		image[y_local * imStep + x_local] = modulo(y_local, step) < (step / 2) ? color1 : color2;
	else
		image[y_local * imStep + x_local] = modulo(y_local, step) < (step / 2) ? color2 : color1;
}

// Function :: fill an image with a chekcerboard pattern
void cuCreateCheckerboard(sl::zed::Mat &image)
{
	// get the image size
	unsigned int width = image.width;
	unsigned int height = image.height;

	// define the block dimension for the parallele computation
	dim3 dimGrid, dimBlock;
	dimBlock.x = 32;
	dimBlock.y = 8;

	dimGrid.x = ceill(width / (float)dimBlock.x);
	dimGrid.y = ceill(height / (float)dimBlock.y);

	// define the size of the square
	int step = 20;

	// define the two colors of the checkerboard
	uchar4 color1 = make_uchar4(250, 250, 250, 255);
	uchar4 color2 = make_uchar4(236, 172, 0, 255);
	
	// call the kernel
	_checkerboard << <dimGrid, dimBlock >> >((uchar4 *)image.data, step, color1, color2, width, height, image.step / sizeof(uchar4));
}

// Kernel :: replace the current image by an other if the depth if above the threshold
__global__ void _croppImage(float* deptharray, float* depth, uchar4 * imageIn, uchar4 * imageOut, uchar4 * mask, float threshold,
	unsigned int width, unsigned int height, unsigned int depthStep, unsigned int imInStep, unsigned int imOutStep, unsigned int maskStep)
{
	// get the position of the current pixel
	int x_local = blockIdx.x * blockDim.x + threadIdx.x;
	int y_local = blockIdx.y * blockDim.y + threadIdx.y;

	// exit if the pixel is out of the size of the image
	if (x_local >= width || y_local >= height) return;

	// get the depth of the current pixel
	float D = depth[y_local * depthStep + x_local];

	// the depth is strickly positive, if not it means that the depth can not be computed
	// the depth should be below the threshold
		
	if ((isfinite(D)) && (D < threshold + 1))// keep the current image if true
		imageOut[y_local * imOutStep + x_local] = imageIn[y_local * imInStep + x_local];
	else // if false : replace current pixel by the pixel of the mask
	    
		imageOut[y_local * imOutStep + x_local] = mask[y_local * maskStep + x_local];
		
	if(D < threshold)
		deptharray[y_local * imOutStep + x_local] = 0;
	else if( threshold + 2 > D && D > threshold )
		deptharray[y_local * imOutStep + x_local] = 1;
	else
		deptharray[y_local * imOutStep + x_local] = 2;
	//printf("imInStep data: %d ; imOutStep data: %d; \n", imInStep, imOutStep);
	
	    
}

// Function :: replace the current image by an other if the depth if above the threshold
void cuCroppImageByDepth(float *array, sl::zed::Mat &depth, sl::zed::Mat &imageLeft, sl::zed::Mat &imageCut, sl::zed::Mat &mask, float threshold)
{
	// get the image size
	unsigned int width = depth.width;
	unsigned int height = depth.height;
	//printf("width: %d ; height: %d ;", width, height);
	//define the array which carry the information
	
//	float *deptharray = (float*) malloc(width*height);
/*
	for(int i=0; i<width; i++){
	    for(int j=0; j<height; j++)
		deptharray[j*height + i] = 0;
	}	
*/
	// define the block dimension for the parallele computation
	dim3 dimGrid, dimBlock;
	dimBlock.x = 32;
	dimBlock.y = 8;

	dimGrid.x = ceill(width / (float)dimBlock.x);
	dimGrid.y = ceill(height / (float)dimBlock.y);
	
	// call the kernel
	_croppImage << <dimGrid, dimBlock >> >((float *) array, (float *)depth.data, (uchar4 *)imageLeft.data, (uchar4 *)imageCut.data, 
		(uchar4 *)mask.data, threshold, width, height,
		depth.step / sizeof(float), imageLeft.step / sizeof(uchar4), imageCut.step / sizeof(uchar4), mask.step / sizeof(uchar4));
}

//__global__ void _maximalSquare(float* matrix,  int *size){ 



__global__ void _maximalSquare(float* matrix, int *size, float* dist){
	
	int big = size[0]*size[0];
	int small =size[1]*size[1];
	int close = 0;
	int far = 0;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idx > 1280  || idy > 720  ) return;
	

	for(int i=0; i<size[0]; i++){
	    for(int j = 0; j <size[0]; j++){
		if(matrix[(j+idy)*1280+ i + idx] >= 1) close++;
	//	if(matrix[(j+idy+size[0]/2-size[1]/2)*1280+i+idx+size[0]/2-size[1]/2] == 2) far++;
	    }	
	}

	for(int i=0; i<size[1]; i++){
	    for(int j = 0; j < size[1]; j++){
		if(matrix[(j+idy+size[0]/2-size[1]/2)*1280+i+idx+size[0]/2-size[1]/2] == 2) far++;
	    }
	}

	int tempX = idx;
	int tempY = idy;
	if(close == big && far == small){
	//if(close == big){
	    tempX = idx + size[0]/2;
	    tempY = idy + size[0]/2;

	    dist[idy*1280+idx] = sqrt(((float)tempX-640)*((float)tempX-640) + ((float)tempY-360)*((float)tempY-360));

	    
	}else{
	    dist[idy*1280+idx] = 65535;
	}
	
	
}


void cuFindMaximalSquare(float *d_arraySquare, int* d_size, float* dist){
	dim3 threadsperBlock(32,8);
	// call the kernel
	dim3 numBlocks(1280/threadsperBlock.x, 720/threadsperBlock.y);
	_maximalSquare << <numBlocks, threadsperBlock >> >((float*) d_arraySquare,  (int*) d_size, (float*) dist);

}



__global__ void _minDist(float* minVal, float* dist){

	extern __shared__ float shared[];

	int tid = threadIdx.x;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 720*1280 ) return;
	
	shared[tid] = dist[i];

	__syncthreads();

	for(unsigned int s = blockDim.x/2; s>0; s>>=1){
	   
	    if(tid < s ){
		if(shared[tid] > shared[tid+s])
		    shared[tid] = shared[tid+s];
	    }	
	    __syncthreads();
	} 
	if(tid == 0) {
	    minVal[blockIdx.x] = shared[0];
	}
}


/*

__device__ void warpReduce(volatile float* sdata, unsigned int tid){
	if(blockDim.x >= 64) sdata[tid] += sdata[tid+32];
	if(blockDim.x >= 32) sdata[tid] += sdata[tid+16];
	if(blockDim.x >= 16) sdata[tid] += sdata[tid+8];
	if(blockDim.x >= 8) sdata[tid] += sdata[tid+4];
	if(blockDim.x >= 2) sdata[tid] += sdata[tid+1];
}


__global__ void _minDist(float* odata, float* idata){
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x*2) + tid;
	unsigned int gridSize = blockDim.x*2*gridDim.x;
	sdata[tid] = 0;
	while(i<1280*720) {sdata[tid] = idata[i] + idata[i+blockDim.x]; i+=gridSize;}
	__syncthreads();
	if(blockDim.x >= 512) {if(tid<256) {sdata[tid] += sdata[tid+256];} __syncthreads();}
	if(blockDim.x >= 256) {if(tid<128) {sdata[tid] += sdata[tid+128];} __syncthreads();}
	if(blockDim.x >= 128) {if(tid<64) {sdata[tid] += sdata[tid+64];} __syncthreads();}

	if(tid < 32) warpReduce(sdata, tid);
	if(tid == 0) odata[blockIdx.x] = sdata[0];
} 
*/

void cuFindMinDist(float* size, float* dist){

	int threads = 720;
	int block = 1280;
		
	_minDist<<<block, threads, sizeof(float)*threads>>>((float*)size, (float*) dist);
}



__global__ void _findPos(float* dist, int* xPos, int* yPos, float* out){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idx > 1280 || idy > 720) return;
	
	double temp1 = 0;
	double temp2 = 0;
	temp1 = (double)dist[1280*idy + idx];
	temp2 = (double)out[0];
	//printf("out value : %f\n", temp2);
	//if(dist[idy*1280+idx]<65535)
	 //   printf("dist value: %f; real value: %f\n", temp1, temp2);
	if(temp1 == temp2){
	    xPos[0] = idx;
	//    printf(" x position: %d\n", idx);
	    yPos[0] = idy;
	//    printf(" y position: %d\n", idy);
	}
}


void cuFindPos(float* d_dist, int* d_xPos, int* d_yPos,float* out){
	dim3 threadsperBlock(16,12);
	
	dim3 numBlocks(1280/threadsperBlock.x, 720/threadsperBlock.y);
	_findPos << <numBlocks, threadsperBlock >> >((float*) d_dist, (int*) d_xPos, (int*) d_yPos, (float*) out);
	
}



































































































