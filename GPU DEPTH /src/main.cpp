///////////////////////////////////////////////////////////////////////////
//Followed the instruction from the ZED Stereo Lab and code guides
//Bing Zhang
//UAV Object Detection
///////////////////////////////////////////////////////////////////////////



//standard include
#include <stdio.h>
#include <string.h>
#include <chrono>

//ZED include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

//OpenCV include
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Cuda functions include
#include "kernel.cuh"

// Cuda time record
#include "time.h"
#include "cuda_runtime.h"
#include "ctime"

using namespace sl::zed;
using namespace std;



int main(int argc, char **argv) {

    if (argc > 2) {
        std::cout << "Only the path of an image can be passed in arg" << std::endl;
        return -1;
    }
    
    Camera* zed = new Camera(HD720);

    bool loadImage = false;
    if (argc == 2)
        loadImage = true;

    InitParams parameters;
    parameters.unit = UNIT::METER; // this sample is designed to work in METER
    ERRCODE err = zed->init(parameters);
    
    // ERRCODE display
    cout << errcode2str(err) << endl;

    // Quit if an error occurred
    if (err != SUCCESS) {
        delete zed;
        return 1;
    }

    // print on screen the keys that can be used
    bool printHelp = false;
    std::string helpString = "[p] increase distance, [m] decrease distance, [q] quit";

    // get width and height of the ZED images
    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;



    // CUDA square calculation setting
    float *array = (float*)malloc(width*height*sizeof(float));
    float *out = (float*)malloc(sizeof(float)*1280);
    memset(out, 65535, sizeof(float)*1280);
    float *d_array;

    cudaMalloc(&d_array, width*height*sizeof(float));

    float *dist = (float*)malloc(width*height*sizeof(float));
    


    float *d_arraySquare;
    cudaMalloc(&d_arraySquare, width*height*sizeof(float));
    float *d_dist;
    cudaMalloc(&d_dist, width*height*sizeof(float));
    int *xPos = (int*)malloc(sizeof(int)*width);
    
    int *yPos = (int*)malloc(sizeof(int)*width);
    
    int *d_xPos;
    int *d_yPos;
    int *d_size;

    float* d_out;
	
    cudaMalloc(&d_out,sizeof(float)*1280);    
    cudaMalloc(&d_size, sizeof(int)*2);
    cudaMalloc(&d_xPos, sizeof(int)*width);
    cudaMalloc(&d_yPos, sizeof(int)*height);

    // create and alloc GPU memory for the image matrix
    Mat imageCropp;
    imageCropp.data = (unsigned char*) nppiMalloc_8u_C4(width, height, &imageCropp.step);
    imageCropp.setUp(width, height, 4, sl::zed::UCHAR, GPU);


    // create and alloc GPU memory for the mask matrix
    Mat imageCheckerboard;
    imageCheckerboard.data = (unsigned char*) nppiMalloc_8u_C4(width, height, &imageCheckerboard.step);
    imageCheckerboard.setUp(width, height, 4, sl::zed::UCHAR, GPU);

    if (loadImage) { // if an image is given in argument we load it and use it as background
        string imageName = argv[1];
        cv::Mat imageBackground = cv::imread(imageName);

        if (imageBackground.empty()) { // if the image can't be load we will use a generated image
            loadImage = false;
            cout << " -> ERROR : can't load image : " << imageName << ", generating a synthetic image instead." << endl;
        } else {// adapt the size of the given image to the size of the zed image
            cv::resize(imageBackground, imageBackground, cv::Size(width, height));
            // we work with image in 4 channels for memory alignement purpose
            cv::cvtColor(imageBackground, imageBackground, CV_BGR2BGRA);
            // copy the image from the CPU to the GPU
            cudaMemcpy2D( (uchar*) imageCheckerboard.data, imageCheckerboard.step, (Npp8u*) imageBackground.data, imageBackground.step, imageBackground.step, height, cudaMemcpyHostToDevice);
	    	    
	    cudaMemcpy(d_array, array, width*height*sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    if (!loadImage)// Compute the checkerboard only one time, it will be use to mask the invalid area
        cuCreateCheckerboard(imageCheckerboard);

    // create a CPU image for display purpose
    cv::Mat imDisplay(height, width, CV_8UC4);

    // define a distance threshold, above it the image will be replace by the mask
    float distCut = 2.; // in Meters
    bool threshAsChange = true;

    char key = ' ';

    std::cout << " Press 'p' to increase distance threshold" << std::endl;
    std::cout << " Press 'm' to decrease distance threshold" << std::endl;

    // launch a loop
    bool run = true;

    float depth = 0;

    //Find maximalSquare initial data
    float **matrix = (float**)malloc(width*sizeof(float*));
	float *submatrix = (float*)malloc(sizeof(float)*height);
	for(int i = 0; i<width; i++){
	    matrix[i] = (float*)malloc(sizeof(float)*height);
	}
    int *size ;
	size = (int*)malloc(sizeof(int)*2);
    memset(size, 0, sizeof(int));
    int row = 0;
    int col = 0;
    long long square = 0;	


    float temp = 0;
    memset(xPos, 0, sizeof(int)*width);
    memset(yPos, 0, sizeof(int)*height);
    memset(dist, 65535, sizeof(float)*width*height);
    size[0] = 60;
    size[1] = 20;
	
    cudaMemcpy(d_xPos, xPos, sizeof(int)*width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_yPos, yPos, sizeof(int)*height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_size, size, sizeof(int)*2, cudaMemcpyHostToDevice);
    //===============================
	
  
    while (run) {
	
        // Grab the current images and compute the disparity
        // we want a full depth map for better visual effect
        bool res = zed->grab(FILL);

        // get the left image
        // !! WARNING !! this is not a copy, here we work with the data allocated by the zed object
        // this can be done ONLY if we call ONE time this methode before the next grab, make a copy if you want to get multiple IMAGE
        Mat imageLeftGPU = zed->retrieveImage_gpu(LEFT);

        // get the depth
        // !! WARNING !! this is not a copy, here we work with the data allocated by the zed object
        // this can be done ONLY if we call ONE time this methode before the next grab, make a copy if you want to get multiple MEASURE
        Mat depthGPU = zed->retrieveMeasure_gpu(DEPTH);
	
        // Call the cuda function that mask the image area wich are deeper than the threshold
        cuCroppImageByDepth(d_array, depthGPU, imageLeftGPU, imageCropp, imageCheckerboard, distCut);
            cudaDeviceSynchronize();
        // Copy the processed image frome the GPU to the CPU for display
        cudaMemcpy2D((uchar*) imDisplay.data, imDisplay.step, (Npp8u*) imageCropp.data, imageCropp.step, imageCropp.getWidthByte(), imageCropp.height, cudaMemcpyDeviceToHost);
	
        // Copy the array from gpu to cpu
        cudaMemcpy(array, d_array, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	
        if (printHelp) // write help text on the image if needed
            cv::putText(imDisplay, helpString, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(111, 111, 111, 255), 2);

        // display the result
        
	
        
        key = cv::waitKey(20);

	cudaMemcpy(d_arraySquare, array, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, dist, width*height*sizeof(float), cudaMemcpyHostToDevice);
	clock_t begin = clock();
	cuFindMaximalSquare(d_arraySquare,  d_size, d_dist);
	
	cudaMemcpy(d_out, out, sizeof(float)*1280, cudaMemcpyHostToDevice);
	clock_t end = clock();
	cuFindMinDist(d_out, d_dist);
	
	cudaMemcpy(out, d_out, sizeof(float)*1280, cudaMemcpyDeviceToHost);
	for(int i=0; i<1280; i++)
		if(out[0] > out[i]) out[0] = out[i];
	
//			cout<<"Min dist from GPU: "<<out[0]<<endl;
	//-------------------------------------------------------------------------------------
/*	
	//---------------CPU position testing--------------------------------------
	cudaMemcpy(dist, d_dist, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<1280*720; i++)
	    if(dist[0] > dist[i])
		out[0] = dist[i];
	cout << "Min dist in CPU: "<<out[0] << "\n";

	cudaMemcpy(d_dist, dist, width*height*sizeof(float), cudaMemcpyHostToDevice);
*/		//----------------CPU position testing---------------------------------------
	

	//----------------Find the x and y position----------------------------------
	cudaMemcpy(d_out, out, 1280*sizeof(float), cudaMemcpyHostToDevice);
	cuFindPos(d_dist, d_xPos, d_yPos, d_out);
	cudaMemcpy(xPos, d_xPos, sizeof(int)*width, cudaMemcpyDeviceToHost);
	cudaMemcpy(yPos, d_yPos, sizeof(int)*height, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//---------------------------------------------------------------------------

	//clock_t begin = clock();
	
	//============================= UAV control part ========================================
	if(xPos[0] > 620 && xPos[0] < 660 && yPos[0] > 300 && yPos[0] < 340)
		putText(imDisplay, "Go straight", cv::Point(400,160), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
	else if(xPos[0] > 660 && xPos[0] < 1280 && yPos[0] > 0 && yPos[0] < 300)
		putText(imDisplay, "Move right then move up", cv::Point(400,160), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
	else if(xPos[0] > 660 && xPos[0] < 1280 && yPos[0] > 340 && yPos[0] < 660)
		putText(imDisplay, "Move right then move down", cv::Point(400,160), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
	else if(xPos[0] > 0 && xPos[0] < 620 && yPos[0] > 0 && yPos[0] < 300)
		putText(imDisplay, "Move left then move up", cv::Point(400,160), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
	else if(xPos[0] > 0 && xPos[0] < 620 && yPos[0] > 340 && yPos[0] < 640)
		putText(imDisplay, "Move left then move down", cv::Point(400,160), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);		
	//else	
	//	putText(imDisplay, "Stop no route", cv::Point(640,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);		
	//====================================================================================

	if(out[0] < 65535 || out[0]  >0){
		rectangle(imDisplay,cv::Point(xPos[0]-size[0]/2, yPos[0]-size[0]/2), cv::Point(xPos[0]+size[0]/2, yPos[0]+size[0]/2),cv::Scalar(255, 0, 0),3,8,0);
		rectangle(imDisplay,cv::Point(xPos[0]-size[1]/2, yPos[0]-size[1]/2), cv::Point(xPos[0]+size[1]/2, yPos[0]+size[1]/2),cv::Scalar(255, 0, 0),3,8,0);
	//	line(imDisplay, cv::Point(640, 320), cv::Point(xPos[0], yPos[0]), cv::Scalar(110,220,0), 2, 8);
	}
	else{
		rectangle(imDisplay,cv::Point(640-size[0]/2, 360-size[0]/2), cv::Point(640+size[0]/2, 360+size[0]/2),cv::Scalar(255, 0, 0),3,8,0);
		rectangle(imDisplay,cv::Point(640-size[1]/2, 360-size[1]/2), cv::Point(640+size[1]/2, 360+size[1]/2),cv::Scalar(255, 0, 0),3,8,0);
	}

        switch (key) // handle the pressed key
        {
            case 'q': // close the program
            case 'Q':
                run = false;
                break;

            case 'p': // increase the distance threshold
            case 'P':
                distCut += 0.25;
                threshAsChange = true;
                break;

            case 'm': // decrease the distance threshold
            case 'M':
                distCut = (distCut > 1 ? distCut - 0.25 : 1);
                threshAsChange = true;
                break;

        
	    default:
                break;
        }

	cv::imshow("Image cropped by distance", imDisplay);
        if (threshAsChange) {
            cout << "New distance threshold " << distCut << "m" << endl;
            threshAsChange = false;
        }
		
		
		//gpuend = clock();

		double elapsedTimegpu = double(end-begin)/CLOCKS_PER_SEC;
		//double elapsedTimecv = double(cvend - cvbegin)/CLOCKS_PER_SEC;
		//cout<<"GPU running time: "<<elapsedTimegpu<<endl;
		//cout<<"CV running time: "<<elapsedTimecv<<endl;

    }

    // free all the allocated memory before quit
   
    imDisplay.release();
    imageCropp.deallocate();
    imageCheckerboard.deallocate();
    delete zed;

    return 0;
}




















































