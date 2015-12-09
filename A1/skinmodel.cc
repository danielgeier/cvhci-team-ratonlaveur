#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <fstream>

#include <opencv2/ml/ml.hpp>
#include <memory>
//#include <opencv/ml.h>


using namespace std;

class SkinModel::SkinModelPimpl {
public:
	CvNormalBayesClassifier *classifier;
};

/// Constructor
SkinModel::SkinModel() : pimpl(new SkinModelPimpl())
{
	//pimpl = new SkinModelPimpl();
	pimpl->classifier= new CvNormalBayesClassifier;
}

/// Destructor
SkinModel::~SkinModel() 
{
}

/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
    //--- IMPLEMENT THIS ---//
    pimpl->classifier->clear();
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
	//--- IMPLEMENT THIS ---//

    cv::Mat image_vectors = cv::Mat::zeros(img.rows*img.cols  , 3, CV_32FC1);
    cv::Mat responses = cv::Mat::zeros(img.rows*img.cols  , 1, CV_32FC1);

    int k=0;
    cv::Vec3b bgr =0;

    for (int i=0;i<img.rows;i++)
    {
		for (int j=0;j<img.cols;j++)
		{
			if(k < img.rows * img.cols){

				if(mask[i][j]==0){
					responses.at<float>(k,0)=0;
				}
				else responses.at<float>(k,0)=1;

				bgr = img(i,j);
				for(int color=0;color<3;color++)
				{				
					image_vectors.at<float>(k,color)=(float)bgr[color];
				}

				k++;
			}
		}
	}

	pimpl->classifier->train(image_vectors, responses, cv::Mat(), cv::Mat(), false);  //, varIdx, sampleIdx, false);
}


/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
	//--- IMPLEMENT THIS ---//
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);

    //cout<<"flag 0.5 "<<endl;

	//--- IMPLEMENT THIS ---//
    if (false){
	    for (int row = 0; row < img.rows; ++row) {
	        for (int col = 0; col < img.cols; ++col) {

				if (false)
					skin(row, col) = rand()%256;

				if (false)
					skin(row, col) = img(row,col)[2];

				if (false) {
				
					cv::Vec3b bgr = img(row, col);
					if (bgr[2] > bgr[1] && bgr[1] > bgr[0]) 
						skin(row, col) = 2*(bgr[2] - bgr[0]);
				}
			}
		}
	}

	if (true){

		int k=0;
    	cv::Vec3b bgr =0;

    	cv::Mat samples = cv::Mat::zeros(img.rows*img.cols  , 3, CV_32FC1);
   		cv::Mat results = cv::Mat::zeros(img.rows*img.cols  , 1, CV_32FC1);

		for (int i = 0; i < img.rows; i++) {
	    	for (int j = 0; j < img.cols; j++) {

				bgr = img(i,j);

				for(int color=0;color<3;color++)
				{				
					samples.at<float>(k,color)=(float)bgr[color];
				}

				k++;
			}
		}

		pimpl->classifier->predict(samples, &results);

		k=0;

		for (int i=0;i<img.rows;i++)
	    {
			for (int j=0;j<img.cols;j++)
			{
				skin[i][j]=results.at<float>(k,0)*255;
				k++;
			}
		}
	}

	// cv::_OutputArray op_skin = cv::_OutputArray(cv::Mat1b(img.rows,img.cols));
	// cv::_InputArray kernel = cv::_InputArray(cv::Mat1b(5,5));

	// morphologyEx(skin, op_skin, CV_MOP_OPEN, kernel);

    return skin;
}

