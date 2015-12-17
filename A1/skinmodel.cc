#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv/ml.h"

using namespace std;

class SkinModel::SkinModelPimpl {
public:
	cv::EM *skin, *notSkin;

};

/// Constructor
SkinModel::SkinModel() : pimpl(new SkinModelPimpl())
{


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

	pimpl->skin = new cv::EM();
	pimpl->notSkin = new cv::EM();
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
	//--- IMPLEMENT THIS ---//
	try
	{
		cv::Mat skinPixel = cv::Mat::zeros(1, 3, CV_32FC1);
		cv::Mat notSkinPixel = cv::Mat::zeros(1, 3, CV_32FC1);

		for (int row = 0; row < img.rows; ++row) {
			for (int col = 0; col < img.cols; ++col) {

				cv::Mat pixel = cv::Mat::ones(1, 3, CV_32FC1);
				pixel.at<float>(0, 0) = img(row, col)[0];
				pixel.at<float>(0, 1) = img(row, col)[1];
				pixel.at<float>(0, 2) = img(row, col)[2];




				if (mask[row][col] != 0)
				{

					//cv::Vec3f pixel = cv::Vec3f(img(row, col).val[0], img(row, col).val[1], img(row, col).val[2]);
					skinPixel.push_back(pixel);

				}
				else
				{
					notSkinPixel.push_back(pixel);
					/*notSkinPixel.at<float>(row*col, 0) = img(row, col)[0];
					notSkinPixel.at<float>(row*col, 1) = img(row, col)[1];
					notSkinPixel.at<float>(row*col, 2) = img(row, col)[2];*/
					/*notSkinPixel.at<cv::Vec3f>(row*col, 0) =
					cv::Vec3f(img(row, col).val[0], img(row, col)[1], img(row, col).val[2]);*/
					//mask.at<float>(0,0)
				}
			}
		}
		//cv::Mat logLikelyhoods = cv::Mat::zeros(img.rows*img.cols, 1, CV_32FC3)


		//for (size_t i = 0; i < skinPixel.rows; i++)
		//{
		//	std::cout << skinPixel.at<float>(i, 0) << " " << skinPixel.at<float>(i, 1) << " " << skinPixel.at<float>(i, 2);
		//	std::cout << "\n";
		//}
		//for (size_t i = 0; i < notSkinPixel.rows; i++)
		//{
		//	std::cout << notSkinPixel.at<float>(i, 0) << " " << notSkinPixel.at<float>(i, 1) << " " << notSkinPixel.at<float>(i, 2);
		//	std::cout << "\n";

		//}
		
		// Train skin em
		pimpl->skin->train(skinPixel);
	 
		// Train not-skin em
		pimpl->notSkin->train(notSkinPixel);
	}
	catch (...)
	{
		throw;
	}
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

	for (int row = 0; row < img.rows; ++row) {
		for (int col = 0; col < img.cols; ++col) {


			cv::Mat pixel = cv::Mat::ones(1, 3, CV_32FC1);
			pixel.at<float>(0, 0) = img(row, col)[0];
			pixel.at<float>(0, 1) = img(row, col)[1];
			pixel.at<float>(0, 2) = img(row, col)[2];


			//Zero element is a likelihood logarithm value for the sample. First element is an index of the most probable mixture component for the given sample.
		//	cv::Vec2d skinProb = pimpl->skin->predict(img(row, col));
		//	cv::Vec2d notSkinProb = pimpl->notSkin->predict(img(row, col));
			cv::Vec2d skinProb = pimpl->skin->predict(pixel);
			cv::Vec2d notSkinProb = pimpl->notSkin->predict(pixel);

			if (skinProb[0] > notSkinProb[0])
				skin(row, col) = img(row, col)[2];

			

			/*if (true)
			skin(row, col) = rand() % 256;;*/

			if (false) {

				cv::Vec3b bgr = img(row, col);
				if (bgr[2] > bgr[1] && bgr[1] > bgr[0])
					skin(row, col) = 2 * (bgr[2] - bgr[0]);
			}
		}
	}

	if (true)
	{
		namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
		cv::imshow("Display window", skin);                   // Show our image inside it.
		string name = rand() % 1000 + ".jpg";
		cv::imwrite(name, skin);
		//cv::waitKey(0);                                          // Wait for a keystroke in the window
	}

	return skin;
}

