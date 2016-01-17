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
enum ImageMode { original = 0, illumination_adjust = 1, balanced = 2, illumination_then_balance = 3, balance_then_illumination = 4 };
static ImageMode mode = original;
static int modeCount = 0;
/// Constructor
SkinModel::SkinModel() : pimpl(new SkinModelPimpl())
{


}

/// Destructor
SkinModel::~SkinModel()
{
}

namespace cv
{
	namespace xphoto
	{

		template <typename T>
		void balanceWhite(std::vector < Mat_<T> > &src, Mat &dst,
			const float inputMin, const float inputMax,
			const float outputMin, const float outputMax, const int algorithmType)
		{

			/********************* Simple white balance *********************/
			float s1 = 2.0f; // low quantile
			float s2 = 2.0f; // high quantile

			int depth = 2; // depth of histogram tree
			if (src[0].depth() != CV_8U)
				++depth;
			int bins = 16; // number of bins at each histogram level

			int nElements = int(pow((float)bins, (float)depth));
			// number of elements in histogram tree

			for (size_t i = 0; i < src.size(); ++i)
			{
				std::vector <int> hist(nElements, 0);

				typename Mat_<T>::iterator beginIt = src[i].begin();
				typename Mat_<T>::iterator endIt = src[i].end();

				for (typename Mat_<T>::iterator it = beginIt; it != endIt; ++it)
					// histogram filling
				{
					int pos = 0;
					float minValue = inputMin - 0.5f;
					float maxValue = inputMax + 0.5f;
					T val = *it;

					float interval = float(maxValue - minValue) / bins;

					for (int j = 0; j < depth; ++j)
					{
						int currentBin = int((val - minValue + 1e-4f) / interval);
						++hist[pos + currentBin];

						pos = (pos + currentBin)*bins;

						minValue = minValue + currentBin*interval;
						maxValue = minValue + interval;

						interval /= bins;
					}
				}

				int total = int(src[i].total());

				int p1 = 0, p2 = bins - 1;
				int n1 = 0, n2 = total;

				float minValue = inputMin - 0.5f;
				float maxValue = inputMax + 0.5f;

				float interval = (maxValue - minValue) / float(bins);

				for (int j = 0; j < depth; ++j)
					// searching for s1 and s2
				{
					while (n1 + hist[p1] < s1 * total / 100.0f)
					{
						n1 += hist[p1++];
						minValue += interval;
					}
					p1 *= bins;

					while (n2 - hist[p2] > (100.0f - s2) * total / 100.0f)
					{
						n2 -= hist[p2--];
						maxValue -= interval;
					}
					p2 = p2*bins - 1;

					interval /= bins;
				}

				src[i] = (outputMax - outputMin) * (src[i] - minValue)
					/ (maxValue - minValue) + outputMin;
			}
			/****************************************************************/



			dst.create(/**/ src[0].size(), CV_MAKETYPE(src[0].depth(), int(src.size())) /**/);
			cv::merge(src, dst);
		}

		/*!
		* Wrappers over different white balance algorithm
		*
		* \param src : source image (RGB)
		* \param dst : destination image
		*
		* \param inputMin : minimum input value
		* \param inputMax : maximum input value
		* \param outputMin : minimum output value
		* \param outputMax : maximum output value
		*
		* \param algorithmType : type of the algorithm to use
		*/
		void balanceWhite(const Mat &src, Mat &dst, const int algorithmType,
			const float inputMin, const float inputMax,
			const float outputMin, const float outputMax)
		{
			switch (src.depth())
			{
			case CV_8U:
			{
				std::vector < Mat_<uchar> > mv;
				split(src, mv);
				balanceWhite(mv, dst, inputMin, inputMax, outputMin, outputMax, algorithmType);
				break;
			}
			case CV_16S:
			{
				std::vector < Mat_<short> > mv;
				split(src, mv);
				balanceWhite(mv, dst, inputMin, inputMax, outputMin, outputMax, algorithmType);
				break;
			}
			case CV_32S:
			{
				std::vector < Mat_<int> > mv;
				split(src, mv);
				balanceWhite(mv, dst, inputMin, inputMax, outputMin, outputMax, algorithmType);
				break;
			}
			case CV_32F:
			{
				std::vector < Mat_<float> > mv;
				split(src, mv);
				balanceWhite(mv, dst, inputMin, inputMax, outputMin, outputMax, algorithmType);
				break;
			}
			default:
				CV_Error_(CV_StsNotImplemented,
					("Unsupported source image format (=%d)", src.type()));
				break;
			}
		}
	}
}



/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
	//--- IMPLEMENT THIS ---//
	cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);
	//pimpl->alteration++;
	pimpl->skin = new cv::EM(5);
	pimpl->notSkin = new cv::EM(5);
}
/// Adjust the intensity dynamically.This is supposed to help with bad lightning conditions.
cv::Mat3b adjust_intensity(cv::Mat3b img_rgb, int c = 10)
{
	cv::Mat3b img_hsv = cv::Mat3b::zeros(img_rgb.rows, img_rgb.cols);
	cv::cvtColor(img_rgb, img_hsv, CV_BGR2HSV);
	// divide by zero if v == 0
	//with np.errstate(cv::divide = 'ignore', invalid = 'ignore') :
	for (size_t col = 0; col < img_hsv.cols; col++)
	{
		for (size_t row = 0; row < img_hsv.rows; row++)
		{
			//std::cout << (int)img_hsv.at<cv::Vec3b>(col, row).val[2];

			cv::Vec3b pixel = img_hsv.at<cv::Vec3b>(row, col);

			//float currVal = pixel.val[2];
			float denominator = c * log(pixel.val[2]);
			float intensity = 0;
			if (pixel.val[2] != 0)
			{
				intensity = 2 / (1 + exp((-2 * pixel.val[2]) / (denominator))) - 1;
				intensity *= 255;
				pixel.val[2] = round(intensity);
				img_hsv.at<cv::Vec3b>(row, col) = pixel;
			}
		}
	}
	cv::cvtColor(img_hsv, img_hsv, CV_HSV2BGR);
	return img_hsv;
}


/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{

	// Show different effects
	//namedWindow("Source", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Source", img);                   
	//namedWindow("Valued", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Valued", adj_img);                   
	//cv::Mat3b balancedImg = cv::Mat3b::zeros(img.rows, img.cols);
	//cv::xphoto::balanceWhite(img, balancedImg, 0, 0, 255, 0, 255);
	//namedWindow("Balanced", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Balanced", balancedImg);                 
	//cv::Mat3b BalthenVal = adjust_intensity(balancedImg);
	//namedWindow("BalthenVal", cv::WINDOW_AUTOSIZE);
	//cv::imshow("BalthenVal", BalthenVal);                   
	//cv::xphoto::balanceWhite(adj_img, balancedImg, 0, 0, 255, 0, 255);
	//namedWindow("ValthenBal", cv::WINDOW_AUTOSIZE);
	//cv::imshow("ValthenBal", balancedImg);                   
	//cv::waitKey(0);

	cv::Mat3b newImg = cv::Mat3b::zeros(img.rows, img.cols);
	switch (mode)
	{
	case original:
		newImg = img;
		break;
	case illumination_adjust:
		newImg = adjust_intensity(img);
		break;
	case balanced:
		cv::xphoto::balanceWhite(img, newImg, 0, 0, 255, 0, 255);
		break;
	case illumination_then_balance:
		newImg = adjust_intensity(img);
		cv::xphoto::balanceWhite(img, newImg, 0, 0, 255, 0, 255);
		break;
	case balance_then_illumination:
		cv::xphoto::balanceWhite(img, newImg, 0, 0, 255, 0, 255);
		newImg = adjust_intensity(img);
		break;
	default:
		break;
	}



	//--- IMPLEMENT THIS ---//
	cv::Mat skinPixel = cv::Mat::zeros(1, 3, CV_32FC1);
	cv::Mat notSkinPixel = cv::Mat::zeros(1, 3, CV_32FC1);
	//if(false)
	try
	{
		for (int row = 0; row < newImg.rows; ++row) {
			for (int col = 0; col < newImg.cols; ++col) {

				cv::Mat pixel = cv::Mat::ones(1, 3, CV_32FC1);
				pixel.at<float>(0, 0) = newImg(row, col)[0];
				pixel.at<float>(0, 1) = newImg(row, col)[1];
				pixel.at<float>(0, 2) = newImg(row, col)[2];


				if (mask[row][col] != 0)
				{
					skinPixel.push_back(pixel);
				}
				else
				{
					notSkinPixel.push_back(pixel);
				}
			}
		}
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
	// Train skin em

	// Select next mode
	if (mode != balance_then_illumination)
	{

		modeCount++;
		mode = static_cast<ImageMode>(modeCount / 5);
		cout << mode << "count" << modeCount << endl;
	}
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{

	if (modeCount % 5 == 1)
	{
		cv::FileStorage skinfs("skin" + std::to_string(mode) + ".yml", cv::FileStorage::WRITE);
		cv::FileStorage notskinfs("nonSkin" + std::to_string(mode) + ".yml", cv::FileStorage::WRITE);

		pimpl->skin->write(skinfs);
		pimpl->notSkin->write(notskinfs);
	}

	cv::Mat3b newImg = cv::Mat3b::zeros(img.rows, img.cols);
	switch (mode)
	{
	case original:
		break;
	case illumination_adjust:
		newImg = adjust_intensity(img);
		break;
	case balanced:
		cv::xphoto::balanceWhite(img, newImg, 0, 0, 255, 0, 255);
		break;
	case illumination_then_balance:
		newImg = adjust_intensity(img);
		cv::xphoto::balanceWhite(img, newImg, 0, 0, 255, 0, 255);
		break;
	case balance_then_illumination:
		cv::xphoto::balanceWhite(img, newImg, 0, 0, 255, 0, 255);
		newImg = adjust_intensity(img);
		break;
	default:
		break;
	}

	cv::Mat1b skin = cv::Mat1b::zeros(newImg.rows, newImg.cols);

	for (int row = 0; row < newImg.rows; ++row) {
		for (int col = 0; col < newImg.cols; ++col) {

			cv::Mat pixel = cv::Mat::ones(1, 3, CV_32FC1);
			pixel.at<float>(0, 0) = newImg(row, col)[0];
			pixel.at<float>(0, 1) = newImg(row, col)[1];
			pixel.at<float>(0, 2) = newImg(row, col)[2];

			//Zero element is a likelihood logarithm value for the sample. First element is an index of the most probable mixture component for the given sample.
			cv::Vec2d skinProb = pimpl->skin->predict(pixel);
			cv::Vec2d notSkinProb = pimpl->notSkin->predict(pixel);

			if (skinProb[0] > notSkinProb[0])
				//	skin(row, col) = newImg(row, col)[2];
				skin(row, col) = 255;
		}
	}
	return skin;
}

