#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <opencv2/ml/ml.hpp>
#include <memory>
//#include <opencv/ml.h>
#include <time.h>


using namespace std;

enum SOI_IDX
{
	SKIN,
	NONSKIN,
	THETA
};

static const int NUM_SOIS = 6;
static const unsigned char MASK_SKIN = 255;
static const unsigned char MASK_NONSKIN = 0;
static const int IMG_SIZE = 640 * 480;

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

/// Adjust the intensity dynamically.This is supposed to help with bad lightning conditions.
void adjust_intensity(cv::Mat3b& img_rgb, int c = 10)
{
	cv::cvtColor(img_rgb, img_rgb, CV_BGR2HSV);
	// divide by zero if v == 0
	//with np.errstate(cv::divide = 'ignore', invalid = 'ignore') :
	for (size_t col = 0; col < img_rgb.cols; col++)
	{
		for (size_t row = 0; row < img_rgb.rows; row++)
		{
			//std::cout << (int)img_hsv.at<cv::Vec3b>(col, row).val[2];

			cv::Vec3b pixel = img_rgb.at<cv::Vec3b>(row, col);

			//float currVal = pixel.val[2];
			if (pixel.val[2] != 0)
			{
				float denominator = c * log(pixel.val[2]);
				float intensity = 0;
				intensity = 2 / (1 + exp((-2 * pixel.val[2]) / (denominator))) - 1;
				intensity *= 255;
				pixel.val[2] = round(intensity);
				img_rgb.at<cv::Vec3b>(row, col) = pixel;
			}
		}
	}
	cv::cvtColor(img_rgb, img_rgb, CV_HSV2BGR);
}

cv::Mat3b preprocess(const cv::Mat3b& img)
{
	// Scale intensity adjustment constant c to mean intensity of unprocessed image.
	cv::Scalar meanRgb = cv::mean(img);
	float mean = (meanRgb.val[0] + meanRgb.val[1] + meanRgb.val[2]) / 3.;
	float c = /*20*/(mean * mean * mean) / 10000;
	auto img2 = img;
	adjust_intensity(img2, c);

	cv::xphoto::balanceWhite(img2, img2, 0, 0, 255, 0, 255);

	return img2;
}

class SkinModel::SkinModelPimpl {
public:
	CvKNearest *classifier;
	bool update;
};

/// Constructor
SkinModel::SkinModel() : pimpl(new SkinModelPimpl())
{
	//pimpl = new SkinModelPimpl();
	pimpl->classifier= new CvKNearest;
	pimpl->update = false;
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
    pimpl->update = false;
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img2, const cv::Mat1b& mask)
{
	//--- IMPLEMENT THIS ---//

	cv::Mat3b img = preprocess(img2);

	int nb_pts = img.rows*img.cols/1000;

    cv::Mat image_vectors = cv::Mat::zeros(nb_pts  , 3, CV_32FC1); //img.rows*img.cols
    cv::Mat responses = cv::Mat::zeros(nb_pts  , 1, CV_32FC1);

    int k=0;
    int n=0;
    cv::Vec3b bgr =0;

    int nb_s=0;
    int nb_ns=0;

    int i;
    int j;

  //   for (int i=0;i<img.rows;i++)
  //   {
		// for (int j=0;j<img.cols;j++)
		// {
    		srand(time(NULL));
			while(n<nb_pts){
				//cout<<"nb_ns "<<nb_ns<<" nb_s "<<nb_s<<endl;
				//if(n%30==0){
					i = rand()%img.rows;
					j = rand()%img.cols;
					//cout<<"i "<<i<<" j "<<j<<endl;

					if(mask[i][j]==0){
						responses.at<float>(k,0)=0;
						nb_ns++;
					}
					else {
						responses.at<float>(k,0)=1;
						nb_s++;
					}

					bgr = img(i,j);
					for(int color=0;color<3;color++)
					{				
						image_vectors.at<float>(k,color)=(float)bgr[color];
					}

					k++;
				//}
				n++;
			}
	// 	}
	// }

	//static bool b = false;
	//cout<<"update "<<pimpl->update<<endl;
	//pimpl->update = true ;
	pimpl->classifier->train(image_vectors, responses, cv::Mat(), false, 32, pimpl->update );//, pimpl->update);  //, varIdx, sampleIdx, false);
	//b = b ? b : true;

	//pimpl->update = pimpl->update ? pimpl->update : true;
	pimpl->update = true ;
}


/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
	//--- IMPLEMENT THIS ---//
	pimpl->update = false;
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img2)
{
	cv::Mat3b img = preprocess(img2);

    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);

    int t_begin = time(NULL);
    int nb_neihbours = 6;

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

		cout<<"Nombre de points totals "<<pimpl->classifier->get_sample_count()<<endl;
		// time_t now;
		// cout << "heure de début: " << ctime(&now)<<endl;
		

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
				//cout <<"i "<<i<<" j"<<j<<endl;
				//cout <<i * img.cols + j<<" / "<<img.rows*img.cols<<endl;
				//cout<< setprecision (2) << double(i * img.cols + j )/double(img.rows*img.cols)*100 << "%%" <<endl;
				k++;
			}
		}

		pimpl->classifier->find_nearest(samples, nb_neihbours, &results);

		k=0;

		for (int i=0;i<img.rows;i++)
	    {
			for (int j=0;j<img.cols;j++)
			{
				skin[i][j]=results.at<float>(k,0)*255;
				//cout<<results.at<float>(k,0)<<endl;
				k++;
			}
		}

		// srand(time(NULL));
		// int n=0;
		// while(n<(img.rows*img.cols)/100){
		// 	int i = rand()%img.rows;
		// 	int j = rand()%img.cols;
		// 	cout<<"i "<<i<<" j "<<j<<endl;
		// 	skin[i][j]=255;
		// 	n++;
		// }
	}

	// Create a structuring element (SE)
	int morph_size = 3;
	cv::Mat kernel = cv::getStructuringElement( cv::MORPH_ELLIPSE , cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
	   
	cv::Mat op_skin; // result matrix

	// Apply the specified morphology operation
	morphologyEx(skin, op_skin, CV_MOP_OPEN, kernel);

	morph_size = 1;
	kernel = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

	//morphologyEx(skin, op_skin, CV_MOP_CLOSE, kernel);

	int t_end = time(NULL);
	
	cout<<"Duration = "<<t_end - t_begin <<" sec"<<endl;

    return op_skin;
}

