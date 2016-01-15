#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv/highgui.h>

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

class SkinModel::SkinModelPimpl
{
public:
	vector<vector<float>> skinHists;
	vector<vector<float>> nonskinHists;
	vector<float> skinPos;
	vector<float> nonskinPos;
	// Sources of interest
	vector<vector<vector<float>>> sois;
	vector<vector<float>> fixedSois;

	float dempsterShaferClassify(const unsigned char* pixel, int hack);
};

/// Constructor
SkinModel::SkinModel() : pimpl(new SkinModelPimpl())
{
	pimpl->skinHists.resize(NUM_SOIS);
	for (vector<float>& hist : pimpl->skinHists)
		hist.resize(256);

	pimpl->nonskinHists.resize(NUM_SOIS);
	for (vector<float>& hist : pimpl->nonskinHists)
		hist.resize(256);

	pimpl->skinPos.resize(IMG_SIZE);
	pimpl->nonskinPos.resize(IMG_SIZE);

	pimpl->sois.resize(NUM_SOIS);
	for (vector<vector<float>>& masses : pimpl->sois)
	{
		masses.resize(3); // skin, nonskin, theta
		for (vector<float>& m : masses)
			m.resize(256);
	}

//	pimpl->sois[NUM_SOIS - 1].resize(3); // skin, nonskin, theta
//	for (vector<float>& m : pimpl->sois[NUM_SOIS - 1])
//	{
//		m.resize(IMG_SIZE);
//	}

	pimpl->fixedSois.resize(6);
	for (vector<float>& s : pimpl->fixedSois)
		s.resize(3);
}

/// Destructor
SkinModel::~SkinModel()
{
}

void updateHist(const unsigned char pixel[3], vector<vector<float>>& hists)
{
	unsigned char b = pixel[0];
	unsigned char g = pixel[1];
	unsigned char r = pixel[2];

	hists[0][b] += 1;
	hists[1][g] += 1;
	hists[2][r] += 1;
	hists[3][abs(r - g)] += 1;
	hists[4][abs(r - b)] += 1;
	hists[5][r > g && r > b] += 1;
}

void normalize(vector<float>& v)
{
	float max = *max_element(v.begin(), v.end());

	for (float& f : v)
		f /= max;
}


/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
	for (vector<float>& hist : pimpl->skinHists)
		fill(hist.begin(), hist.end(), 0);

	for (vector<float>& hist : pimpl->nonskinHists)
		fill(hist.begin(), hist.end(), 0);

	for (vector<vector<float>>& masses : pimpl->sois)
		for (vector<float> m : masses)
			fill(m.begin(), m.end(), 0);

	//for (vector<float>& m : pimpl->sois[NUM_SOIS - 1])
	//	fill(m.begin(), m.end(), 0);
//
//	fill(pimpl->skinPos.begin(), pimpl->skinPos.end(), 0);
//	fill(pimpl->nonskinPos.begin(), pimpl->nonskinPos.end(), 0);
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
	cv::Mat3b prepImg = preprocess(img);

	for (int row = 0; row < prepImg.rows; ++row)
	{
		for (int col = 0; col < prepImg.cols; ++col)
		{
			switch (mask[row][col])
			{
			case MASK_SKIN:
				updateHist(prepImg.at<cv::Vec3b>(row, col).val, pimpl->skinHists);
				//pimpl->skinPos[row * 640 + col]++;
				break;

			case MASK_NONSKIN:
				updateHist(prepImg.at<cv::Vec3b>(row, col).val, pimpl->nonskinHists);
				//pimpl->skinPos[row * 640 + col] += 0.02;
				//pimpl->nonskinPos[row * 640 + col]++;
				break;

			default:
				throw "Mask image is ambigious.";
			}
		}
	}
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
	for (vector<float>& hist : pimpl->skinHists)
		normalize(hist);
	for (vector<float>& hist : pimpl->nonskinHists)
		normalize(hist);

	//normalize(pimpl->skinPos);
	//normalize(pimpl->nonskinPos);

	// Calculate mass values.
	for (int idxSoi = 0; idxSoi < NUM_SOIS; ++idxSoi)
	{
		for (int intensity = 0; intensity < 256; ++intensity)
		{
			float skin = pimpl->skinHists[idxSoi][intensity];
			float nonskin = pimpl->nonskinHists[idxSoi][intensity];
			float mass = (skin - nonskin) / (skin + nonskin);

			if (mass > 0)
			{
				pimpl->sois[idxSoi][SKIN][intensity] = mass;
			}
			else
			{
				mass = -mass;
				pimpl->sois[idxSoi][NONSKIN][intensity] = mass;
			}

			pimpl->sois[idxSoi][THETA][intensity] = 1 - mass;
		}
	}
	
//	// Calculate mass values for pos.
//	for (int pos = 0; pos < IMG_SIZE; ++pos)
//	{
//		float skin = pimpl->skinPos[pos];
//		float nonskin = pimpl->nonskinPos[pos];
//		float mass = (skin - nonskin) / (skin + nonskin);
//
//		if (mass > 0)
//		{
//			pimpl->sois[NUM_SOIS - 1][SKIN][pos] = mass;
//		}
//		else
//		{
//			mass = -mass;
//			pimpl->sois[NUM_SOIS - 1][NONSKIN][pos] = mass;
//		}
//
//		pimpl->sois[NUM_SOIS - 1][THETA][pos] = 1 - mass;
//	}
}


const static vector<vector<bool>> COMBINATIONS = //{ { false, true, false, true, true, false, true },{ true, true, true, true, false, false, true },{ true, false, false, true, false, false, true },{ true, true, false, true, false, false, true },{ true, false, true, true, false, false, true },{ false, false, true, false, true, true, false },{ false, false, false, false, true, true, false },{ false, true, false, false, true, true, false },{ true, true, false, false, true, false, false },{ true, true, true, true, false, true, true },{ true, false, true, false, true, false, false },{ true, false, false, true, false, true, true },{ true, true, true, false, true, true, false },{ false, false, false, true, true, false, true },{ false, true, true, true, true, false, true },{ true, true, false, false, false, false, false },{ true, false, true, false, false, false, false },{ false, false, true, false, false, true, true },{ false, true, false, false, false, true, true },{ true, true, true, true, true, true, false },{ false, false, false, true, true, true, true },{ true, false, false, true, true, true, false },{ false, true, true, true, true, true, true },{ false, false, true, false, false, false, true },{ false, false, false, false, false, true, false },{ false, true, false, false, false, false, true },{ true, true, true, false, true, false, true },{ false, true, true, false, false, true, false },{ true, false, false, false, true, false, true },{ false, false, true, true, true, false, false },{ false, true, false, true, true, false, false },{ false, false, false, true, false, true, false },{ true, true, false, true, false, false, false },{ false, true, true, true, false, true, false },{ true, false, true, true, false, false, false },{ true, false, false, false, true, true, true },{ false, false, false, false, true, true, true },{ false, false, true, true, false, false, false },{ true, true, true, true, false, true, false },{ false, true, false, true, false, false, false },{ true, false, false, true, false, true, false },{ true, true, false, true, true, false, false },{ true, false, true, true, true, false, false },{ false, false, false, false, true, false, true },{ true, true, true, false, false, true, false },{ false, true, true, false, true, false, true },{ true, true, false, false, false, false, true },{ true, false, false, false, false, true, false },{ true, false, true, false, false, false, true },{ false, false, false, true, true, true, false },{ false, true, true, false, true, true, false },{ true, false, false, true, true, true, true },{ false, true, true, true, true, true, false },{ true, true, false, false, false, true, true },{ true, false, true, false, false, true, true },{ false, false, true, false, false, false, false },{ false, true, false, false, false, false, false },{ true, true, true, true, true, false, true },{ true, false, false, true, true, false, true },{ false, false, false, true, false, true, true },{ false, false, true, false, true, false, false },{ false, true, true, true, false, true, true },{ false, true, false, false, true, false, false },{ true, true, false, false, true, true, false },{ true, false, false, false, true, true, false },{ true, false, true, false, true, true, false },{ false, false, true, true, false, false, true },{ false, true, false, true, false, false, true },{ false, false, false, true, false, false, true },{ false, true, true, true, false, false, true },{ true, true, false, true, true, false, true },{ true, false, true, true, true, false, true },{ false, false, false, false, true, false, false },{ false, false, true, true, false, true, true },{ true, true, true, false, false, true, true },{ false, true, true, false, true, false, false },{ false, true, false, true, false, true, true },{ true, false, false, false, false, true, true },{ true, true, true, false, true, true, true },{ true, false, true, true, true, true, true },{ false, false, false, false, false, false, false },{ true, true, true, false, false, false, true },{ true, true, false, false, false, true, false },{ false, true, true, false, false, false, false },{ true, false, false, false, false, false, true },{ true, false, true, false, false, true, false },{ false, false, true, true, true, true, false },{ true, true, true, true, true, false, false },{ false, true, false, true, true, true, false },{ true, false, false, true, true, false, false },{ true, true, false, true, false, true, false },{ true, false, true, true, false, true, false },{ false, false, true, false, true, false, true },{ false, true, false, false, true, false, true },{ true, true, false, false, true, true, true },{ true, true, true, true, false, false, false },{ true, false, true, false, true, true, true },{ true, false, false, true, false, false, false },{ true, true, false, true, true, true, true },{ false, false, false, true, false, false, false },{ false, false, true, false, true, true, true },{ false, true, true, true, false, false, false },{ false, true, false, false, true, true, true },{ true, true, false, false, true, false, true },{ true, false, true, false, true, false, true },{ false, false, true, true, false, true, false },{ false, true, false, true, false, true, false },{ false, false, false, true, true, false, false },{ true, true, false, true, true, true, false },{ false, true, true, true, true, false, false },{ true, false, true, true, true, true, false },{ false, false, true, false, false, true, false },{ false, false, false, false, false, false, true },{ true, true, true, false, false, false, false },{ false, true, false, false, false, true, false },{ false, true, true, false, false, false, true },{ true, false, false, false, false, false, false },{ false, false, true, true, true, true, true },{ false, true, false, true, true, true, true },{ false, false, false, false, false, true, true },{ true, true, false, true, false, true, true },{ true, true, true, false, true, false, false },{ false, true, true, false, false, true, true },{ true, false, true, true, false, true, true },{ true, false, false, false, true, false, false },{ false, true, true, false, true, true, true },{ false, false, true, true, true, false, true } };
{{false, true, true, true, true, false},{true, false, false, true, true, true},{true, true, false, false, true, false},{false, false, true, false, true, false},{false, true, true, false, false, false},{true, false, false, false, false, false},{false, true, false, true, true, false},{true, false, false, true, false, true},{true, false, true, true, true, false},{false, false, false, true, false, false},{false, true, true, true, false, true},{false, true, true, false, true, true},{false, false, true, true, false, false},{true, false, false, true, true, false},{true, false, true, true, false, true},{false, false, true, false, true, true},{true, true, true, false, true, false},{false, true, true, false, false, true},{true, false, true, false, false, false},{false, false, true, true, true, true},{false, false, false, true, true, false},{true, false, true, true, true, true},{false, true, true, true, false, false},{false, false, true, false, false, false},{true, true, true, false, false, true},{false, true, true, false, true, false},{true, false, true, false, true, true},{false, false, true, true, false, true},{true, true, true, true, false, false},{true, false, true, true, false, false},{true, true, true, false, true, true},{true, false, true, false, false, true},{false, false, false, true, true, true},{false, false, true, true, true, false},{false, false, false, false, false, false},{true, true, true, false, false, false},{true, false, true, false, true, false},{false, true, false, false, true, false},{true, true, true, true, false, true},{true, true, false, true, true, false},{false, false, false, false, true, true},{false, true, true, true, true, true},{false, true, false, false, false, true},{true, true, true, true, true, false},{true, true, false, true, false, true},{false, true, false, true, false, false},{false, false, false, false, false, true},{true, true, false, false, false, false},{false, true, false, false, true, true},{true, false, false, false, true, false},{true, true, false, true, true, true},{false, true, false, true, true, true},{false, false, false, false, true, false},{true, true, false, false, true, true},{false, true, false, false, false, false},{true, false, false, false, false, true},{true, true, false, true, false, false},{false, false, true, false, false, true},{false, true, false, true, false, true},{true, false, false, true, false, false},{true, true, false, false, false, true},{false, false, false, true, false, true},{true, false, false, false, true, true}};


float SkinModel::SkinModelPimpl::dempsterShaferClassify(const unsigned char pixel[3], int hack)
{
	float b = pixel[0];
	float g = pixel[1];
	float r = pixel[2];


	fixedSois[0][SKIN] = r > 140 ? 0.364339536 : 0.;
	fixedSois[0][NONSKIN] = !(r > 140) ? 0.524533342 : 0.;
	fixedSois[1][SKIN] = (75 < g) && (g < 193) ? 0.147315460 : 0.;
	fixedSois[1][NONSKIN] = !((75 < g) && (g < 193)) ? 0.430099211 : 0.;
	fixedSois[2][SKIN] = (43 < b) && (b < 161) ? 0.222264545 : 0.;
	fixedSois[2][NONSKIN] = !((43 < b) && (b < 161)) ? 0.440255872 : 0.;
	fixedSois[3][SKIN] = (28 < abs(r - g)) && (abs(r - g) < 130) ? 0.546243640 : 0.;
	fixedSois[3][NONSKIN] = !((28 < abs(r - g)) && (abs(r - g) < 130)) ? 0.675261991 : 0.;
	fixedSois[4][SKIN] = (45 < abs(r - b)) && (abs(r - b) < 187) ? 0.496808781 : 0.;
	fixedSois[4][NONSKIN] = !((45 < abs(r - b)) && (abs(r - b) < 187)) ? 0.673853774 : 0.;
	fixedSois[5][SKIN] = (r > g) && (r > b) ? 0.387888000 : 0.;
	fixedSois[5][NONSKIN] = !((r > g) && (r > b)) ? 0.968090001 : 0.;



	float totalPSkin = 0.0;
	float totalPNonskin = 0.0;

	for (const vector<bool>& combination : COMBINATIONS)
	{
		float pSkin = 1.0;
		float pNonskin = 1.0;

		for (size_t i = 0; i < combination.size(); ++i)
		{
			bool isTheta = combination[i];

			if (isTheta)
			{
				float theta = 1 - fixedSois[i][SKIN] +
					fixedSois[i][NONSKIN];
				pSkin *= theta;
				pNonskin *= theta;
			}
			else
			{
				pSkin *= fixedSois[i][SKIN];
				pNonskin *= fixedSois[i][NONSKIN];
			}
		}

		totalPSkin += pSkin;
		totalPNonskin += pNonskin;
	}

	return totalPSkin - totalPNonskin;
}

bool IsWithinEllipse(int x, int y, int h = 317, int k = 310, int major = 270, int minor = 187)
{
	float val = (pow((x - h), 2) / pow(major, 2)) + (pow((y - k), 2) / pow(minor, 2));
	if (val <= 1)
		return true;

	return false;
}

/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
	cv::Mat3b prepImg = preprocess(img);
    cv::Mat1b skin = cv::Mat1b::zeros(prepImg.rows, prepImg.cols);
    cv::Mat1f preSkin = cv::Mat1f::zeros(prepImg.rows, prepImg.cols);
	preSkin = -1.;

	for (int row = 0; row < prepImg.rows; ++row)
	{
		for (int col = 0; col < prepImg.cols; ++col)
		{
			if (!IsWithinEllipse(row, col))
				continue;

			auto pixel = prepImg.at<cv::Vec3b>(row, col).val;
			preSkin.at<float>(row, col) = pimpl->dempsterShaferClassify(pixel, row * 640 + col);
		}
	}

	cv::normalize(preSkin, preSkin, 255., 0., cv::NORM_MINMAX);

	auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 2, 2 });
	cv::morphologyEx(preSkin, preSkin, cv::MORPH_OPEN, kernel);

	preSkin.convertTo(skin, CV_8UC1);

	sort(preSkin.begin(), preSkin.end());
	int vgood = preSkin.at<float>(640. * 480. * 0.99);
	int vbad = preSkin.at<float>(640. * 480. * 0.90);

	for (uchar& s : skin)
	{
		if (s < vbad)
			s = 0;
		//else if (s > vgood)
		//	s = 255;
	}
	cout << vbad << ' ' << vgood << endl;

	cv::threshold(skin, skin, 128/*(vgood + vgood) / 2.*/, 255., cv::THRESH_BINARY);

	const bool SHOW_IMAGES = true;
	if (SHOW_IMAGES)
	{
		for (int row = 0; row < prepImg.rows; ++row)
		{
			for (int col = 0; col < prepImg.cols; ++col)
			{

				uchar* pixel = prepImg.at<cv::Vec3b>(row, col).val;

				uchar val = skin.at<unsigned char>(row, col);
				//if (val == 255) {
				pixel[1] = val;
				//}
			}
		}
		namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
		imshow("Display window", prepImg); // Show our image inside it.
		cv::waitKey(0); // Wait for a keystroke in the window
	}

	return skin;
}
