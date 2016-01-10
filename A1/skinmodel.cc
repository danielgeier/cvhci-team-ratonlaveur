#include "skinmodel.h"
#include <cmath>
#include <iostream>
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

class SkinModel::SkinModelPimpl
{
public:
	vector<vector<float>> skinHists;
	vector<vector<float>> nonskinHists;
	// Sources of interest
	vector<vector<vector<float>>> sois;

	bool dempsterShaferClassify(const unsigned char* pixel);
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

	pimpl->sois.resize(NUM_SOIS);
	for (vector<vector<float>>& masses : pimpl->sois)
	{
		masses.resize(3); // skin, nonskin, theta
		for (vector<float>& m : masses)
			m.resize(256);
	}
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

void normalizeHist(vector<float>& hist)
{
	float max = *max_element(hist.begin(), hist.end());

	for (float& f : hist)
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
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
	// TODO Preprocessing

	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			switch (mask[row][col])
			{
			case MASK_SKIN:
				updateHist(img.at<cv::Vec3b>(row, col).val, pimpl->skinHists);
				break;

			case MASK_NONSKIN:
				updateHist(img.at<cv::Vec3b>(row, col).val, pimpl->nonskinHists);
				break;

			default:
				throw exception("Mask image is ambigious.");
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
		normalizeHist(hist);
	for (vector<float>& hist : pimpl->nonskinHists)
		normalizeHist(hist);

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

			pimpl->sois[idxSoi][THETA][intensity] = mass;
		}
	}
}


const static vector<vector<bool>> COMBINATIONS = {{false, true, true, true, true, false},{true, false, false, true, true, true},{true, true, false, false, true, false},{false, false, true, false, true, false},{false, true, true, false, false, false},{true, false, false, false, false, false},{false, true, false, true, true, false},{true, false, false, true, false, true},{true, false, true, true, true, false},{false, false, false, true, false, false},{false, true, true, true, false, true},{false, true, true, false, true, true},{false, false, true, true, false, false},{true, false, false, true, true, false},{true, false, true, true, false, true},{false, false, true, false, true, true},{true, true, true, false, true, false},{false, true, true, false, false, true},{true, false, true, false, false, false},{false, false, true, true, true, true},{false, false, false, true, true, false},{true, false, true, true, true, true},{false, true, true, true, false, false},{false, false, true, false, false, false},{true, true, true, false, false, true},{false, true, true, false, true, false},{true, false, true, false, true, true},{false, false, true, true, false, true},{true, true, true, true, false, false},{true, false, true, true, false, false},{true, true, true, false, true, true},{true, false, true, false, false, true},{false, false, false, true, true, true},{false, false, true, true, true, false},{false, false, false, false, false, false},{true, true, true, false, false, false},{true, false, true, false, true, false},{false, true, false, false, true, false},{true, true, true, true, false, true},{true, true, false, true, true, false},{false, false, false, false, true, true},{false, true, true, true, true, true},{false, true, false, false, false, true},{true, true, true, true, true, false},{true, true, false, true, false, true},{false, true, false, true, false, false},{false, false, false, false, false, true},{true, true, false, false, false, false},{false, true, false, false, true, true},{true, false, false, false, true, false},{true, true, false, true, true, true},{false, true, false, true, true, true},{false, false, false, false, true, false},{true, true, false, false, true, true},{false, true, false, false, false, false},{true, false, false, false, false, true},{true, true, false, true, false, false},{false, false, true, false, false, true},{false, true, false, true, false, true},{true, false, false, true, false, false},{true, true, false, false, false, true},{false, false, false, true, false, true},{true, false, false, false, true, true}};

bool SkinModel::SkinModelPimpl::dempsterShaferClassify(const unsigned char pixel[3])
{
	float b = pixel[0];
	float g = pixel[1];
	float r = pixel[2];

	int idxs[] = {b, g, r, abs(r - g), abs(r - b), r > g && r > b};

	float totalPSkin = 0.0;
	float totalPNonskin = 0.0;

	for (const vector<bool> combination : COMBINATIONS)
	{
		float pSkin = 1.0;
		float pNonskin = 1.0;

		for (size_t i = 0; i < combination.size(); ++i)
		{
			bool isTheta = combination[i];

			if (isTheta)
			{
				pSkin *= sois[i][THETA][idxs[i]];
				pNonskin *= sois[i][THETA][idxs[i]];
			}
			else
			{
				pSkin *= sois[i][SKIN][idxs[i]];
				pNonskin *= sois[i][NONSKIN][idxs[i]];
			}
		}

		totalPSkin += pSkin;
		totalPNonskin += pNonskin;
	}

	return totalPSkin > totalPNonskin;
}

/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);

	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			auto pixel = img.at<cv::Vec3b>(row, col).val;

			if (pimpl->dempsterShaferClassify(pixel))
				skin.at<unsigned char>(row, col) = 255;
		}
	}

	const bool SHOW_IMAGES = false;
	if (SHOW_IMAGES)
	{
		namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
		imshow("Display window", skin); // Show our image inside it.
		//string name = rand() % 1000 + ".jpg";
		//imwrite(name, skin);
		cv::waitKey(0); // Wait for a keystroke in the window
	}

	return skin;
}
