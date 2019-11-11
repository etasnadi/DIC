#ifndef IMAGEIO_H
#define IMAGEIO_H

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "common.h"

using namespace std;
using namespace cv;

typedef struct {
	int x1;
	int x2;
	int y1;
	int y2;
} clip;

typedef struct {
	int x;
	int y;
} point;

typedef struct {
	int x;
	int y;
	int stacks;
} dimensions;

// Auxiliary functions
point get_cart(int linear_coord, dimensions dims);
int get_linear(point p, dimensions dims);

void normalise_and_scale(float* out_img, Mat img, dimensions padded_stack_dims, clip& img_clip);
vector<Mat> initOutputStack(dimensions dims, int nElements);
Mat loadImage(string path);
void saveImage(string path, Mat img);
vector<Mat> loadStack(string inputPattern, int nElements);
void saveStack(string outputPattern, vector<Mat>& stack);

clip getClip(int _x1, int _x2, int _y1, int _y2);

#endif
