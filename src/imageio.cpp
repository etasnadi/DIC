#include <iostream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "imageio.h"
#include "common.h"

using namespace std;
using namespace cv;

clip getClip(int _x1, int _x2, int _y1, int _y2) {
	clip c;
	c.x1 = _x1;
	c.x2 = _x2;
	c.y1 = _y1;
	c.y2 = _y2;
	return c;
}

// Auxiliary functions
point get_cart(int linear_coord, dimensions dims){
	point p;
	p.y = linear_coord / dims.x;
	p.x = linear_coord - p.y*dims.x;
	return p;
}

int get_linear(point p, dimensions dims){
	return dims.x*p.y+p.x;
}

/**
 * Normalises and crops the image acquired from the device in out_img, and crops it according to the direction.
 *
 */
void normalise_and_scale(float* out_img, Mat img, dimensions padded_stack_dims, clip& img_clip){
	int padded_image_size = padded_stack_dims.x*padded_stack_dims.y;
	float min = findMin(out_img, padded_image_size );
	float max = findMax(out_img, padded_image_size );

	dimensions cropped_dims;
	cropped_dims.x = padded_stack_dims.x-2;
	cropped_dims.y = padded_stack_dims.y-2;

	// p.x, p.y: the coordinates in the padded image.
	point p;
	for(p.x = 0; p.x < padded_stack_dims.x; p.x++){
		for(p.y = 0; p.y < padded_stack_dims.y; p.y++){

			if( (p.x >= img_clip.x1 && p.x <= img_clip.x2) && (p.y >= img_clip.y1 && p.y <= img_clip.y2) ){

				// c.x, c.y: the coordinates in the cropped image
				point c;
				c.x = p.x-img_clip.x1;
				c.y = p.y-img_clip.y1;

				int lc_cropped = get_linear(c, cropped_dims);
				int lc_padded = get_linear(p, padded_stack_dims);

				// Normalise
				out_img[lc_padded] -= min;
				out_img[lc_padded] /= max;
				// Scale into [0,255]
				out_img[lc_padded] *= 255;
				img.data[lc_cropped] = (uchar) out_img[lc_padded];
			}
		}
	}
}

vector<Mat> initOutputStack(dimensions dims, int nElements){
	vector<Mat> result(nElements);
	for(int i = 0; i < nElements; i++){
		 result[i] = Mat(dims.y, dims.x, CV_8UC1);
	}
	return result;
}

Mat loadImage(string path) {
	//IF_VERBOSE
	//cout << "Loading image: " << path << "... ";

	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data) {
		throw string("Can't open the image specified!");
	}

	//IF_VERBOSE
	//cout << "Done. [" << img.cols << "x" << img.rows << "]" << endl;
	return img;
}

void saveImage(string path, Mat img){
	//IF_VERBOSE
	//cout << "Saving image: " << path << endl;

	bool success = imwrite(path, img);

	if (!success) {
		throw string("Can't save the result into the file specified!");
	}
}

vector<Mat> loadStack(string inputPattern, int nElements){
	//IF_VERBOSE
	//cout << "Loading stack: " << inputPattern << endl;
	vector<Mat> stack(nElements);
	for(int i = 0; i < nElements; i++){
		stack[i] = loadImage(inputPattern);
	}
	return stack;
}

void saveStack(string outputPattern, vector<Mat>& stack){
	//IF_VERBOSE
	//cout << "Saving stack: " << outputPattern << endl;
	for(int i = 0; i < stack.size(); i++){
		saveImage(outputPattern, stack[i]);
	}
}
