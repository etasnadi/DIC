#ifdef A_0_90
#define dx1 fdx
#define dy1 fdy
#define dx2 bd
#define dy2 bd
#endif

#ifdef A_90_180
#define dx1 fdx
#define dy1 bdy
#define dx2 bd
#define dy2 fd
#endif

#ifdef A_180_270
#define dx1 bdx
#define dy1 bdy
#define dx2 fd
#define dy2 fd
#endif

#ifdef A_270_360
#define dx1 bdx
#define dy1 fdy
#define dx2 fd
#define dy2 bd
#endif

typedef struct {
	int x;
	int y;
} point;

typedef point dimensions;

typedef struct {
	int cols;
	int rows;
	int stacks;
} image;

typedef struct {
	size_t mask_size;
	dimensions dim_selector;
} mask_spec1D;

typedef struct {
	int x1;
	int x2;
	int y1;
	int y2;
} clip;

/*
__private int i(int lin_coord, int se_id, image se_dims){
	return (se_dims.rows*se_dims.cols) * se_id + lin_coord;
}*/

// input: linear coordinate in the plane and a stack id
// output: the merged stack coordinate
// (i1.1, i1.2, i1.3, ...),(i2.1, i2.2, i2.3, ...),(i3.1, i3.2, i3.3, ...)
// (i1.1, i2.1, i3.1, ...),(i1.2, i2.2, i3.2, ...),(i1.3, i2.3 i3.3, ...)
__private int zip(int lin_coord, int se_id, image stack_dims){
	//return lin_coord+(stack_dims.cols*stack_dims.rows);
	return stack_dims.stacks*lin_coord+se_id;
}

__private int unzip(int lin_coord, int se_id, image stack_dims){
	//return lin_coord+(stack_dims.cols*stack_dims.rows);
	return (stack_dims.rows*stack_dims.cols) * se_id + lin_coord;
}

// row major coordinate computation: (x,y)->n
__private int get_linear(int x, int y, int rows, int cols){
	return cols*y+x;
}

__private int get_linear_pt(point p, image img){
	return img.cols*p.y+p.x;
}

// row major coordinate computation: n->(x,y)
__private point get_cart(int linear_coord, image img){
	point p;
	p.y = linear_coord / img.cols;
	p.x = linear_coord - p.y*img.cols;
	return p;
}

__private void get_cart2(int lc, point * p, image img){
	p->y = lc / img.cols;
	p->x = lc - (p->y)*img.cols;
}

__private int sgn(float n){
	if(n < 0){
		return -1;
	}else if(n > 0){
		return 1;
	}else{
		return 0;
	}
}
 
__private int cutoff(int n, int min, int max){
	if(n < min){
		return min;
	}
	if(n > max){
		return max;
	}
	return n;
}


// ................... Compute first derivatives

__private
float bd(float pm1, float p, float pp1){
	return (-pm1 + p)/2;
}

__private
float fd(float pm1, float p, float pp1){
	return (-p + pp1)/2;
}

__private
float fdx(__global float *data, int x, int y, int se_id, image img){
	int lc = zip(img.cols * y + x, se_id, img);
	x = cutoff(x+1, 0, img.cols-1);
	int lc_xp1_y0 = zip(img.cols * y + x, se_id, img);
	return (-data[lc] + data[lc_xp1_y0])/2.0f;
}


__private
float bdx(__global float *data, int x, int y, int se_id, image img){
	int lc = zip(img.cols * y + x, se_id, img);
	x = cutoff(x-1, 0, img.cols-1);
	int lc_xm1_y0 = zip(img.cols * y + x, se_id, img);
	return (-data[lc_xm1_y0] + data[lc])/2.0f;
}

__private
float fdy(__global float *data, int x, int y, int se_id, image img){
	int lc = zip(img.cols * y + x, se_id, img);
	y = cutoff(y+1, 0, img.rows-1);
	int lc_x0_yp1 = zip(img.cols * y + x, se_id, img);
	return (-data[lc] + data[lc_x0_yp1])/2.0f;
}

__private
float bdy(__global float *data, int x, int y, int se_id, image img){
	int lc = zip(img.cols * y + x, se_id, img);
	y = cutoff(y-1, 0, img.rows-1);
	int lc_x0_ym1 = zip(img.cols * y + x, se_id, img);
	return (-data[lc_x0_ym1] + data[lc])/2.0f;
}

__private
float norm_(float a, float b){
	float eps = 0.000000000000000000000001;
	return sqrt(a*a + b*b + eps);
}


__kernel
void update(
		__global float * f,
		__global float * g,
		__global float * diff,
		float dirx,
		float diry,
		float wAccept,
		float wSmooth,
		image img
){
	
	int tid = get_global_id(0);
	
	if(tid >= img.cols*img.rows) return;
	
	for(int se_id = 0; se_id < img.stacks; se_id++){
		
		point p;
		get_cart2(tid, &p, img);
		
		int lc = img.cols * p.y + p.x;
		
		int cc_xm1 = cutoff(p.x-1, 0, img.cols-1);
		int lc_xm1 = img.cols * p.y + cc_xm1;
		
		int cc_xp1 = cutoff(p.x+1, 0, img.cols-1);
		int lc_xp1 = img.cols * p.y + cc_xp1;
		
		int cc_ym1 = cutoff(p.y-1, 0, img.rows-1);
		int lc_ym1 = img.cols * cc_ym1 + p.x;
		
		int cc_yp1 = cutoff(p.y+1, 0, img.rows-1);
		int lc_yp1 = img.cols * cc_yp1 + p.x;
		
		// First derivatives
		float fx_ = dx1(f, p.x, p.y, se_id, img);
		float fy_ = dy1(f, p.x, p.y, se_id, img);
	
		// Second pure derivatives
		float fxx_ = dx2(
				dx1(f, cc_xm1, p.y, se_id, img), 
				dx1(f, p.x, p.y, se_id, img), 
				dx1(f, cc_xp1, p.y, se_id, img)
			);
		
		float fyy_ = dy2(
				dy1(f, p.x, cc_ym1, se_id, img), 
				dy1(f, p.x, p.y, se_id, img),
				dy1(f, p.x, cc_yp1, se_id, img)
			);
		
		
		float fxy_ = (
					dy2(
							dx1(f, p.x, cc_ym1, se_id, img), 
							dx1(f, p.x, p.y, se_id, img),
							dx1(f, p.x, cc_yp1, se_id, img)
					)
					+
					dx2(
							dy1(f, cc_xm1, p.y, se_id, img),
							dy1(f, p.x, p.y, se_id, img),
							dy1(f, cc_xp1, p.y, se_id, img)
						)
					)/2.0f;
		
		// Compute the norms, and the divergence	
		float fx_xm1 = dx1(f, cc_xm1, p.y, se_id, img);
		float fy_xm1 = dy1(f, cc_xm1, p.y, se_id, img);
		float nxm1 = norm_(fx_xm1, fy_xm1);
		
		float fx_xp1 = dx1(f, cc_xp1, p.y, se_id, img);
		float fy_xp1 = dy1(f, cc_xp1, p.y, se_id, img);
		float nxp1 = norm_(fx_xp1, fy_xp1);
		
		float fx_ym1 = dx1(f, p.x, cc_ym1, se_id, img);
		float fy_ym1 = dy1(f, p.x, cc_ym1, se_id, img);
		float nym1 = norm_(fx_ym1, fy_ym1);
		
		float fx_yp1 = dx1(f, p.x, cc_yp1, se_id, img);
		float fy_yp1 = dy1(f, p.x, cc_yp1, se_id, img);
		float nyp1 = norm_(fx_yp1, fy_yp1);
		
		float fx_dx1 = (-fx_xm1/nxm1 + fx_xp1/nxp1)/2.0f;
		float fy_dy1 = (-fy_ym1/nym1 + fy_yp1/nyp1)/2.0f;
		float div_ = (fx_dx1 + fy_dy1)/2.0f;
		
		// Add the difference
		float der_ = (dirx*dirx)*fxx_ + (2*dirx*diry)*fxy_ + (diry*diry)*fyy_;
		g[zip(lc, se_id, img)] += wAccept * (der_ - diff[zip(lc, se_id, img)] + wSmooth * div_);
		
		// Cut the negatives
		if(g[zip(lc, se_id, img)] < 0.0f){
			g[zip(lc, se_id, img)] = 0.0f;
		}
	 	
		//g[zip(lc, se_id, img)] = diff[lc];
	}
	
}

__kernel
void compute_diff(
		float dirx,
		float diry,
		__global float * f,
		__global float * diff,
		image img)
{
	int tid = get_global_id(0); 
	if(tid >= img.cols*img.rows) return;
	for(int se_id = 0; se_id < img.stacks; se_id++){
		int lc = zip(tid, se_id, img);
		
		point p;
		get_cart2(tid, &p, img);
		float gx_tid = dx1(f, p.x, p.y, se_id, img);
		float gy_tid = dy1(f, p.x, p.y, se_id, img);
		diff[lc] = sgn(dirx)*dirx*dirx*gx_tid + sgn(diry)*diry*diry*gy_tid;
	}

}

__kernel void zero_array(__global float * A, __global float * B, image img){
	int tid = get_global_id(0);
	if(tid >= img.cols*img.rows) return;
	for(int se_id = 0; se_id < img.stacks; se_id++){
		int lc = zip(tid, se_id, img);
		A[lc] = 0.0f;
		B[lc] = 0.0f;
	}
}

// The image img is the padded image size!
__kernel void init_working_image(
		__global uchar * in_img,	// To store the input image
		__global float * A,			// To store the original image
		point frame,
		clip img_clip,
		image img_padded)
{	
	int tid = get_global_id(0); 
	if(tid >= img_padded.cols*img_padded.rows) return;
	
	for(int se_id = 0; se_id < img_padded.stacks; se_id ++){	
		point pt_padded;
		get_cart2(tid, &pt_padded, img_padded);
		int lc_padded = zip(img_padded.cols * pt_padded.y + pt_padded.x, se_id, img_padded);
		
		image img_orig;
		img_orig.cols = img_padded.cols-2;
		img_orig.rows = img_padded.rows-2;
		
		point pt_orig;
		pt_orig.x = pt_padded.x-1;
		pt_orig.y = pt_padded.y-1;
		int orig_lc = unzip(img_orig.cols * pt_orig.y + pt_orig.x, se_id, img_orig);
		
		int n = img_padded.cols*img_padded.rows;
		if(tid < n){
			bool is_frame = 
					pt_padded.x == 0 || 
					pt_padded.y == img_padded.rows-1 || 
					pt_padded.y == 0 || 
					pt_padded.x == img_padded.cols-1;
			
			if(is_frame){
				if(pt_padded.x == frame.x || pt_padded.y == frame.y){
					A[lc_padded] = 1.0f;
				}else{
					A[lc_padded] = 0.0f;
				}
			}else{
				A[lc_padded] = ((float)in_img[orig_lc])/255.0f; // Scale to [0,1]
			}
		}
	}
}

__kernel void get_result(
		__global float * work,
		__global float * out,
		image img)
{
	int tid = get_global_id(0);
	if(tid >= img.cols*img.rows) return;
	
	for(int se_id = 0; se_id < img.stacks; se_id++){
		int lc_work = zip(tid, se_id, img); //(tid*stack_size)+se_id;
		int lc_out = unzip(tid, se_id, img); //(img.cols*img.rows*se_id)+tid;
		
		out[lc_out] = work[lc_work]; // Return unnormalised, normalise on CPU to [0,1] and scale back to [0,255] by multiplying 255.	
	}
	
}