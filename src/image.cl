
const sampler_t s_nearest = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;	
const sampler_t s_linear = CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;	
const sampler_t s_repeat = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT;	

__kernel void image_test(__read_only image2d_t im, float4* out){
	/* nearest */
	
	out[0] = read_imagef(im, s_nearest, (float2)(0.5f, 0.5f));
	out[1] = read_imagef(im, s_nearest, (float2)(0.8f, 0.5f));
	out[2] = read_imagef(im, s_nearest, (float2)(1.3f, 0.5f));
	
	
	/* linear */
	
	out[3] = read_imagef(im, s_linear, (float2)(0.5f, 0.5f));
	out[4] = read_imagef(im, s_linear, (float2)(0.8f, 0.5f));
	out[5] = read_imagef(im, s_linear, (float2)(1.3f, 0.5f));
	
	
	/* repeat */
	
	out[6] = read_imagef(im, s_repeat, (float2)(4.5f, 0.5f));
	out[7] = read_imagef(im, s_repeat, (float2)(5.0f, 0.5f));
	out[8] = read_imagef(im, s_repeat, (float2)(6.5f, 0.5f));
	
}
