#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <math.h>
#include <vector>
#include <iostream>

#include <CL/cl.hpp>

#include "opencv2/core/core.hpp"

#include "main.h"
#include "common.h"
#include "clio.h"
#include "SimpleConfig.h"

#define RETURN_SUCCESS 0
#define RETURN_ERROR 1

#define STR_CONF "--str-conf"
#define FILE_CONF "--file-conf"
#define DEVINFO "--get-devinfo"
#define DEF_PARAMS "verbose=1,wAccept=0.025f,wSmooth=0.0125f,direction=135.0f,nIter=20000,locSize=64,kernelSrcPath=dic-rec.cl,input='',output='',platformId=0,deviceId=0"

#define SEP ";"

#define CL_HEADER_A_0_90 "#define A_0_90\n\0"
#define CL_HEADER_A_90_180 "#define A_90_180\n\0"
#define CL_HEADER_A_180_270 "#define A_180_270\n\0"
#define CL_HEADER_A_270_360 "#define A_270_360\n\0"

using namespace cv;
using namespace cl;
using namespace std;

int verbose = 0;

// Select platform and device!
int selected_platform = 0;
int selected_device = 0;
// ...

point frame;
clip img_clip;

float wAccept = 0.025f;
float wSmooth = 0.0125f;
::size_t locSize = 64; // Number of cores in a thread group of the device
int nIter = 20000; // Number of iterations
float direction = 0.0f; // The direction

string inputStackPath;
string outputStackPath;
int nStackElements = 1;

string kernel_source_path;
char cl_header[20];

cl_mem create_float_buffer(cl_context context, ::size_t img_size, cl_int *ret){
	cl_mem buff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*img_size, NULL, ret);
	cl_call(*ret, __LINE__);
	return buff;
}

void update_kernel(
		cl_call_env env,
		cl_mem *fa,
		cl_mem *fb,
		cl_mem *diff,
		float dirx,
		float diry,
		float wAccept,
		float wSmooth,
		dimensions img_dims)
{

	cl_call(clSetKernelArg(env.kernel, 0, sizeof(cl_mem), (void*)fa), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 1, sizeof(cl_mem), (void*)fa), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 2, sizeof(cl_mem), (void*)diff), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 3, sizeof(float), &dirx), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 4, sizeof(float), &diry), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 5, sizeof(float), &wAccept), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 6, sizeof(float), &wSmooth), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 7, sizeof(dimensions), &img_dims), __LINE__);
	cl_call(clEnqueueNDRangeKernel(env.command_queue, env.kernel, 1, NULL, env.config.global_work_size, env.config.local_work_size, 0, NULL, NULL), __LINE__);
}

/* Copies the input image to a GPU buffer.*/
void init_image(cl_call_env env,
		cl_mem *alg_input,	// The stack will be copied to this buffer
		cl_mem *work,		// The working area
		vector<Mat> input_stack,
		point frame,
		clip img_clip,
		dimensions image_input_dims,
		dimensions image_work_dims)
{

	int img_input_size = image_input_dims.x*image_input_dims.y;
	cl_call(clSetKernelArg(env.kernel, 0, sizeof(cl_mem), (void*)alg_input), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 1, sizeof(cl_mem), (void*)work), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 2, sizeof(point), &frame), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 3, sizeof(clip), &img_clip),__LINE__);
	cl_call(clSetKernelArg(env.kernel, 4, sizeof(dimensions), &image_work_dims),__LINE__);
	for(int i = 0; i < image_work_dims.stacks; i++){
		int stack_slice_size = image_input_dims.x*image_input_dims.y;

		cl_call(clEnqueueWriteBuffer(
				env.command_queue,
				*alg_input,
				CL_TRUE,
				i*stack_slice_size,
				sizeof(uchar)*img_input_size,
				input_stack[i].data, 0, NULL, NULL), __LINE__);
	}

	*env.ret = clEnqueueNDRangeKernel(env.command_queue, env.kernel, 1, NULL, env.config.global_work_size, env.config.local_work_size, 0, NULL, NULL); cl_call(*env.ret, __LINE__);
}

void zero_image(cl_call_env env,
		cl_mem *fa,
		cl_mem *fb,
		dimensions image_dims)
{
	cl_call(clSetKernelArg(env.kernel, 0, sizeof(cl_mem), (void*)fa), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 1, sizeof(cl_mem), (void*)fb), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 2, sizeof(dimensions), &image_dims), __LINE__);
	cl_call(clEnqueueNDRangeKernel(env.command_queue, env.kernel, 1, NULL, env.config.global_work_size, env.config.local_work_size, 0, NULL, NULL), __LINE__);
}

/*
 * Computes:
 *
 * diff = sign(dirx)*dirx^2*gx + sign(diry)*diry^2*gy;
 *
 */
void compute_diff(cl_call_env env,
		float dirx,
		float diry,
		cl_mem * f,
		cl_mem * diff,
		dimensions image_dims)
{
	cl_call(clSetKernelArg(env.kernel, 0, sizeof(float), &dirx), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 1, sizeof(float), &diry), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 2, sizeof(cl_mem), (void*)f), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 3, sizeof(cl_mem), (void*)diff), __LINE__);
	cl_call(clSetKernelArg(env.kernel, 4, sizeof(dimensions), &image_dims), __LINE__);
	cl_call(clEnqueueNDRangeKernel(env.command_queue, env.kernel, 1, NULL, env.config.global_work_size, env.config.local_work_size, 0, NULL, NULL), __LINE__);
}

void setDirection(float direction, dimensions img_dims){
	if(direction < 90){
		cout << "1." << endl;
		frame.x = img_dims.x-1;
		frame.y = 0;
		img_clip = getClip(0, img_dims.x-3, 1, img_dims.y-2);
		img_clip.x1 = 0;
		strcpy(cl_header, CL_HEADER_A_0_90);
	}else if(direction >= 90 && direction < 180){
		cout << "2." << endl;
		frame.x = 0;
		frame.y = 0;
		img_clip = getClip(1, img_dims.x-2, 2, img_dims.y-1);
		strcpy(cl_header, CL_HEADER_A_90_180);
	}else if(direction >= 180 && direction < 270){
		cout << "3." << endl;
		frame.x = 0;
	    frame.y = img_dims.y-1;
		img_clip = getClip(2, img_dims.x-1, 1, img_dims.y-2);
	    strcpy(cl_header, CL_HEADER_A_180_270);
	}else{
		cout << "4." << endl;
	    frame.x = img_dims.x-1;
	    frame.y = img_dims.y-1;
		img_clip = getClip(1, img_dims.x-2, 0, img_dims.y-3);
		strcpy(cl_header, CL_HEADER_A_270_360);
	}
	if(verbose)
	printf("Direction: %f; White frame: x: %d, y: %d; Crop: x: [%d,%d], y: [%d,%d]\n",
			direction, frame.x, frame.y, img_clip.x1, img_clip.x2, img_clip.y1, img_clip.y2);
}

SimpleConfig getDefaultConf() {
	SimpleConfig conf;
	conf.addFromString(DEF_PARAMS);
	return conf;
}

void applySettings(SimpleConfig& conf) {
	selected_device = conf.getFProperty("deviceId");
	selected_platform = conf.getFProperty("platformId");

	verbose = conf.getIProperty("verbose");
	wAccept = conf.getFProperty("wAccept");
	wSmooth = conf.getFProperty("wSmooth");
	direction = conf.getFProperty("direction");
	nIter = conf.getIProperty("nIter");
	locSize = conf.getIProperty("locSize");

	inputStackPath = conf.getSProperty("input");
	outputStackPath = conf.getSProperty("output");

	kernel_source_path = conf.getSProperty("kernelSrcPath").c_str();
}

string getNextArg(int argc, char** argv, int& currArgId) {
	if (++currArgId >= argc) {
		throw string("No value for the parameter.");
	}
	return string(argv[currArgId]);
}

SimpleConfig parseArgs(int argc, char** argv){
	SimpleConfig conf = getDefaultConf();
	for (int argId = 1; argId < argc; argId++) {
		string currArg(argv[argId]);
		if (currArg.compare(STR_CONF) == 0) {
			string currArgVal = getNextArg(argc, argv, argId);
			conf.addFromString(currArgVal);
		}
		else if (currArg.compare(FILE_CONF) == 0) {
			string currArgVal = getNextArg(argc, argv, argId);
			conf.addFromConfigFile(currArgVal);
		}
		else {
			cerr << "Warning: can't understand the parameter." << endl;
		}
	}
	return conf;
}

void selectDevices(cl_device_id* selected_device_ids, cl_uint* num_selected_devices, int selected_platform, int selected_device){
	auto clInfo = getClInfo();

	if(selected_platform >= clInfo.first.size()){
		throw string("The selected OpenCL platform does not exist! ") +
				string("Re-run the application with the --get-devinfo parameter ") +
				string("in order to display the available platforms/devices in your computer.");
	}

	if(selected_device >= clInfo.second[selected_platform].size()){
		throw string("The selected OpenCL device does not found in the selected platform. ") +
				string("Re-run the application with the --get-devinfo parameter ") +
				string("in order to display the available platforms/devices in your computer.");
	}

	selected_device_ids[0] = clInfo.second[selected_platform][selected_device];
	*num_selected_devices = 1;

	cout << "Using platform: "
			<< getPlatformInfo(clInfo.first[selected_platform], CL_PLATFORM_NAME)
			<< ", OpenCL version: "
			<< getPlatformInfo(clInfo.first[selected_platform], CL_PLATFORM_VERSION)
			<< " with device "
			<< getDeviceInfo(selected_device_ids[0], CL_DEVICE_NAME)
			<< endl;
}

void launchReconstruction() {

	while(direction < 0.0f){
		direction += 360.0f;
	}

	while(direction >= 360.0f){
		direction -= 360.0f;
	}

	float directionRad = direction*M_PI / 180.0f;
	cout << "Direction (rad): " << directionRad << endl;

	float dirx = cos(directionRad);
	float diry = -sin(directionRad);

	vector<Mat> input_stack;
	vector<Mat> output_stack;

	kernel kernel_resource;

	// We select the devices for the context.
	cl_device_id selected_device_ids[10];
	cl_uint num_selected_devices;
	// We only use one device at the same time
	cl_device_id active_device_id;

	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_program program = NULL;

	cl_int ret;

	cl_mem imin;
	cl_mem imout;
	cl_mem gOrig;
	cl_mem fa;
	cl_mem fb;
	cl_mem diff;

	input_stack = loadStack(inputStackPath, nStackElements);

	dimensions input_stack_dims = { input_stack[0].cols, input_stack[0].rows, nStackElements };
	dimensions padded_stack_dims = { (input_stack[0].cols) + 2, (input_stack[0].rows) + 2, nStackElements };

	if(verbose)
		cout << "Stack size: " << input_stack_dims.x << "x" << input_stack_dims.y << "x" << input_stack_dims.stacks << endl;
		cout << "Stack size (padded): " << padded_stack_dims.x << "x" << padded_stack_dims.y << "x" << padded_stack_dims.stacks << endl;
	int input_stack_slice_size = input_stack_dims.x * input_stack_dims.y;
	int padded_stack_slice_size = padded_stack_dims.x * padded_stack_dims.y;
	int padded_stack_size = padded_stack_dims.x * padded_stack_dims.y * padded_stack_dims.stacks;
	int input_stack_size = padded_stack_dims.x * padded_stack_dims.y * nStackElements;

	setDirection(direction, padded_stack_dims);

	output_stack = initOutputStack(input_stack_dims, nStackElements);

	// Only choose one computing device!
	selectDevices(selected_device_ids, &num_selected_devices, selected_platform, selected_device);
	active_device_id = selected_device_ids[0];

	context = clCreateContext(NULL, num_selected_devices, selected_device_ids, NULL, NULL, &ret);
	cl_call(ret, __LINE__);

	::size_t glob_work_size[1];
	glob_work_size[0] = (padded_stack_slice_size / locSize + 1)*locSize;
	::size_t loc_work_size[1] = { locSize };

	kernel_config config = { glob_work_size, loc_work_size };
	if(verbose)
		cout << "Thread group config: global: " << glob_work_size[0] << ", local: " << loc_work_size[0] << endl;

	command_queue = clCreateCommandQueue(context, active_device_id, 0, &ret);
	cl_call(ret, __LINE__);

	/* Prepare the buffers. */
	imin = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uchar)*input_stack_size, NULL, &ret);
	cl_call(ret, __LINE__);

	imout = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*padded_stack_size, NULL, &ret);
	cl_call(ret, __LINE__);

	gOrig = create_float_buffer(context, padded_stack_size, &ret);
	fa = create_float_buffer(context, padded_stack_size, &ret);
	fb = create_float_buffer(context, padded_stack_size, &ret);
	diff = create_float_buffer(context, padded_stack_size, &ret);

	if(verbose)
	cout << "Loading OpenCL kernel: " << kernel_source_path << endl;

	kernel_resource = readKernelFile(const_cast<char*>(kernel_source_path.c_str()));

	// Build program
	const char *sources[2];
	sources[0] = cl_header;
	sources[1] = kernel_resource.src;

	program = clCreateProgramWithSource(context, 2, sources, NULL, &ret);
	cl_call(ret, __LINE__);

	ret = clBuildProgram(program, 1, &active_device_id, "", NULL, NULL);
	cl_program_diag(ret, program, active_device_id);

	// Prepare the kernels.
	cl_kernel kernel_zero_array = clCreateKernel(program, "zero_array", &ret);
	cl_call(ret, __LINE__);

	cl_kernel kernel_init_working_image = clCreateKernel(program, "init_working_image", &ret);
	cl_call(ret, __LINE__);

	cl_kernel kernel_comute_diff = clCreateKernel(program, "compute_diff", &ret);
	cl_call(ret, __LINE__);

	cl_kernel kernel_get_result = clCreateKernel(program, "get_result", &ret);
	cl_call(ret, __LINE__);

	cl_kernel kernel_update = clCreateKernel(program, "update", &ret);
	cl_call(ret, __LINE__);

	if(verbose)
		cout << "Copying stack to the device..." << endl;

	// Initialise the image to float in gOrig.
	cl_call_env callEnv;
	callEnv.kernel = kernel_init_working_image;
	callEnv.command_queue = command_queue;
	callEnv.config = config;
	callEnv.ret = &ret;
	init_image(callEnv,
		&imin,
		&gOrig,
		input_stack,
		frame,
		img_clip,
		input_stack_dims,
		padded_stack_dims);

	if(verbose)
		cout << "Executing stack reconstruction..." << endl;

	callEnv.kernel = kernel_zero_array;
	zero_image(callEnv,
		&fa, &fb,
		padded_stack_dims);

	callEnv.kernel = kernel_comute_diff;
	compute_diff(callEnv,
		dirx,
		diry,
		&gOrig,
		&diff,
		padded_stack_dims);

	cl_mem *src = &fa;
	cl_mem *target = &fb;
	cl_mem *tmp;

	callEnv.kernel = kernel_update;
	int iter = 0;
	while (iter < nIter) {
		iter++;
		update_kernel(callEnv,
			src,
			target,
			&diff, //&gOrig,
			dirx,
			diry,
			wAccept,
			wSmooth,
			padded_stack_dims);
		tmp = src;
		src = target;
		target = tmp;
	}

	if(verbose)
		cout << "Reading out result..." << endl;

	// Read out the results
	cl_call(clSetKernelArg(kernel_get_result, 0, sizeof(cl_mem), (void*)&fa), __LINE__);
	cl_call(clSetKernelArg(kernel_get_result, 1, sizeof(cl_mem), (void*)&imout), __LINE__);
	cl_call(clSetKernelArg(kernel_get_result, 2, sizeof(dimensions), &padded_stack_dims), __LINE__);
	cl_call(clEnqueueNDRangeKernel(command_queue, kernel_get_result, 1, NULL, glob_work_size, loc_work_size, 0, NULL, NULL), __LINE__);


	for (int i = 0; i < nStackElements; i++) {

		float* out_img = new float[padded_stack_slice_size];
		cl_call(clEnqueueReadBuffer(
			command_queue,
			imout,
			CL_TRUE,
			padded_stack_slice_size * sizeof(float)*i,
			sizeof(float)*padded_stack_slice_size,
			out_img,
			0, NULL, NULL), __LINE__);

		normalise_and_scale(out_img, output_stack[i], padded_stack_dims, img_clip);
	}

	cl_call(clReleaseMemObject(imin), __LINE__);
	cl_call(clReleaseMemObject(imout), __LINE__);
	cl_call(clReleaseMemObject(gOrig), __LINE__);
	cl_call(clReleaseMemObject(fa), __LINE__);
	cl_call(clReleaseMemObject(fb), __LINE__);
	cl_call(clReleaseMemObject(diff), __LINE__);

	cl_call(clReleaseKernel(kernel_init_working_image), __LINE__);
	cl_call(clReleaseKernel(kernel_zero_array), __LINE__);
	cl_call(clReleaseKernel(kernel_get_result), __LINE__);
	cl_call(clReleaseKernel(kernel_comute_diff), __LINE__);
	cl_call(clReleaseKernel(kernel_update), __LINE__);

	cl_call(clReleaseProgram(program), __LINE__);
	cl_call(clReleaseCommandQueue(command_queue), __LINE__);
	cl_call(clReleaseContext(context), __LINE__);

	free(kernel_resource.src);

	saveStack(outputStackPath, output_stack);
}

void printDeviceInfo(){
	auto cl_info = getClInfo();

	for(int pid = 0; pid < cl_info.first.size(); pid++){

		cout << pid << SEP;
		for(int did = 0; did < cl_info.second[pid].size(); did++){
			cl_platform_id plat_id = cl_info.first[pid];
			cl_device_id dev_id = cl_info.second[pid][did];
			cout << did
					<< SEP
					<< getDeviceInfo(dev_id, CL_DEVICE_NAME)
					<< " (" << getDeviceInfo(dev_id, CL_DEVICE_VENDOR) << ")"
					<< endl;
		}

	}
}

int main(int argc, char** argv){
	if(argc > 0){
		if(string(argv[1]).compare(DEVINFO) == 0){
			try{
				printDeviceInfo();
			}catch(string& err){
				cerr << err << endl;
				return RETURN_ERROR;
			}catch(...){
				cerr << "Unknown error." << endl;
				return RETURN_ERROR;
			}
			return RETURN_SUCCESS;
		}
	}

	SimpleConfig conf = parseArgs(argc, argv);
	applySettings(conf);

	if(verbose)
	conf.printSettings();

	unsigned int return_code = RETURN_SUCCESS;

	try {
		launchReconstruction();
	}
	catch (string& error) {
		cerr << "Error during the reconstruction: " << error << endl;
		return_code = RETURN_ERROR;
	}
	catch (...) {
		cerr << "Unknown error during the reconstruction" << endl;
		return_code = RETURN_ERROR;
	}

	return return_code;
}

