#ifndef CLIO_H
#define CLIO_H

#include <CL/cl.hpp>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

#define MAX_SOURCE_SIZE 100000

extern int verbose;

typedef struct {
	char* src;
	int src_size;
} kernel;

string getErrorString(cl_int error);
void cl_program_diag(cl_int ret, cl_program program, cl_device_id device_id);
void cl_call(cl_int cl_call_return, int line);
kernel readKernelFile(char* path);
string getDeviceInfo(cl_device_id device_id, cl_device_info info);
string getPlatformInfo(cl_platform_id platform_id, cl_platform_info info);
pair<vector<cl_platform_id>, vector<vector<cl_device_id>>> getClInfo();

#endif
