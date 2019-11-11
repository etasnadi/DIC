/*
 * clio.c
 *
 *  Created on: May 8, 2017
 *      Author: etasnadi
 */
#include <CL/cl.hpp>
#include <iostream>
#include <stdio.h>

#include <CL/cl.hpp>

#include "clio.h"

using namespace std;

string getErrorString(cl_int error){
	switch(error){
		// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
    }
}

void cl_call(cl_int cl_call_return, int line){
	//cout << "Kernel executed with code: " << cl_call_return << endl;
	if(cl_call_return != CL_SUCCESS){
		cerr << "OpenCL error! Message: \"" << getErrorString(cl_call_return) << "\"" << endl << "Line: " << line << endl;
	}
}

void cl_program_diag(cl_int ret, cl_program program, cl_device_id device_id){
	//IF_VERBOSE
	//printf("OpenCL diagnostics:\n");
	if(ret != CL_SUCCESS){
		size_t max_log_size = 1000000;
		char* build_log = (char*) malloc(max_log_size * sizeof(char));
		size_t returned_log_size;
		clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, max_log_size, (void*) build_log, &returned_log_size);
		//IF_VERBOSE
		//fprintf(stderr, "Build log:\n==========\n%s\n==========\n", build_log);
		if(returned_log_size > max_log_size){
			fprintf(stderr, "Warning: the returned log size doesn't fit the allocated space for the log.\n");
		}
	}else{
		//IF_VERBOSE
		//printf("The program compiled successfully.\n");
	}
}

kernel readKernelFile(char* path){
	FILE *fp;
	char *kernel_src_str;
	size_t kernel_code_size;

	fp = fopen(path, "r");
	if (fp == NULL) {
		throw string("Can't open the kernel source file!");
	}
	kernel_src_str = (char*)malloc(MAX_SOURCE_SIZE);
	kernel_code_size = fread(kernel_src_str, 1, MAX_SOURCE_SIZE, fp);
	kernel_src_str[kernel_code_size] = 0;
	fclose(fp);
	//printf("--> Loading kernel: %s\nKernel source size: %d\nKernel source:\n==========\n%s\n==========\n", path, (int)kernel_code_size, kernel_src_str);
	kernel k;
	k.src = kernel_src_str;
	k.src_size = kernel_code_size;
	return k;
}

string getPlatformInfo(cl_platform_id platform_id, cl_platform_info info){
	::size_t nSiz = 100;
	char name[nSiz];
	cl_call(clGetPlatformInfo(platform_id, info, nSiz, name, NULL), __LINE__);
	return string(name);
}

string getDeviceInfo(cl_device_id device_id, cl_device_info info){
	::size_t nSiz = 100;
	char name[nSiz];
	cl_call(clGetDeviceInfo(device_id, info, nSiz, name, NULL), __LINE__);
	return string(name);
}

pair<vector<cl_platform_id>, vector<vector<cl_device_id>>> getClInfo(){
	vector<vector<cl_device_id>> available_device_ids;
	vector<cl_platform_id> available_platform_ids;

	int maxEntries = 10;

	vector<cl_platform_id> platform_ids(maxEntries);
	cl_uint nPlatforms;
	cl_call(clGetPlatformIDs(maxEntries, &platform_ids[0], &nPlatforms), __LINE__);

	available_platform_ids = vector<cl_platform_id>(nPlatforms);
	available_device_ids = vector<vector<cl_device_id>>(nPlatforms);
	for(int pid = 0; pid < nPlatforms; pid++){
		vector<cl_device_id> device_ids(maxEntries);
		cl_uint nDevices;
		cl_call(clGetDeviceIDs(platform_ids[pid], CL_DEVICE_TYPE_ALL, maxEntries, &device_ids[0],
			&nDevices), __LINE__);
		available_platform_ids[pid] = platform_ids[pid];
		available_device_ids[pid] = vector<cl_device_id>(nDevices);

		for(int did = 0; did < nDevices; did++){
			available_device_ids[pid][did] = device_ids[did];
		}
	}
	return make_pair(available_platform_ids, available_device_ids);
}

