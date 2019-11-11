#ifndef TEST_H
#define TEST_H

#include "imageio.h"

#define MAX_MASK_SIZE 10

// OpenCL compatible mask specification
typedef struct {
	size_t mask_size;
	dimensions dim_selector;
} mask_spec1D;

// Convolution mask with specification and the data.
typedef struct {
	mask_spec1D mask_spec;
	float *mask;
} mask1D;

typedef struct {
	size_t *global_work_size;
	size_t *local_work_size;
} kernel_config;

typedef struct {
	cl_kernel kernel;
	cl_command_queue command_queue;
	kernel_config config;
	cl_int *ret;
} cl_call_env;

#endif
