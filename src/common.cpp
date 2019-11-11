#include <float.h>
#include "common.h"

float findMax(float* arr, int size){
	float max = -FLT_MAX;
	for(int i = 0; i < size; i++){
		if(arr[i] > max){
			max = arr[i];
		}
	}
	return max;
}

float findMin(float* arr, int size){
	float min = FLT_MAX;
	for(int i = 0; i < size; i++){
		if(arr[i] < min){
			min = arr[i];
		}
	}
	return min;
}

int get_sign(float f){
	if(f == 0){
		return 0;
	}else if(f < 0){
		return -1;
	}else{
		return 1;
	}
}

