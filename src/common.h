/*
 * common.h
 *
 *  Created on: 2018 szept. 15
 *      Author: ervin
 */

#ifndef COMMON_H
#define COMMON_H

#ifndef M_PI 
#define M_PI 3.14159265
#endif

extern int verbose;

float findMax(float* arr, int size);
float findMin(float* arr, int size);

int get_sign(float f);

#endif /* COMMON_H_ */
