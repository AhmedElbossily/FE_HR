//Copyright ETH Zurich, IWF

//This file is part of iwf_mfree_gpu_3d.

//iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "types.h"

__host__ __device__  float4_t cubic_spline(float4_t posi, float4_t posj, float_t hi) {

	float4_t w;
	w.x = 0.;
	w.y = 0.;
	w.z = 0.;
	w.w = 0.;

	float_t xij = posi.x-posj.x;
	float_t yij = posi.y-posj.y;
	float_t zij = posi.z-posj.z;

	float_t rr2=xij*xij+yij*yij+zij*zij;
	float_t h1 = 1/hi;
	float_t fourh2 = 4*hi*hi;
	if(rr2>=fourh2 || rr2 < 1e-8) {
		return w;
	}

	float_t der;
	float_t val;

	float_t rad=sqrtf(rr2);
	float_t q=rad*h1;
	float_t fac = (M_1_PI)*h1*h1*h1;	// 3D*h1*h1;

	const bool radgt=q>1;
	if (radgt) {
		float_t _2mq  = 2-q;
		float_t _2mq2 = _2mq*_2mq;
		val = 0.25*_2mq2*_2mq;
		der = -0.75*_2mq2 * h1/rad;
	} else {
		val = 1 - 1.5*q*q*(1-0.5*q);
		der = -3.0*q*(1-0.75*q) * h1/rad;
	}

	w.x = val*fac;
	w.y = der*xij*fac;
	w.z = der*yij*fac;
	w.w = der*zij*fac;

	return w;
}


__host__ __device__  float4_t hyperbolic_spline(float4_t posi, float4_t posj, float_t hi) {

	float4_t w;
	w.x = 0.;
	w.y = 0.;
	w.z = 0.;
	w.w = 0.;

	float_t xij = posi.x-posj.x;
	float_t yij = posi.y-posj.y;
	float_t zij = posi.z-posj.z;

	float_t rr2=xij*xij+yij*yij+zij*zij;
	float_t h1 = 1/hi;
	float_t fourh2 = 4*hi*hi;
	if(rr2>=fourh2 || rr2 < 1e-8) {
		return w;
	}

	float_t der;
	float_t val;

	float_t rad=sqrtf(rr2);
	float_t q=rad*h1;
	float_t fac = 15/(62* M_PI *hi*hi*hi);	// 3D*h1*h1;



 	const bool radgt=q>1;
	if (radgt) {
        val = (2 - q)*  (2 - q) * (2 - q);
		der = - 3.0 *  (2 - q) *  (2 - q) * h1/rad;
    }else{
		val = q *q *q - 6* q + 6;
		der = (3.0 * q * q - 6 ) * h1/rad;
	} 

	
/* 	if (q >= 0 && q < 1) {
        val = q *q *q - 6* q + 6;
		der = (3.0 * q * q - 6 ) * h1/rad;
    }
    else if (q >= 1 && q < 2) {
        val = (2 - q)*  (2 - q) * (2 - q);
		der = - 3.0 *  (2 - q) *  (2 - q) * h1/rad;
    }
    else {
        val = 0.0f;
    } 
*/

	w.x = val*fac;
	w.y = der*xij*fac;
	w.z = der*yij*fac;
	w.w = der*zij*fac;
	return w;
} 
__host__ __device__  float_t lapl_pse(float4_t posi, float4_t posj, float_t hi) {
	float_t xi = posi.x;
	float_t yi = posi.y;
	float_t zi = posi.z;

	float_t xj = posj.x;
	float_t yj = posj.y;
	float_t zj = posj.z;

	float_t xij = xi-xj;
	float_t yij = yi-yj;
	float_t zij = zi-zj;

	float_t xx = sqrt(xij*xij + yij*yij + zij*zij);

	float_t h2 = hi*hi;
	float_t h4 = h2*h2;
	float_t h5 = hi*h4;
	float_t w2_pse  = +4./(h5*M_PI*sqrt(M_PI))*exp(-xx*xx/(h2));

	return w2_pse;
}

__host__ __device__ float_t hyperbolic_spline_val(float_t q, float_t h) {
    const float_t pi = 3.14159265358979323846;
    const float_t coeff = 15.0 / (7.0  *pi*  pow(h, 3));
    const float_t r = q / h;
    const float_t term = pow(1 - r * r, 3);
    return coeff * term;
}

__host__ __device__ float_t hyperbolic_spline_derivative(float_t q, float_t h) {
    const float_t pi = 3.14159265358979323846;
    const float_t coeff = -45.0 / (7.0  *pi*  pow(h, 4));
    const float_t r = q / h;
    const float_t term = r  *pow(1 - r*  r, 2);
    return coeff * term;
}
/* 
__host__ __device__  float4_t hyperbolic_spline(float4_t posi, float4_t posj, float_t hi) {

	float4_t w;
	w.x = 0.;
	w.y = 0.;
	w.z = 0.;
	w.w = 0.;

	float_t xij = posi.x-posj.x;
	float_t yij = posi.y-posj.y;
	float_t zij = posi.z-posj.z;

	float_t rr2=xij*xij+yij*yij+zij*zij;
	float_t h1 = 1/hi;
	float_t fourh2 = 4*hi*hi;
	if(rr2>=fourh2 || rr2 < 1e-8) {
		return w;
	}

	float_t rad=sqrtf(rr2);
	float_t q=rad*h1;
	float_t val = hyperbolic_spline_val( q, hi) ;
	float_t der = hyperbolic_spline_derivative( q, hi);

	w.x = val;
	w.y = der*xij;
	w.z = der*yij;
	w.w = der*zij;
	return w;
} */
#endif /* KERNELS_CUH_ */
