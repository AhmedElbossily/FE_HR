// Copyright ETH Zurich, IWF

// This file is part of iwf_mfree_gpu_3d.

// iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

#include "interactions_gpu.h"

#include <thrust/device_vector.h>

#include "eigen_solver.cuh"
#include "kernels.cuh"

// physical constants on device
__constant__ static phys_constants physics;
__constant__ static corr_constants correctors;
__constant__ static geom_constants geometry;
__constant__ static trml_constants trml_sub;
__constant__ static trml_constants trml_rod;
__constant__ static rod_constants rod;

__constant__ static joco_constants johnson_cook_sub;
__constant__ static joco_constants johnson_cook_rod;

// thermal constants on host
static trml_constants h_thermals_sub;
static trml_constants h_thermals_rod;

// is there thermal conduction in workpiece (and/or into tool?)
static bool m_thermal_workpiece = false;
static bool m_thermal_tool = false;

// textures for fast access to read only attribtues in interactions
texture<float4_tex_t, 1, cudaReadModeElementType> pos_tex;
texture<float4_tex_t, 1, cudaReadModeElementType> vel_tex;

// texture<float4_tex_t, 1, cudaReadModeElementType> S_tex;	//can't put 3x3 mats in textures
// texture<float4_tex_t, 1, cudaReadModeElementType> R_tex;

texture<float_tex_t, 1, cudaReadModeElementType> h_tex;
texture<float_tex_t, 1, cudaReadModeElementType> rho_tex;
texture<float_tex_t, 1, cudaReadModeElementType> p_tex;
texture<float_tex_t, 1, cudaReadModeElementType> T_tex;
texture<float_tex_t, 1, cudaReadModeElementType> tool_particle_tex;
texture<float_tex_t, 1, cudaReadModeElementType> k_tex;

// textures for fast access to hashing information
texture<int, 1, cudaReadModeElementType> hashes_tex;

texture<int, 1, cudaReadModeElementType> cells_start_tex;
texture<int, 1, cudaReadModeElementType> cells_end_tex;

#ifdef USE_DOUBLE
static __inline__ __device__ double fetch_double(texture<int2, 1> t, int i)
{
	int2 v = tex1Dfetch(t, i);
	return __hiloint2double(v.y, v.x);
}

static __inline__ __device__ double2 fetch_double(texture<int4, 1> t, int i)
{
	int4 v = tex1Dfetch(t, i);
	return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}

static __inline__ __device__ double4 fetch_double2(texture<int4, 1> t, int i)
{
	int4 v1 = tex1Dfetch(t, 2 * i + 0);
	int4 v2 = tex1Dfetch(t, 2 * i + 1);

	return make_double4(__hiloint2double(v1.y, v1.x), __hiloint2double(v1.w, v1.z),
						__hiloint2double(v2.y, v2.x), __hiloint2double(v2.w, v2.z));
}
#endif

__device__ __forceinline__ bool isnaninf(float_t val)
{
	return isnan(val) || isinf(val);
}

__device__ __forceinline__ void hash(int i, int j, int k, int &idx)
{
	// idx = i*geometry.ny*geometry.nz+j*geometry.nz + k;

	// hashes[idx] = iz*ny*nx + iy*nx + ix;
	idx = k * geometry.ny * geometry.nx + j * geometry.nx + i;
}

__device__ __forceinline__ void unhash(int &i, int &j, int &k, int idx)
{
	/* i = idx / (geometry.nz*geometry.ny);
	j = (idx - i*geometry.ny*geometry.nz) / geometry.nz;
	k = idx % geometry.nz; */

	// for hashes[idx] = iz*ny*nx + iy*nx + ix;

	k = idx / (geometry.ny * geometry.nx);
	j = (idx - k * geometry.ny * geometry.nx) / geometry.nx;
	i = idx % geometry.nx;
}

__global__ void do_interactions_heat(float_t *T_t, float_t *cp, int N, float_t alpha_wp, float_t alpha_tool)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	// load physical constants
	float_t mass = physics.mass;

	// load particle data at pidx
	float4_t pi = texfetch4(pos_tex, pidx);
	float_t hi = texfetch1(h_tex, pidx);
	float_t Ti = texfetch1(T_tex, pidx);
	float_t ki = texfetch1(k_tex, pidx);
	float_t rhoi = texfetch1(rho_tex, pidx);

	float_t alpha = 1.0 / (rhoi * cp[pidx]);

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	int gd = geometry.gd;

	int low_i = gi - gd < 0 ? 0 : gi - gd;
	int low_j = gj - gd < 0 ? 0 : gj - gd;
	int low_k = gk - gd < 0 ? 0 : gk - gd;

	int shift_high = gd + 1;
	int high_i = gi + shift_high > nx ? nx : gi + shift_high;
	int high_j = gj + shift_high > ny ? ny : gj + shift_high;
	int high_k = gk + shift_high > nz ? nz : gk + shift_high;

	float_t T_ti = 0.;

	// iterate over neighboring boxes
	for (int kk = low_k; kk < high_k; kk++)
	{
		for (int jj = low_j; jj < high_j; jj++)
		{
			for (int ii = low_i; ii < high_i; ii++)
			{

				int idx;
				hash(ii, jj, kk, idx);

				int c_start = tex1Dfetch(cells_start_tex, idx);
				int c_end = tex1Dfetch(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					float4_t pj = texfetch4(pos_tex, iter);
					float_t Tj = texfetch1(T_tex, iter);
					float_t rhoj = texfetch1(rho_tex, iter);
					float_t kj = texfetch1(k_tex, iter);

					float_t w2_pse = lapl_pse(pi, pj, hi); // Laplacian by PSE-method

					float_t is_tool_particle_j = texfetch1(tool_particle_tex, iter);
					if (is_tool_particle_j == 0 || is_tool_particle_j == 2)
						mass = 1.0 * 1.0 * 1.0 * 1000.0 * 7800.0 * 1.0e-6 ;	///////////////////////////////////////////////////////

					T_ti += ((2 * ki * kj) / (ki + kj)) * (mass / rhoj) * (Tj - Ti) * w2_pse;
				}
			}
		}
	}

	T_t[pidx] = alpha * T_ti;
}

__global__ void do_interactions_monaghan(float3_t *__restrict__ cfm, const float_t *__restrict__ blanked, const mat3x3_t *__restrict__ S,
										 const mat3x3_t *__restrict__ R, mat3x3_t *__restrict__ v_der, mat3x3_t *__restrict__ S_der,
										 float_t *__restrict__ T_t, float3_t *__restrict__ pos_t, float3_t *__restrict__ vel_t,
										 unsigned int N, const float_t *__restrict__ in_tool, const float_t *__restrict__ fixed,
										 float_t dt, float_t *__restrict__ cp, float_t *__restrict__ E)
{

	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;

	float_t is_tool_particle_i = texfetch1(tool_particle_tex, pidx);
	if (is_tool_particle_i == 0 || is_tool_particle_i == 2)
		return;

	// load particle data at pidx
	float4_t pi = texfetch4(pos_tex, pidx);
	float4_t vi = texfetch4(vel_tex, pidx);

	bool is_billet = is_tool_particle_i == 1;

	// load physical constants
	float_t mass = physics.mass;

	float_t physics_G = E[pidx] / (2. * (1. + physics.nu));
	float_t physics_K = 2.0 * physics_G * (1 + physics.nu) / (3 * (1 - 2 * physics.nu));
	float_t K = physics_K;
	// float_t K = physics.K;
	//  load correction constants
	float_t wdeltap = correctors.wdeltap;
	float_t alpha = correctors.alpha;
	float_t beta = correctors.beta;
	float_t eta = correctors.eta;
	float_t eps_rod = correctors.xspheps;
	float_t eps_sub = correctors.xspheps;

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	mat3x3_t Si = S[pidx];
	mat3x3_t Ri = R[pidx];
	float_t hi = texfetch1(h_tex, pidx);
	float_t rhoi = texfetch1(rho_tex, pidx);
	float_t prsi = texfetch1(p_tex, pidx);

	float_t rhoi21 = 1. / (rhoi * rhoi);

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	int gd = geometry.gd;

	int low_i = gi - gd < 0 ? 0 : gi - gd;
	int low_j = gj - gd < 0 ? 0 : gj - gd;
	int low_k = gk - gd < 0 ? 0 : gk - gd;

	int shift_high = gd + 1;
	int high_i = gi + shift_high > nx ? nx : gi + shift_high;
	int high_j = gj + shift_high > ny ? ny : gj + shift_high;
	int high_k = gk + shift_high > nz ? nz : gk + shift_high;

	// init vars to be written at pidx
	mat3x3_t vi_der(0.);
	mat3x3_t Si_der(0.);
	float3_t vi_t = make_float3_t(0., 0., 0.);
	float3_t vi_adv_t = make_float3_t(0., 0., 0.);
	float3_t xi_t = make_float3_t(0., 0., 0.);
	float3_t cfmi = make_float3_t(0., 0., 0.);
	float_t cfmi_count(0.0);

#ifdef Thermal_Conduction_Brookshaw
	float_t ki = texfetch1(k_tex, pidx);

	float_t T_lapl = 0.; // Laplacian of temperature field
	float_t Ti = 0.;
	float_t thermal_alpha = 0.;

	if (is_billet)
		thermal_alpha = trml_sub.alpha;
	else
		thermal_alpha = trml_rod.alpha;

	if (thermal_alpha != 0.)
	{
		Ti = texfetch1(T_tex, pidx);
	}
#endif
#ifdef CSPM
	mat3x3_t B(0.);

	if (/* !is_tool_particle_i == 1. */ true)
	{

		// iterate over neighboring boxes
		for (int kk = low_k; kk < high_k; kk++)
		{
			for (int jj = low_j; jj < high_j; jj++)
			{
				for (int ii = low_i; ii < high_i; ii++)
				{

					int idx;
					hash(ii, jj, kk, idx);

					// iterate over particles contained in a neighboring box
					int c_start = tex1Dfetch(cells_start_tex, idx);
					int c_end = tex1Dfetch(cells_end_tex, idx);

					if (c_start == 0xffffffff)
						continue;

					for (int iter = c_start; iter < c_end; iter++)
					{

						if (blanked[iter] == 1. /* || fixed[iter] == 1*/)
						{
							continue;
						}

						float_t is_tool_particle_j = texfetch1(tool_particle_tex, iter);

						if (is_tool_particle_i == is_tool_particle_j)
						{

							float4_t pj = texfetch2(pos_tex, iter);
							float_t rhoj = texfetch1(rho_tex, iter);

							const float_t volj = mass / rhoj;

							float4_t ww = cubic_spline(pi, pj, hi);
							// float4_t ww = hyperbolic_spline(pi, pj, hi);

							float_t w_x = ww.y;
							float_t w_y = ww.z;
							float_t w_z = ww.w;

							const float_t delta_x = pi.x - pj.x;
							const float_t delta_y = pi.y - pj.y;
							const float_t delta_z = pi.z - pj.z;

							// copute CSPM / Randles Libersky Correction Matrix
							B[0][0] -= volj * delta_x * w_x;
							B[1][0] -= volj * delta_x * w_y;
							B[2][0] -= volj * delta_x * w_z;

							B[0][1] -= volj * delta_y * w_x;
							B[1][1] -= volj * delta_y * w_y;
							B[2][1] -= volj * delta_y * w_z;

							B[0][2] -= volj * delta_z * w_x;
							B[1][2] -= volj * delta_z * w_y;
							B[2][2] -= volj * delta_z * w_z;
						}
					}
				}
			}
		}
	}

	// save invert
	mat3x3_t invB(1.);
	float_t det_B = glm::determinant(B);
	if (det_B > 1e-8)
	{
		invB = glm::inverse(B);
	}
#endif

	// iterate over neighboring boxes
	for (int kk = low_k; kk < high_k; kk++)
	{

		for (int jj = low_j; jj < high_j; jj++)
		{

			for (int ii = low_i; ii < high_i; ii++)
			{

				int idx;
				hash(ii, jj, kk, idx);

				// iterate over particles contained in a neighboring box
				int c_start = tex1Dfetch(cells_start_tex, idx);
				int c_end = tex1Dfetch(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					if (blanked[iter] == 1. /* || fixed[iter] == 1*/)
					{
						continue;
					}
					float_t is_tool_particle_j = texfetch1(tool_particle_tex, iter);

					// load vars at neighbor particle
					float4_t pj = texfetch4(pos_tex, iter);
					float_t hj = texfetch1(h_tex, iter);

					/****************** calculate the center of mass ********************************/
					float_t xij = pi.x - pj.x;
					float_t yij = pi.y - pj.y;
					float_t zij = pi.z - pj.z;
					float_t rij_2 = xij * xij + yij * yij + zij * zij;
					float_t rij = sqrt(rij_2);

					if (rij == 0.)
						continue;

					if ((rij <= hi) && (is_tool_particle_i == is_tool_particle_j))
					{ // perform only on the rod
						cfmi.x += mass * xij;
						cfmi.y += mass * yij;
						cfmi.z += mass * zij;

						cfmi_count++;
					}

					// compute kernel
					float4_t ww = cubic_spline(pi, pj, hi);
					// float4_t ww = hyperbolic_spline(pi, pj, hi);

					// correct by CSPM matrix if def'd
#ifndef CSPM
					float_t w = ww.x;
					float_t w_x = ww.y;
					float_t w_y = ww.z;
					float_t w_z = ww.w;
#else
					float_t w = ww.x;
					float_t w_x = (ww.y * invB[0][0] + ww.z * invB[1][0] + ww.w * invB[2][0]);
					float_t w_y = (ww.y * invB[0][1] + ww.z * invB[1][1] + ww.w * invB[2][1]);
					float_t w_z = (ww.y * invB[0][2] + ww.z * invB[1][2] + ww.w * invB[2][2]);
#endif
					if (w == 0 && w_x == 0 && w_y == 0 && w_z == 0)
						continue;

					float_t rhoj = texfetch1(rho_tex, iter);
					// if (!(is_tool_particle_i == 1.) && !(is_tool_particle_j == 1.)) {
					if (is_tool_particle_i == is_tool_particle_j)
					{

						float4_t vj = texfetch4(vel_tex, iter);
						float_t prsj = texfetch1(p_tex, iter);

						mat3x3_t Sj = S[iter];
						mat3x3_t Rj = R[iter];

						float_t volj = mass / rhoj;
						float_t rhoj21 = 1. / (rhoj * rhoj);

						vi_der[0][0] += (vj.x - vi.x) * w_x * volj;
						vi_der[0][1] += (vj.x - vi.x) * w_y * volj;
						vi_der[0][2] += (vj.x - vi.x) * w_z * volj;

						vi_der[1][0] += (vj.y - vi.y) * w_x * volj;
						vi_der[1][1] += (vj.y - vi.y) * w_y * volj;
						vi_der[1][2] += (vj.y - vi.y) * w_z * volj;

						vi_der[2][0] += (vj.z - vi.z) * w_x * volj;
						vi_der[2][1] += (vj.z - vi.z) * w_y * volj;
						vi_der[2][2] += (vj.z - vi.z) * w_z * volj;

						float_t Rxx = 0.;
						float_t Ryy = 0.;
						float_t Rzz = 0.;

						float_t Rxy = 0.;
						float_t Rxz = 0.;
						float_t Ryz = 0.;

						// compute artificial stress
						if (wdeltap > 0)
						{
							float_t fab = w / wdeltap;
							fab *= fab; // to the power of 4
							fab *= fab;

							Rxx = fab * (Ri[0][0] + Rj[0][0]);
							Rxy = fab * (Ri[0][1] + Rj[0][1]);
							Ryy = fab * (Ri[1][1] + Rj[1][1]);
							Rxz = fab * (Ri[0][2] + Rj[0][2]);
							Ryz = fab * (Ri[1][2] + Rj[1][2]);
							Rzz = fab * (Ri[2][2] + Rj[2][2]);
						}

						// derive stress original
						Si_der[0][0] += mass * ((Si[0][0] - prsi) * rhoi21 + (Sj[0][0] - prsj) * rhoj21 + Rxx) * w_x;
						Si_der[0][1] += mass * (Si[0][1] * rhoi21 + Sj[0][1] * rhoj21 + Rxy) * w_y;
						Si_der[0][2] += mass * (Si[0][2] * rhoi21 + Sj[0][2] * rhoj21 + Rxz) * w_z;

						Si_der[1][0] += mass * (Si[1][0] * rhoi21 + Sj[1][0] * rhoj21 + Rxy) * w_x;
						Si_der[1][1] += mass * ((Si[1][1] - prsi) * rhoi21 + (Sj[1][1] - prsj) * rhoj21 + Ryy) * w_y;
						Si_der[1][2] += mass * (Si[1][2] * rhoi21 + Sj[1][2] * rhoj21 + Ryz) * w_z;

						Si_der[2][0] += mass * (Si[2][0] * rhoi21 + Sj[2][0] * rhoj21 + Rxz) * w_x;
						Si_der[2][1] += mass * (Si[2][1] * rhoi21 + Sj[2][1] * rhoj21 + Ryz) * w_y;
						Si_der[2][2] += mass * ((Si[2][2] - prsi) * rhoi21 + (Sj[2][2] - prsj) * rhoj21 + Rzz) * w_z;

						float_t vijx = vi.x - vj.x;
						float_t vijy = vi.y - vj.y;
						float_t vijz = vi.z - vj.z;

						float_t vijposij = vijx * xij + vijy * yij + vijz * zij;
						float_t rhoij = 0.5 * (rhoi + rhoj);

						if (vijposij < 0.)
						{
							float_t ci = sqrtf(K / rhoi);
							float_t cj = sqrtf(K / rhoj);

							float_t cij = 0.5 * (ci + cj);
							float_t hij = 0.5 * (hi + hj);

							float_t r2ij = xij * xij + yij * yij + zij * zij;
							float_t muij = (hij * vijposij) / (r2ij + eta * eta * hij * hij);
							float_t piij = (-alpha * cij * muij + beta * muij * muij) / rhoij;

							vi_t.x += -mass * piij * w_x;
							vi_t.y += -mass * piij * w_y;
							vi_t.z += -mass * piij * w_z;
						}

						// add xsph correction
						xi_t.x += -eps_sub * w * mass / rhoij * vijx;
						xi_t.y += -eps_sub * w * mass / rhoij * vijy;
						xi_t.z += -eps_sub * w * mass / rhoij * vijz;
					}

#ifdef Thermal_Conduction_Brookshaw
					float_t Tj = 0.;
					// thermal, 3D Brookshaw
					if (thermal_alpha != 0.)
					{
						Tj = texfetch1(T_tex, iter);
						if (rij > 1e-8)
						{
							float_t eijx = xij / rij;
							float_t eijy = yij / rij;
							float_t eijz = zij / rij;
							float_t rij1 = 1. / rij;
							float_t kj = texfetch1(k_tex, iter);

							T_lapl += ((4 * ki * kj) / (ki + kj)) * (mass / rhoj) * (Ti - Tj) * rij1 * (eijx * w_x + eijy * w_y + eijz * w_z);
						}
					}
#endif
				}
			}
		}
	}

	// write back, biger than 4 is rigit

	S_der[pidx] = Si_der;
	v_der[pidx] = vi_der;

	pos_t[pidx] = xi_t;
	vel_t[pidx] = vi_t;

	// Boundaries
	if ((cfmi_count <= 120) && (cfmi_count > 0))
	{
		cfmi.x = cfmi.x / (cfmi_count * mass);
		cfmi.y = cfmi.y / (cfmi_count * mass);
		cfmi.z = cfmi.z / (cfmi_count * mass);

		cfm[pidx].x = cfmi.x;
		cfm[pidx].y = cfmi.y;
		cfm[pidx].z = cfmi.z;
	}
	else
	{
		cfmi.x = 0.0;
		cfmi.y = 0.0;
		cfmi.z = 0.0;

		cfm[pidx].x = cfmi.x;
		cfm[pidx].y = cfmi.y;
		cfm[pidx].z = cfmi.z;
	}

#ifdef Thermal_Conduction_Brookshaw
	if (thermal_alpha != 0.)
	{
		thermal_alpha = 1.0 / (rhoi * cp[pidx]);
		T_t[pidx] = thermal_alpha * T_lapl;
	}
#endif
}

__global__ void do_ni_smoothing(float3_t *__restrict__ cfm, int *__restrict__ surface, const float_t *__restrict__ blanked,
								unsigned int N, const float_t *__restrict__ in_tool, float3_t *sum_buff, int *count_buff, float_t *fixed)
{

	int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	int th = threadIdx.x;

	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;
	if ((cfm[pidx].x == 0) && (cfm[pidx].y == 0) && (cfm[pidx].z == 0))
		return;

	float_t in_tooli = in_tool[pidx];

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	// load particle data at pidx
	float4_t pi = texfetch4(pos_tex, pidx);
	float_t hi = texfetch1(h_tex, pidx);

	hi = hi / 2.;

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	int gd = geometry.gd;

	int low_i = gi - gd < 0 ? 0 : gi - gd;
	int low_j = gj - gd < 0 ? 0 : gj - gd;
	int low_k = gk - gd < 0 ? 0 : gk - gd;

	int shift_high = gd + 1;
	int high_i = gi + shift_high > nx ? nx : gi + shift_high;
	int high_j = gj + shift_high > ny ? ny : gj + shift_high;
	int high_k = gk + shift_high > nz ? nz : gk + shift_high;

	count_buff[pidx] = 0.0;

	// iterate over neighboring boxes
	for (int kk = low_k; kk < high_k; kk++)
	{
		for (int jj = low_j; jj < high_j; jj++)
		{
			for (int ii = low_i; ii < high_i; ii++)
			{

				int idx;
				hash(ii, jj, kk, idx);

				// iterate over particles contained in a neighboring box
				int c_start = tex1Dfetch(cells_start_tex, idx);
				int c_end = tex1Dfetch(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					float3_t cfmj = cfm[iter];

					// if (cfmj.x == 0. && cfmj.y == 0. && cfmj.z == 0.) continue;
					if ((cfm[iter].x == 0) && (cfm[iter].y == 0) && (cfm[iter].z == 0))
						continue;

					float4_t pj = texfetch4(pos_tex, iter);
					float_t in_toolj = in_tool[iter];

					/****************** calculate the center of mass ********************************/
					float_t xij = pi.x - pj.x;
					float_t yij = pi.y - pj.y;
					float_t zij = pi.z - pj.z;
					float_t r = sqrtf(xij * xij + yij * yij + zij * zij);

					if (r <= hi && (in_tooli == in_toolj))
					{
						sum_buff[pidx].x += cfmj.x;
						sum_buff[pidx].y += cfmj.y;
						sum_buff[pidx].z += cfmj.z;

						count_buff[pidx]++;
					}
				}
			}
		}
	}
}

__global__ void do_calculate_init_cfm(float3_t *__restrict__ cfm, const float_t *__restrict__ blanked,
									  unsigned int N, const float_t *__restrict__ in_tool, const float_t *__restrict__ fixed, int step)

{

	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;
	if (in_tool[pidx] == 1)
		return;
	if (step > 1)
		return;

	float_t is_tool_particle_i = texfetch1(tool_particle_tex, pidx);
	float4_t pi = texfetch4(pos_tex, pidx);

	// load physical constants
	float_t mass = physics.mass;
	float_t hi = texfetch1(h_tex, pidx);

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	int gd = geometry.gd;

	int low_i = gi - gd < 0 ? 0 : gi - gd;
	int low_j = gj - gd < 0 ? 0 : gj - gd;
	int low_k = gk - gd < 0 ? 0 : gk - gd;

	int shift_high = gd + 1;
	int high_i = gi + shift_high > nx ? nx : gi + shift_high;
	int high_j = gj + shift_high > ny ? ny : gj + shift_high;
	int high_k = gk + shift_high > nz ? nz : gk + shift_high;

	float3_t cfmi = make_float3_t(0., 0., 0.);
	float_t cfmi_count(0.0);

	// iterate over neighboring boxes
	for (int kk = low_k; kk < high_k; kk++)
	{

		for (int jj = low_j; jj < high_j; jj++)
		{

			for (int ii = low_i; ii < high_i; ii++)
			{

				int idx;
				hash(ii, jj, kk, idx);

				// iterate over particles contained in a neighboring box
				int c_start = tex1Dfetch(cells_start_tex, idx);
				int c_end = tex1Dfetch(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					if (blanked[iter] == 1. /* || fixed[iter] == 1*/)
					{
						continue;
					}
					float_t is_tool_particle_j = texfetch1(tool_particle_tex, iter);
					float4_t pj = texfetch4(pos_tex, iter);
					float_t xij = pi.x - pj.x;
					float_t yij = pi.y - pj.y;
					float_t zij = pi.z - pj.z;
					float_t rij_2 = xij * xij + yij * yij + zij * zij;
					float_t rij = sqrt(rij_2);

					if (rij == 0.)
						continue;

					if ((rij <= hi) && (is_tool_particle_i == is_tool_particle_j))
					{
						cfmi.x += mass * xij;
						cfmi.y += mass * yij;
						cfmi.z += mass * zij;

						cfmi_count++;
					}
				}
			}
		}
	}

	// Boundaries
	if (/* (cfmi_count <= 120) &&  */ (cfmi_count > 0))
	{
		cfmi.x = cfmi.x / (cfmi_count * mass);
		cfmi.y = cfmi.y / (cfmi_count * mass);
		cfmi.z = cfmi.z / (cfmi_count * mass);

		cfm[pidx].x = cfmi.x;
		cfm[pidx].y = cfmi.y;
		cfm[pidx].z = cfmi.z;
	}
	else
	{
		cfmi.x = 0.0;
		cfmi.y = 0.0;
		cfmi.z = 0.0;

		cfm[pidx].x = cfmi.x;
		cfm[pidx].y = cfmi.y;
		cfm[pidx].z = cfmi.z;
	}
}

__global__ void do_ni_smoothing_dividing(float3_t *__restrict__ cfm, int *__restrict__ surface, const float_t *__restrict__ blanked,
										 unsigned int N, const float_t *__restrict__ in_tool, float3_t *sum_buff, int *count_buff, float_t *fixed, int step)
{

	int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	int th = threadIdx.x;

	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;
	if ((cfm[pidx].x == 0) && (cfm[pidx].y == 0) && (cfm[pidx].z == 0))
		return;

	if (step > 1 && in_tool[pidx] != 1)
		return;

	int count_buff_pidx = count_buff[pidx];

	if (count_buff_pidx != 0)
	{
		cfm[pidx].x = sum_buff[pidx].x / count_buff_pidx;
		cfm[pidx].y = sum_buff[pidx].y / count_buff_pidx;
		cfm[pidx].z = sum_buff[pidx].z / count_buff_pidx;
	}
}

__global__ void do_free_surface_detection(float3_t *__restrict__ cfm, int *__restrict__ surface, const float_t *__restrict__ blanked,
										  unsigned int N, const float_t *__restrict__ in_tool, float_t *fixed, int step)
{

	int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	int th = threadIdx.x;

	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;
	if ((cfm[pidx].x == 0) && (cfm[pidx].y == 0) && (cfm[pidx].z == 0))
		return;
	if (step > 1 && in_tool[pidx] != 1)
		return;

	float_t in_tooli = in_tool[pidx];

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	// load particle data at pidx
	float4_t pi = texfetch4(pos_tex, pidx);
	float_t hi = texfetch1(h_tex, pidx);

	hi = (hi / 1.7) * 1.2;

	float4_t T;
	T.x = pi.x + cfm[pidx].x * hi;
	T.y = pi.y + cfm[pidx].y * hi;
	T.z = pi.z + cfm[pidx].z * hi;

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	int gd = geometry.gd + 1; // for 3h smoothing distance
	int low_i = gi - gd < 0 ? 0 : gi - gd;
	int low_j = gj - gd < 0 ? 0 : gj - gd;
	int low_k = gk - gd < 0 ? 0 : gk - gd;

	int shift_high = gd + 1;
	int high_i = gi + shift_high > nx ? nx : gi + shift_high;
	int high_j = gj + shift_high > ny ? ny : gj + shift_high;
	int high_k = gk + shift_high > nz ? nz : gk + shift_high;

	// iterate over neighboring boxes
	for (int kk = low_k; kk < high_k; kk++)
	{
		for (int jj = low_j; jj < high_j; jj++)
		{
			for (int ii = low_i; ii < high_i; ii++)
			{

				int idx;
				hash(ii, jj, kk, idx);

				// iterate over particles contained in a neighboring box
				int c_start = tex1Dfetch(cells_start_tex, idx);
				int c_end = tex1Dfetch(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					float4_t pj = texfetch4(pos_tex, iter);
					float_t in_toolj = in_tool[iter];
					if (in_tooli != in_toolj)
						continue;

					float_t xji = pj.x - pi.x;
					float_t yji = pj.y - pi.y;
					float_t zji = pj.z - pi.z;

					float_t xjT = pj.x - T.x;
					float_t yjT = pj.y - T.y;
					float_t zjT = pj.z - T.z;

					float_t r = sqrtf(xji * xji + yji * yji + zji * zji);
					float_t rt = sqrtf(xjT * xjT + yjT * yjT + zjT * zjT);

					if ((r >= hi * sqrtf(2.)) && (rt < hi))
					{

						cfm[pidx].x = 0;
						cfm[pidx].y = 0;
						cfm[pidx].z = 0;
						surface[pidx] = 0;
						continue;
					}

					float_t ndotx = cfm[pidx].x * xji + cfm[pidx].y * yji + cfm[pidx].z * zji;
					// float_t ndotx = -cfm[pidx].x * xji - cfm[pidx].y * yji - cfm[pidx].z * zji;
					float_t cone = acosf(ndotx / r);

					if ((r <= hi * sqrtf(2.)) && cone < ((40. * M_PI) / 180.))
					{
						cfm[pidx].x = 0;
						cfm[pidx].y = 0;
						cfm[pidx].z = 0;
						surface[pidx] = 0;
					}
				}
			}
		}
	}
}

__global__ void do_interactions_rod_force_Songwon(particle_gpu particles, float_t dt, float3 *forces)
{

	// exhaustive contact algorithm (n^2)

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int N = particles.N;
	if (pidx >= N)
		return;
	if (particles.blanked[pidx] == 1.)
		return;

	int part = particles.tool_particle[pidx];
	if (part == 0.)
		return; // die return
	if (part == 2)
		return; // die return

	if (particles.surface[pidx] == 0)
		return;

	vec3_t fN(0., 0., 0.);

	// load particle data at pidx
	float4_t vi = texfetch4(vel_tex, pidx);
	float_t rhoi = texfetch1(rho_tex, pidx);
	float_t Ci = sqrt(physics.K / rhoi);

	float_t hi = texfetch1(h_tex, pidx);
	float4_t pi = texfetch4(pos_tex, pidx);

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	// find neighboring boxes (take care not to iterate beyond size of cell lists structure)
	int gd = geometry.gd;

	int low_i = gi - gd < 0 ? 0 : gi - gd;
	int low_j = gj - gd < 0 ? 0 : gj - gd;
	int low_k = gk - gd < 0 ? 0 : gk - gd;

	int shift_high = gd + 1;
	int high_i = gi + shift_high > nx ? nx : gi + shift_high;
	int high_j = gj + shift_high > ny ? ny : gj + shift_high;
	int high_k = gk + shift_high > nz ? nz : gk + shift_high;

	// float_t in_tooli= in_tool[pidx];

	float_t di = rod.dz;

	float_t maxr = di / 2.; // max(radius[i], radius[j]);
	float_t rav = di;
	// float_t rsqc = maxr * maxr + rav * rav; // 4.0*maxr*maxr;

	float_t ni[3];
	ni[0] = particles.cfm[pidx].x;
	ni[1] = particles.cfm[pidx].y;
	ni[2] = particles.cfm[pidx].z;

	float_t SMALL = 1.0e-15;

	float_t slave_mass = physics.mass;

	float_t nav[3];
	nav[0] = 0;
	nav[1] = 0;
	nav[2] = 0;

	// iterate over neighboring boxes
	for (int kk = low_k; kk < high_k; kk++)
	{
		for (int jj = low_j; jj < high_j; jj++)
		{
			for (int ii = low_i; ii < high_i; ii++)
			{
				int idx;
				hash(ii, jj, kk, idx);

				// iterate over particles contained in a neighboring box
				int c_start = tex1Dfetch(cells_start_tex, idx);
				int c_end = tex1Dfetch(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					if (particles.blanked[iter] == 1. || particles.surface[iter] == 0 || particles.tool_particle[iter] == 1)
					{ // here you need to add surface 0
						continue;
					}

					// load vars at neighbor particle
					float4_t pj = texfetch4(pos_tex, iter);
					float4_t vj = texfetch4(vel_tex, iter);

					float_t xij = pi.x - pj.x;
					float_t yij = pi.y - pj.y;
					float_t zij = pi.z - pj.z;

					float_t vxij = vi.x - vj.x;
					float_t vyij = vi.y - vj.y;
					float_t vzij = vi.z - vj.z;

					/* float_t rsqc;
					rsqc =       maxr * maxr + rav * rav;   */
					float_t rsq = sqrt(xij * xij + yij * yij + zij * zij);

					if (rsq < rav)
					{
						// printf("xxxxxxxxxxxxxxXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxxxxx, %.20lf\n", rsq);

						float_t nj[3];
						nj[0] = particles.cfm[iter].x;
						nj[1] = particles.cfm[iter].y;
						nj[2] = particles.cfm[iter].z;

						float_t ab = -ni[0] * nj[0] - ni[1] * nj[1] - ni[2] * nj[2];
						float_t a_abs = sqrt(ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]);
						float_t b_abs = sqrt(nj[0] * nj[0] + nj[1] * nj[1] + nj[2] * nj[2]);
						float_t cosTheta = ab / (a_abs * b_abs);
						float_t Theta_radian = acos(cosTheta);
						float_t Theta_degree = (Theta_radian * 180.) / 3.14159265359;

						float_t nij[3];
						float_t absnij;
						nij[0] = ni[0] - nj[0];
						nij[1] = ni[1] - nj[1];
						nij[2] = ni[2] - nj[2];
						absnij = sqrt(nij[0] * nij[0] + nij[1] * nij[1] + nij[2] * nij[2]);

						int case_number = 0;
						nav[0] = -nj[0];
						nav[1] = -nj[1];
						nav[2] = -nj[2];
						/*if (Theta_degree < 70)
						{

							float_t test1 = xij * ni[0] + yij * ni[1] + zij * ni[2];
							float_t test2 =-xij * nj[0] - yij * nj[1] + zij * nj[2];
							float_t test3 = xij * nij[0] + yij * nij[1] +zij * nij[2];
							test3 /= (absnij + SMALL);

							if (test1 > max(test2, test3))
							{
								nav[0] = ni[0];
								nav[1] = ni[1];
								nav[2] = ni[2];
								case_number = 1;
							}
							else if (test2 > max(test1, test3))
							{
								nav[0] = -nj[0];
								nav[1] = -nj[1];
								nav[2] = -nj[2];
								case_number = 2;

							}
							else
							{
								nav[0] = nij[0] / (absnij+SMALL);
								nav[1] = nij[1] / (absnij+SMALL);
								nav[2] = nij[2] / (absnij+SMALL);
								case_number = 3;

							}
						}
						else
						{
							nav[0] = nij[0] / (absnij + SMALL);
							nav[1] = nij[1] / (absnij + SMALL);
							nav[2] = nij[2] / (absnij + SMALL);
							case_number = 4;

						} */

						float_t rijDotnav = xij * nav[0] + yij * nav[1] + zij * nav[2];
						float_t pnav = rav - fabs(rijDotnav);
						float_t pnavDot = vxij * nav[0] + vyij * nav[1] + vzij * nav[2];

						if (pnav > 0)
						{

							/*************************************** Dr. Kirk method: https://doi.org/10.1016/j.matdes.2021.109514 *****************/
							//  float_t PFAC = 0.05; this the gloden one
							//	float_t PFAC = 0.1; was not good with 0.5 PF
							// 0.075 was working fine with partioning factor 0.5, 0.1 was too much with the new joco
							// 0.005 is the smallest value that can be used an prevent particles penetration
							// 0.005 was too small was not able to generate hate enough // 0.05 suden plast after 40 frames//0.2 was too match// same for 0.15 too much
							// 0.1 was too much too same 0.075

							/* float_t PFAC = 0.01;
							float_t DFAC = PFAC;

							float_t pd = pnav;
							float_t kij = ((slave_mass*slave_mass)/(slave_mass+slave_mass))*(PFAC/(dt*dt));
							float_t dpN = pnavDot;
							float_t cij = DFAC *(slave_mass+slave_mass) *sqrt((kij*(slave_mass+slave_mass))/(slave_mass*slave_mass));

							vec3_t fN_each(0.,0.,0.);
							fN_each.x = -1.*(kij*pd + cij*dpN)*nav[0] ;
							fN_each.y = -1.*(kij*pd + cij*dpN)*nav[1] ;
							fN_each.z = -1.*(kij*pd + cij*dpN)*nav[2] ;

							fN.x 	+=  fN_each.x ;
							fN.y 	+=  fN_each.y ;
							fN.z 	+=  fN_each.z ;  */

							/*************************************** Prof. Songwon method: doi:10.1016/j.ijimpeng.2007.04.009 *********************************/
							float_t pd = pnav;
							float_t dpN = pnavDot;
							float_t PFAC = 0.01;
							// printf("Intersection between part: %f and %f with value %.20lf and limit %.20lf, and value %.20lf, theta:%lf, case: %d, id: %d, %d, direction: %f, %f, %f\n", particles.tool_particle[pidx], particles.tool_particle[iter], rsq,rav, pd,Theta_degree,case_number, particles.unique_idx[pidx], particles.unique_idx[iter],nav[0], nav[1], nav[2] );

							float_t rhoj = texfetch1(rho_tex, iter);
							float_t Cj = sqrt(physics.K / rhoj);
							float_t Ei_init = physics.E;
							float_t Ej_init = physics.E;
							float_t d0 = di;

							float_t Ei = Ei_init;
							float_t Ej = Ej_init;
							/*
													// Adding the effect of temp in decreasing the E for rod particle
													 float_t k_Ei = Ei_init /(johnson_cook_rod.Tmelt - johnson_cook_rod.Tref);
													float_t Ei   = Ei_init - k_Ei *(particles.T[pidx] -  johnson_cook_rod.Tref);
													if(particles.T[pidx] > johnson_cook_rod.Tmelt){ Ei = 0;}

													// Adding the effect of temp in decreasing the E for the substrate particle
													float_t k_Ej = Ej_init /(johnson_cook_rod.Tmelt - johnson_cook_rod.Tref);
													float_t Ej   = Ej_init - k_Ej *(particles.T[iter] -  johnson_cook_rod.Tref);
													if(particles.T[iter] > johnson_cook_rod.Tmelt){ Ej = 0;} ////////////////////////////////////////////////////////
							*/

							// float_t alpha_1 = ((rhoj * Cj) / (rhoj *Cj + rhoi * Ci)) * (rhoi * Ci);
							float_t alpha_1 = rhoi * Ci;
							float_t alpha_2 = Ei * (1. / d0);

							// printf("Ei=%lf, Ej=%lf, T_i=%lf, T_iter=%lf, Tmelt=%lf \n", Ei, Ej, particles.T[pidx], particles.T[iter], johnson_cook_rod.Tmelt);

							float_t mass_j = physics.mass;
							float_t vol_j = mass_j / rhoj;

							float_t mass_i = physics.mass;
							float_t vol_i = mass_i / rhoi;

							float_t area_i = 2 * 3.14159265359 * (di / 2.) * hi;

							float4_t ww = cubic_spline(pi, pj, hi);
							float_t w = ww.x;
							float_t w_x = ww.y;
							float_t w_y = ww.z;
							float_t w_z = ww.w;

							float_t h1 = 1. / (hi);
							float_t q = rod.dz * h1;
							float_t fac = (M_1_PI)*h1 * h1 * h1;
							float_t wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));

							vec3_t fN_each(0., 0., 0.);
							float_t df = 1.0;
							fN_each.x = -1. * df * (alpha_1 * dpN + alpha_2 * pd) * nav[0] * area_i * vol_j * (w);
							fN_each.y = -1. * df * (alpha_1 * dpN + alpha_2 * pd) * nav[1] * area_i * vol_j * (w);
							fN_each.z = -1. * df * (alpha_1 * dpN + alpha_2 * pd) * nav[2] * area_i * vol_j * (w);

							/* if(isnaninf(fN_each.x) || isnaninf(fN_each.y) || isnaninf(fN_each.z))
							printf("%lf, %lf, %lf \n",nav[0], nav[1],nav[2]); */

							fN.x += fN_each.x;
							fN.y += fN_each.y;
							fN.z += fN_each.z;
							/************************************************** Friction force ******************************************************/

							// Calculating the friction force
							vec3_t fT(0., 0., 0.);
							vec3_t vm = vec3_t(vj.x, vj.y, vj.z); // velocity of substrate particle
							vec3_t vp = vec3_t(vi.x, vi.y, vi.z); // velocity of rod particle

							vec3_t nis = vec3_t(nav[0], nav[1], nav[2]);
							{
								// initialize the friction force by zerp
								float3_t fric;
								fric.x = 0.;
								fric.y = 0.;
								fric.z = 0.;
								/*
								fric.x =  particles.ft[pidx].x;
								fric.y =  particles.ft[pidx].y;
								fric.z =  particles.ft[pidx].z;
								*/

								vec3_t v = vp - vm;
								vec3_t vr = v - v * nis;
								vec3_t fricold(fric.x, fric.y, fric.z);

								// Adding the effect of temp in decreasing the fric surface coefficient
								/* float_t friction_mu_init = 0.35;
								//float_t friction_mu_init = 0.57;
								float_t k_frition = friction_mu_init /(johnson_cook_rod.Tmelt - johnson_cook_rod.Tref);
								float_t friction_mu = friction_mu_init - k_frition *(particles.T[pidx] -  johnson_cook_rod.Tref);

								// If the temp is greater than melting temp, fric coeff =0
								if(particles.T[pidx] > johnson_cook_rod.Tmelt  || particles.T[iter] > johnson_cook_rod.Tmelt ) friction_mu = 0; */

								float_t friction_mu = 0.57;

								float_t contact_alpha = alpha_2; /////////////////////////////////////////////////////////////////////////////////////////////////////

								vec3_t kdeltae = contact_alpha * slave_mass * vr / dt;

								float_t fy = friction_mu * glm::length(fN_each);

								vec3_t fstar = fricold - kdeltae;

								if (glm::length(fstar) != 0)
									fT = fy * fstar / glm::length(fstar);
								// printf("%lf, %lf, %lf, %lf \n",nav[0], nav[1],nav[2],vp.z);

								/* if (glm::length(fstar) > fy) {

									fT  = fy*fstar/glm::length(fstar);

								} else {
									fT = fstar;

								}  */

								// printf("%lf\n",alpha_2);
							}

							particles.ft[pidx].x += fT.x;
							particles.ft[pidx].y += fT.y;
							particles.ft[pidx].z += fT.z;

							/////////////////////////////////////// heating ////////////////////////////////////////////////////////////////////////
							float_t f_fric_mag = sqrtf(fT.x * fT.x + fT.y * fT.y + fT.z * fT.z);

							if (f_fric_mag != 0.)
							{

								float3_t normal = make_float3_t(nav[0], nav[1], nav[2]);
								float3_t v_diff = make_float3_t(vi.x - vj.x, vi.y - vj.y, vi.z - vj.z);

								float_t v_diff_dot_normal = v_diff.x * normal.x + v_diff.y * normal.y + v_diff.z * normal.z;
								float3_t v_relative = make_float3_t(v_diff.x - v_diff_dot_normal, v_diff.y - v_diff_dot_normal, v_diff.z - v_diff_dot_normal);

								float_t v_rel_mag = sqrtf(v_relative.x * v_relative.x + v_relative.y * v_relative.y + v_relative.z * v_relative.z);

								particles.T[pidx] += 1.0 * trml_sub.eta * dt * f_fric_mag * v_rel_mag / (particles.cp[pidx] * physics.mass); // billet

								// printf("%lf \n", 1.0*trml_sub.eta*dt*f_fric_mag*v_rel_mag/(particles.cp[pidx]*physics.mass));

								// float_t sub_T= 0.3*trml_rod.eta*dt*f_fric_mag*v_rel_mag/(particles.cp[iter]*physics.mass); // die
								// particles.T[iter] += sub_T; //substrate

								// atomicAdd(&(particles.T[iter]), sub_T);
							}

							/*************************************** Iter paerticle*****************************************************************/

							/* particles.fc[iter].x +=  -1.*fN_each.x;
							particles.fc[iter].y +=  -1.*fN_each.y;
							particles.fc[iter].z +=  -1.*fN_each.z;

							particles.ft[iter].x -= fT.x;
							particles.ft[iter].y -=	fT.y;
							particles.ft[iter].z -=	fT.z;  */

							/* atomicAdd(&( particles.fc[iter].x), -1.*fN_each.x);
							atomicAdd(&( particles.fc[iter].y), -1.*fN_each.y);
							atomicAdd(&( particles.fc[iter].z), -1.*fN_each.z);

							atomicAdd(&( particles.ft[iter].x), -1.*fT.x);
							atomicAdd(&( particles.ft[iter].y), -1.*fT.y);
							atomicAdd(&( particles.ft[iter].z), -1.*fT.z);
							 */
						}
					}
				}
			}
		}
	}

	particles.fc[pidx].x = fN.x;
	particles.fc[pidx].y = fN.y;
	particles.fc[pidx].z = fN.z;

	if (true)
	{
		float fx = float(fN.x + particles.ft[pidx].x);
		float fy = float(fN.y + particles.ft[pidx].y);
		float fz = float(fN.z + particles.ft[pidx].z);

		fx = (isnan(fx)) ? 0. : fx;
		fy = (isnan(fy)) ? 0. : fy;
		fz = (isnan(fz)) ? 0. : fz;

		atomicAdd(&(forces[0].x), fx);
		atomicAdd(&(forces[0].y), fy);
		atomicAdd(&(forces[0].z), fz);
	}
}
__global__ void do_mem_set_zero(float3_t *__restrict__ cfm, int *__restrict__ surface, const float_t *__restrict__ blanked,
								const float_t *__restrict__ in_tool, int N, int s)
{

	// exhaustive contact algorithm (n^2)

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;
	if (s > 1 && in_tool[pidx] != 1.)
		return;

	surface[pidx] = 0;
	cfm[pidx].x = 0;
	cfm[pidx].y = 0;
	cfm[pidx].z = 0;
}

__global__ void do_mem_set_zero_buff(float3_t *__restrict__ sum_buff, int *__restrict__ count_buff, const float_t *__restrict__ blanked,
									 const float_t *__restrict__ in_tool, int N, int s)
{

	// exhaustive contact algorithm (n^2)

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;
	if (s > 1 && in_tool[pidx] != 1.)
		return;

	count_buff[pidx] = 0;
	sum_buff[pidx].x = 0;
	sum_buff[pidx].y = 0;
	sum_buff[pidx].z = 0;
}

__global__ void do_mem_copy_buff(float3_t *__restrict__ sum_buff, float3_t *__restrict__ cfm, const float_t *__restrict__ blanked,
								 const float_t *__restrict__ in_tool, int N, int s)
{

	// exhaustive contact algorithm (n^2)

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;
	if (s > 1 && in_tool[pidx] != 1.)
		return;

	sum_buff[pidx].x = cfm[pidx].x;
	sum_buff[pidx].y = cfm[pidx].y;
	sum_buff[pidx].z = cfm[pidx].z;
}

__device__ void save_forces(float3 *forces, vec3_t fN, vec3_t fT, float_t h_p, mat3x3_t h_S, float3_t vel_t)
{
	float_t mass = physics.mass;

	float_t fx = float(fN.x + fT.x);
	float_t fy = float(fN.y + fT.y);
	float_t fz = float(fN.z + fT.z);

	/* float_t fx = vel_t.x * mass;
	float_t fy = vel_t.y * mass;
	float_t fz = vel_t.z * mass; */

	fx = (isnan(fx)) ? 0. : fx;
	fy = (isnan(fy)) ? 0. : fy;
	fz = (isnan(fz)) ? 0. : fz;

	atomicAdd(&(forces[0].x), fx);
	atomicAdd(&(forces[0].y), fy);
	atomicAdd(&(forces[0].z), fz);
}

__device__ void kirk_contact_force(vec3_t &fN_each, float_t pd, vec3_t vij, vec3_t nav, float_t dt, float_t p_temp, bool extruding)
{
	float_t DFAC = 0.2; // 0.2for damping

	float_t PFAC_init = 0.1; // 0.001 was Good for both heating and extrusion phases

	/* if (extruding)
		PFAC_init = 0.001;
	else
		PFAC_init = 0.001; */

	float_t PFAC = PFAC_init;

	/* float_t k_PFAC = PFAC_init / (johnson_cook_rod.Tmelt - johnson_cook_rod.Tref);
	float_t PFAC = PFAC_init - k_PFAC * (p_temp - johnson_cook_rod.Tref);

	if (p_temp > johnson_cook_rod.Tmelt)
		PFAC = 0.0; */

	float_t slave_mass = physics.mass;
	float_t dpN = vij.x * nav.x + vij.y * nav.y + vij.z * nav.z;

	float_t kij = ((slave_mass * slave_mass) / (slave_mass + slave_mass)) * (PFAC / (dt * dt));
	float_t cij = DFAC * (slave_mass + slave_mass) * sqrt((kij * (slave_mass + slave_mass)) / (slave_mass * slave_mass));

	fN_each.x += -1. * (kij * pd + cij * dpN) * nav.x;
	fN_each.y += -1. * (kij * pd + cij * dpN) * nav.y;
	fN_each.z += -1. * (kij * pd + cij * dpN) * nav.z;
}

__device__ void kirk_contact_force_above_dive(vec3_t &fN_each, float_t pd, vec3_t vij, vec3_t nav, float_t dt, float_t p_temp, bool extruding)
{
	float_t DFAC = 0.2; // 0.2for damping

	float_t PFAC_init = 0.001; // 0.001 was Good

	float_t PFAC = PFAC_init;
	float_t slave_mass = physics.mass;
	float_t dpN = vij.x * nav.x + vij.y * nav.y + vij.z * nav.z;

	float_t kij = ((slave_mass * slave_mass) / (slave_mass + slave_mass)) * (PFAC / (dt * dt));
	float_t cij = DFAC * (slave_mass + slave_mass) * sqrt((kij * (slave_mass + slave_mass)) / (slave_mass * slave_mass));

	fN_each.x += -1. * (kij * pd + cij * dpN) * nav.x;
	fN_each.y += -1. * (kij * pd + cij * dpN) * nav.y;
	fN_each.z += -1. * (kij * pd + cij * dpN) * nav.z;
}

__device__ double sigma_yield_interaction(joco_constants jc, double eps_pl, double eps_pl_dot, double t)
{
	double theta = (t - jc.Tref) / (jc.Tmelt - jc.Tref);

	double Term_A = jc.A + jc.B * pow(eps_pl, jc.n);
	// double Term_A = jc.A +(678.e6) * pow(eps_pl, 0.71);
	double Term_B = 1.0;

	double eps_dot = eps_pl_dot / jc.eps_dot_ref;

	if (eps_dot > 1.0)
	{
		Term_B = 1.0 + jc.C * log_t(eps_dot);
	}
	else
	{
		Term_B = pow((1.0 + eps_dot), jc.C);
	}

	double Term_C = 1.0 - pow(theta, jc.m);
	return Term_A * Term_B * Term_C;
}

__device__ float_t calculate_heat_generation(vec3_t fT_t, vec3_t vr, float_t eta, float_t dt, float_t cp, float_t mass)
{
	float_t f_fric_mag = sqrt(fT_t.x * fT_t.x + fT_t.y * fT_t.y + fT_t.z * fT_t.z);
	float_t v_rel_mag = sqrt(vr.x * vr.x + vr.y * vr.y + vr.z * vr.z);
	float_t T = (eta * dt * f_fric_mag * v_rel_mag) / (cp * mass);
	return T;
}

__device__ void calculate_contact_force(bool &is_sticking, vec3_t &fN, vec3_t &fT, vec3_t &vr, vec3_t &fricold, float_t gN, vec3_t normal, vec3_t vs, float_t dt, float_t p_temp, bool extruding, float_t contact_alpha, float_t slave_mass, float_t friction_mu, float_t ffl, float_t cp, particle_gpu &particles, unsigned int pidx, float_t &T)
{
	vec3_t kdeltae = contact_alpha * slave_mass * vr / dt;
	float_t fy = friction_mu * glm::length(fN);
	vec3_t fstar = fricold - kdeltae;
	float_t fstar_mag = glm::length(fstar);
	is_sticking = false;

	if (fstar_mag != 0)
	{
		if (fy > ffl)
		{
			fy = ffl;
			is_sticking = true;
		}

		vec3_t fT_t = fy * (fstar / fstar_mag);
		fT += fT_t;
		if (!is_sticking)
			T += calculate_heat_generation(fT_t, vr, trml_sub.eta, dt, cp, physics.mass);
	}
}

__device__ void handle_contact(bool &is_sticking, vec3_t &fN, vec3_t &fT, vec3_t &vr, vec3_t &fricold, float_t gN, vec3_t normal, vec3_t vs, float_t dt, float_t p_temp, bool extruding, float_t contact_alpha, float_t slave_mass, float_t friction_mu, float_t ffl, float_t cp, particle_gpu &particles, unsigned int pidx, float_t &T, float3 *forces)
{
	calculate_contact_force(is_sticking, fN, fT, vr, fricold, gN, normal, vs, dt, p_temp, extruding, contact_alpha, slave_mass, friction_mu, ffl, cp, particles, pidx, T);
	save_forces(forces, fN, fT, particles.p[pidx], particles.S[pidx], particles.vel_t[pidx]);
}

__global__ void interactions_calculate_force_die_using_kirk_method(particle_gpu particles, float_t dt, float3 *forces, float_t top_die_surface, float_t v_die, float_t gWz)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int N = particles.N;
	if (pidx >= N || particles.blanked[pidx] == 1. || particles.tool_particle[pidx] != 1.)
		return;

	float_t cp = particles.cp[pidx];
	float_t p_temp = particles.T[pidx];
	float_t eps_pl = particles.eps_pl[pidx];
	float_t eps_pl_dot = particles.eps_pl_dot[pidx];
	float_t sigma_Y = sigma_yield_interaction(johnson_cook_sub, eps_pl, eps_pl_dot, p_temp);

	float_t ffl = (sigma_Y / sqrt(3.)) * rod.dz * rod.dz;
	float_t top_top_die_surface = top_die_surface + 25.0;
	float_t inner_container_radius = rod.radius;
	float_t die_radius = rod.radius - rod.dz / 4.0;
	// float_t die_radius = rod.radius - rod.dz / 2.0;
	float_t orifice_radius = rod.orifice_radius - rod.dz;
	float_t after_orifice_radius = orifice_radius + 1.0;

	float4_t pi = particles.pos[pidx];
	float r2 = pi.x * pi.x + pi.y * pi.y;
	float sqrt_r2 = sqrt(r2);
	float mag = sqrt_r2;

	vec3_t normal(0., 0., 0.);
	float_t gN = 0;
	float_t contact_alpha = 0.05;

	// float_t mu = -0.0006593407 * p_temp + 0.4145055;
	float_t friction_mu = (p_temp > johnson_cook_rod.Tmelt) ? 0.0 : 0.6;

	bool extruding = abs(gWz) < 9.5 * rod.Vsf;

	float_t dt2 = dt * dt;
	float_t slave_mass = physics.mass;
	vec3_t vs(particles.vel[pidx].x, particles.vel[pidx].y, particles.vel[pidx].z);
	vec3_t fricold(particles.ft[pidx].x, particles.ft[pidx].y, particles.ft[pidx].z);

	vec3_t fN(0., 0., 0.);
	vec3_t fT(0., 0., 0.);
	vec3_t vr(0., 0., 0.);

	if (pi.z > top_top_die_surface && r2 > 30.25)
	{
		// this is A flash
		particles.fixed[pidx] = 7;
	}

	if (pi.z < top_die_surface)
	{
		// this is A billet under the die
		particles.pos[pidx].w = 0;
	}

	if ((r2 < orifice_radius * orifice_radius) && (pi.z > top_die_surface))
	{
		// This is extruded wire
		particles.pos[pidx].w = 1;
		return;
	}

	if ((r2 > die_radius * die_radius) && (pi.z > top_die_surface))
	{
		// This is a flash
		particles.pos[pidx].w = 2;
		// now interact with the container
		if (r2 > inner_container_radius * inner_container_radius)
		{
			gN = sqrt_r2 - inner_container_radius;
			normal.x = pi.x / mag;
			normal.y = pi.y / mag;

			vec3_t v = vs;
			vr = v - v * normal;

			kirk_contact_force(fN, gN, v, normal, dt, p_temp, extruding);
			if (pi.z < top_top_die_surface)
			{
				bool is_sticking = false;
				handle_contact(is_sticking, fN, fT, vr, fricold, gN, normal, vs, dt, p_temp, extruding, contact_alpha, slave_mass, friction_mu, ffl, cp, particles, pidx, particles.T[pidx], forces);
				if (is_sticking)
				{
					particles.vel[pidx].x = 0.;
					particles.vel[pidx].y = 0.;
					particles.vel[pidx].z = 0.;
				}
			}
		}

		particles.fc[pidx] = make_float3_t(fN.x, fN.y, fN.z);
		particles.ft[pidx] = make_float3_t(fT.x, fT.y, fT.z);
		particles.n[pidx] = make_float3_t(normal.x, normal.y, normal.z);
		return;
	}

	float_t p_state = particles.pos[pidx].w;

	if ((p_state == 2) && (pi.z > top_die_surface))
	{
		// This is a flash inertacting with the die
		if (r2 < die_radius * die_radius)
		{
			gN = abs(sqrt_r2 - die_radius);
			normal.x = -pi.x / mag;
			normal.y = -pi.y / mag;

			float_t x = (die_radius / mag) * pi.x;
			float_t y = (die_radius / mag) * pi.y;

			vec3_t w(0.0, 0.0, gWz);
			vec3_t r(x, y, 0.0);
			vec3_t vm = glm::cross(w, r);
			vm.z = v_die;

			vec3_t v = vs - vm;
			vr = v - v * normal;

			kirk_contact_force(fN, gN, v, normal, dt, p_temp, extruding);

			if (pi.z < top_top_die_surface)
			{
				vec3_t kdeltae = contact_alpha * slave_mass * vr / dt;
				float_t fy = friction_mu * glm::length(fN);
				vec3_t fstar = fricold - kdeltae;
				float_t fstar_mag = glm::length(fstar);

				if (fstar_mag != 0)
				{
					if (fy > ffl)
						fy = ffl;

					vec3_t fT_t = fy * (fstar / fstar_mag);
					fT += fT_t;
					particles.T[pidx] += calculate_heat_generation(fT_t, vr, trml_sub.eta, dt, cp, physics.mass);
				}
			}
			save_forces(forces, fN, fT, particles.p[pidx], particles.S[pidx], particles.vel_t[pidx]);
		}

		particles.fc[pidx] = make_float3_t(fN.x, fN.y, fN.z);
		particles.ft[pidx] = make_float3_t(fT.x, fT.y, fT.z);
		particles.n[pidx] = make_float3_t(normal.x, normal.y, normal.z);

		return;
	}

	if (p_state == 1)
	{
		// This is a wire interacting with the orifice
		if (r2 > orifice_radius * orifice_radius && pi.z <= top_die_surface + 8.0)
		{
			gN = abs(sqrt_r2 - orifice_radius);
			normal.x = pi.x / mag;
			normal.y = pi.y / mag;

			float_t x = (orifice_radius / mag) * pi.x;
			float_t y = (orifice_radius / mag) * pi.y;

			vec3_t w(0.0, 0.0, gWz);
			vec3_t r(x, y, 0.0);
			vec3_t vm = glm::cross(w, r);
			vm.z = v_die;

			vec3_t v = vs - vm;
			vr = v - v * normal;

			kirk_contact_force(fN, gN, v, normal, dt, p_temp, extruding);

			bool is_sticking = false;
			handle_contact(is_sticking, fN, fT, vr, fricold, gN, normal, vs, dt, p_temp, extruding, contact_alpha, slave_mass, friction_mu, ffl, cp, particles, pidx, particles.T[pidx], forces);
			if (is_sticking)
			{

				particles.vel[pidx].x = vm.x;
				particles.vel[pidx].y = vm.y;
				particles.vel[pidx].z = vm.z;
			}
		}
		// This is a wire interacting with the after orifice outer surface
		if (r2 > after_orifice_radius * after_orifice_radius && pi.z > top_die_surface + 8.0)
		{
			gN = abs(sqrt_r2 - after_orifice_radius);
			normal.x = pi.x / mag;
			normal.y = pi.y / mag;

			float_t x = (after_orifice_radius / mag) * pi.x;
			float_t y = (after_orifice_radius / mag) * pi.y;

			vec3_t w(0.0, 0.0, gWz);
			vec3_t r(x, y, 0.0);
			vec3_t vm = glm::cross(w, r);
			vm.z = v_die;

			vec3_t v = vs - vm;
			vr = v - v * normal;

			kirk_contact_force(fN, gN, v, normal, dt, p_temp, extruding);
		}
	}
	else
	{
		// this is a billet interacting with the die bottom surface
		if (pi.z > top_die_surface && r2 < die_radius * die_radius) ///////////////////////////////////////////////////////////////////////////////////
		{
			gN = abs(abs(top_die_surface) - abs(pi.z));
			normal.z = 1.0;

			vec3_t w(0.0, 0.0, gWz);
			vec3_t r(pi.x, pi.y, 0.0);
			vec3_t vm = glm::cross(w, r);
			vm.z = v_die;

			vec3_t v = vs - vm;
			vr = v - v * normal;
			kirk_contact_force(fN, gN, v, normal, dt, p_temp, extruding);

			bool is_sticking = false;
			handle_contact(is_sticking, fN, fT, vr, fricold, gN, normal, vs, dt, p_temp, extruding, contact_alpha, slave_mass, friction_mu, ffl, cp, particles, pidx, particles.T[pidx], forces);
			if (is_sticking)
			{
				particles.vel[pidx].x = vm.x;
				particles.vel[pidx].y = vm.y;
				particles.vel[pidx].z = vm.z;
			}
		}
		// this is a billet interacting with the container bottom surface
		if (pi.z < rod.lz_tool)
		{
			gN = abs(rod.lz_tool - abs(pi.z));
			normal.z = -1.0;

			vec3_t v = vs;
			vr = v - v * normal;
			kirk_contact_force(fN, gN, v, normal, dt, p_temp, extruding);

			bool is_sticking = false;
			handle_contact(is_sticking, fN, fT, vr, fricold, gN, normal, vs, dt, p_temp, extruding, contact_alpha, slave_mass, friction_mu, ffl, cp, particles, pidx, particles.T[pidx], forces);
			if (is_sticking)
			{

				particles.vel[pidx].x = 0;
				particles.vel[pidx].y = 0;
				particles.vel[pidx].z = 0;
			}
		}
		// this is a billet interacting with the container side surface
		if (r2 > inner_container_radius * inner_container_radius)
		{
			gN = sqrt_r2 - inner_container_radius;
			normal.x = pi.x / mag;
			normal.y = pi.y / mag;

			vec3_t v = vs;
			vr = v - v * normal;
			kirk_contact_force(fN, gN, v, normal, dt, p_temp, extruding);

			bool is_sticking = false;
			handle_contact(is_sticking, fN, fT, vr, fricold, gN, normal, vs, dt, p_temp, extruding, contact_alpha, slave_mass, friction_mu, ffl, cp, particles, pidx, particles.T[pidx], forces);
			if (is_sticking)
			{

				particles.vel[pidx].x = 0.;
				particles.vel[pidx].y = 0.;
				particles.vel[pidx].z = 0.;
			}
		}
	}

	particles.fc[pidx] = make_float3_t(fN.x, fN.y, fN.z);
	particles.ft[pidx] = make_float3_t(fT.x, fT.y, fT.z);
	particles.n[pidx] = make_float3_t(normal.x, normal.y, normal.z);
}

/***********************************************************************************************************************************************************/
void interactions_monaghan(particle_gpu *particles, const int *cells_start, const int *cells_end, int num_cell, int step)
{
	// run all interactions in one go
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	cudaBindTexture(0, pos_tex, p->pos, sizeof(float_t) * N * 4);
	cudaBindTexture(0, vel_tex, p->vel, sizeof(float_t) * N * 4);
	cudaBindTexture(0, h_tex, p->h, sizeof(float_t) * N);
	cudaBindTexture(0, rho_tex, p->rho, sizeof(float_t) * N);
	cudaBindTexture(0, hashes_tex, p->hash, sizeof(int) * N);
	cudaBindTexture(0, p_tex, p->p, sizeof(float_t) * N);

#ifdef Thermal_Conduction_Brookshaw
	cudaBindTexture(0, T_tex, p->T, sizeof(float_t) * N);
	cudaBindTexture(0, k_tex, p->k, sizeof(float_t) * N);
#endif

	cudaBindTexture(0, tool_particle_tex, p->tool_particle, sizeof(float_t) * N);

	cudaBindTexture(0, cells_start_tex, cells_start, sizeof(int) * num_cell);
	cudaBindTexture(0, cells_end_tex, cells_end, sizeof(int) * num_cell);

	// cudaMemset(particles->cfm, 0., sizeof(float_t)*N*3);
	// cudaMemset(particles->surface, 0., sizeof(int)*N);

	check_cuda_error("before interactions monaghan\n");
	do_mem_set_zero<<<dG, dB>>>(particles->cfm, particles->surface, particles->blanked, particles->tool_particle, particles->N, step);

	do_interactions_monaghan<<<dG, dB>>>(particles->cfm, particles->blanked, particles->S, particles->R, p->v_der, p->S_der, p->T_t,
										 p->pos_t, p->vel_t, p->N, p->tool_particle, p->fixed, global_time_dt, p->cp, p->E);

	do_calculate_init_cfm<<<dG, dB>>>(particles->cfm, particles->blanked, p->N, p->tool_particle, p->fixed, step);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(rho_tex);
	cudaUnbindTexture(hashes_tex);
	cudaUnbindTexture(p_tex);
#ifdef Thermal_Conduction_Brookshaw
	cudaUnbindTexture(T_tex);
	cudaUnbindTexture(k_tex);
#endif

	cudaUnbindTexture(tool_particle_tex);

	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);

	check_cuda_error("interactions monaghan\n");
}

void interactions_ni_smoothing(particle_gpu *particles, const int *cells_start, const int *cells_end, int num_cell, float3_t *sum_buff, int *count_buff, int step)
{
	// run all interactions in one go
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	cudaBindTexture(0, pos_tex, p->pos, sizeof(float_t) * N * 4);
	cudaBindTexture(0, h_tex, p->h, sizeof(float_t) * N);
	cudaBindTexture(0, hashes_tex, p->hash, sizeof(int) * N);
	cudaBindTexture(0, cells_start_tex, cells_start, sizeof(int) * num_cell);
	cudaBindTexture(0, cells_end_tex, cells_end, sizeof(int) * num_cell);

	check_cuda_error("before interactions smoothing\n");

	// cudaMemcpy(sum_buff, particles->cfm, sizeof(float3_t)*N, cudaMemcpyDeviceToDevice);
	// cudaMemset(count_buff, 0, sizeof(int)*N);
	// cudaMemset(sum_buff, 0, sizeof(float_t)*N*3);

	do_mem_copy_buff<<<dG, dB>>>(sum_buff, particles->cfm, particles->blanked, particles->tool_particle, particles->N, step);
	do_mem_set_zero_buff<<<dG, dB>>>(sum_buff, count_buff, particles->blanked, particles->tool_particle, particles->N, step);
	do_ni_smoothing<<<dG, dB>>>(particles->cfm, particles->surface, particles->blanked, p->N, p->tool_particle, sum_buff, count_buff, p->fixed);
	do_ni_smoothing_dividing<<<dG, dB>>>(particles->cfm, particles->surface, particles->blanked, p->N, p->tool_particle, sum_buff, count_buff, p->fixed, step);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(hashes_tex);
	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);

	check_cuda_error("interactions smoothing\n");
}

void interactions_free_surface_detection(particle_gpu *particles, const int *cells_start, const int *cells_end, int num_cell, int step)
{
	// run all interactions in one go
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	cudaBindTexture(0, pos_tex, p->pos, sizeof(float_t) * N * 4);
	cudaBindTexture(0, h_tex, p->h, sizeof(float_t) * N);
	cudaBindTexture(0, hashes_tex, p->hash, sizeof(int) * N);
	cudaBindTexture(0, cells_start_tex, cells_start, sizeof(int) * num_cell);
	cudaBindTexture(0, cells_end_tex, cells_end, sizeof(int) * num_cell);

	check_cuda_error("before interactions smoothing\n");

	do_free_surface_detection<<<dG, dB>>>(particles->cfm, particles->surface, particles->blanked, p->N, p->tool_particle, p->fixed, step);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(hashes_tex);
	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);

	check_cuda_error("do_free_surface_detection\n");
}

void interactions_rod_force_Songwon(particle_gpu *particles, const int *cells_start,
									const int *cells_end, int num_cell,
									int *joined_count, float_t *joined_mean, float3 *forces)
{
	// run all interactions in one go
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	cudaBindTexture(0, pos_tex, p->pos, sizeof(float_t) * N * 4);
	cudaBindTexture(0, rho_tex, p->rho, sizeof(float_t) * N);
	cudaBindTexture(0, h_tex, p->h, sizeof(float_t) * N);

	cudaBindTexture(0, hashes_tex, p->hash, sizeof(int) * N);
	cudaBindTexture(0, cells_start_tex, cells_start, sizeof(int) * num_cell);
	cudaBindTexture(0, cells_end_tex, cells_end, sizeof(int) * num_cell);

	cudaMemset(joined_count, 0, sizeof(int));
	cudaMemset(joined_mean, 0., sizeof(float_t));

	check_cuda_error("before interactions substrate_force_fraser\n");

	top_surface += global_die_velocity * global_time_dt;
	// printf("top_surface:%lf\n", top_surface);
	// do_interactions_rod_force_Songwon_modified_2<<<dG,dB>>>(*particles, global_time_dt, joined_count, joined_mean);
	// do_interactions_rod_force_Songwon<<<dG,dB>>>(*particles, global_time_dt, forces);
	// interactions_calculate_force_die<<<dG, dB>>>(*particles, global_time_dt, forces, top_surface);
	interactions_calculate_force_die_using_kirk_method<<<dG, dB>>>(*particles, global_time_dt, forces, top_surface, global_die_velocity, global_wz);

	// do_interactions_rod_force_Songwon_center_vector<<<dG,dB>>>(*particles, global_time_dt);
	// do_interactions_rod_force_Songwon_type_2<<<dG,dB>>>(*particles, global_time_dt);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(rho_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(hashes_tex);
	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);

	check_cuda_error("interactions substrate_force_fraser\n");
}

void interactions_heat_pse(particle_gpu *particles, const int *cells_start, const int *cells_end, int num_cell)
{
	if (!m_thermal_workpiece)
		return;

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	// cudaBindTexture(0, pos_tex,    p->pos,    sizeof(float_t)*N*2);	// 2D Version!
	cudaBindTexture(0, pos_tex, p->pos, sizeof(float_t) * N * 4);
	cudaBindTexture(0, h_tex, p->h, sizeof(float_t) * N);
	cudaBindTexture(0, rho_tex, p->rho, sizeof(float_t) * N);
	cudaBindTexture(0, T_tex, p->T, sizeof(float_t) * N);
	cudaBindTexture(0, tool_particle_tex, p->tool_particle, sizeof(float_t) * N);
	cudaBindTexture(0, hashes_tex, p->hash, sizeof(int) * N);
	cudaBindTexture(0, k_tex, p->k, sizeof(float_t) * N);

	cudaBindTexture(0, cells_start_tex, cells_start, sizeof(int) * num_cell);
	cudaBindTexture(0, cells_end_tex, cells_end, sizeof(int) * num_cell);

	do_interactions_heat<<<dG, dB>>>(p->T_t, p->cp, p->N, h_thermals_sub.alpha, h_thermals_rod.alpha);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(rho_tex);
	cudaUnbindTexture(T_tex);
	cudaUnbindTexture(tool_particle_tex);
	cudaUnbindTexture(hashes_tex);
	cudaUnbindTexture(k_tex);

	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);
}

void interactions_setup_geometry_constants(grid_base *g)
{
	geom_constants geometry_h;
	geometry_h.nx = g->nx();
	geometry_h.ny = g->ny();
	geometry_h.nz = g->nz();
	geometry_h.bbmin_x = g->bbmin_x();
	geometry_h.bbmin_y = g->bbmin_y();
	geometry_h.bbmin_z = g->bbmin_z();
	geometry_h.dx = g->dx();
	geometry_h.gd = g->gd();

	cudaMemcpyToSymbol(geometry, &geometry_h, sizeof(geom_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_physical_constants(phys_constants physics_h)
{
	cudaMemcpyToSymbol(physics, &physics_h, sizeof(phys_constants), 0, cudaMemcpyHostToDevice);
	if (physics_h.mass == 0 || isnan(physics_h.mass))
	{
		printf("WARNING: invalid mass set!\n");
	}
}

void interactions_setup_corrector_constants(corr_constants correctors_h)
{
	cudaMemcpyToSymbol(correctors, &correctors_h, sizeof(corr_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_rod_constants(rod_constants rod_h)
{
	cudaMemcpyToSymbol(rod, &rod_h, sizeof(rod_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_thermal_constants_substrate(trml_constants trml_h)
{
	h_thermals_sub = trml_h;
	m_thermal_workpiece = trml_h.alpha != 0.;
#if defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_PSE)
	if (m_thermal_workpiece)
	{
		printf("considering thermal diffusion in workpiece\n");
#if !(defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_PSE))
		printf("Interactive wp warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
	}
	printf("Diffusitvity workpiece: %e\n", trml_h.alpha);
#endif

	cudaMemcpyToSymbol(trml_sub, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	check_cuda_error("error copying thermal constants.\n");
}

void interactions_setup_thermal_constants_rod(trml_constants trml_h, tool_3d_gpu *tool)
{
	h_thermals_rod = trml_h;
	m_thermal_tool = trml_h.alpha != 0.;
#if defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_PSE)
	if (m_thermal_tool)
	{
		tool->set_thermal(true);
		printf("considering thermal diffusion from workpiece into tool\n");
#if !(defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_PSE))
		printf("Interactive tool warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
	}
	printf("Diffusitvity tool: %e\n", trml_h.alpha);
#endif

	cudaMemcpyToSymbol(trml_rod, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	check_cuda_error("error copying thermal constants.\n");
}

void interactions_setup_johnson_cook_constants(joco_constants johnson_cook_sub_h, joco_constants johnson_cook_rod_h)
{
	cudaMemcpyToSymbol(johnson_cook_sub, &johnson_cook_sub_h, sizeof(joco_constants), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(johnson_cook_rod, &johnson_cook_rod_h, sizeof(joco_constants), 0, cudaMemcpyHostToDevice);
}

//////////////////////////////////////////////////////////////// independent_threading ////////////////////////////////////////////////////////////////////////////

__global__ void do_ni_smoothing_nested_independent_threading(float3_t *__restrict__ cfm, int *__restrict__ surface, const float_t *__restrict__ blanked,
															 unsigned int N, const float_t *__restrict__ in_tool, float3_t *sum_buff, int *count_buff, int shift)
{

	int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	int th = threadIdx.x;

	if (pidx >= N)
		return;

	// you need to launch another kernel to set these values or use cuda memset
	float3_t cfmi = make_float3_t(0., 0., 0.);

	// check if it is in the core
	if (surface[pidx / shift] == 0)
		return;

	// check if it is in the substrate
	if (in_tool[pidx / shift] == 1.)
		return;

	// check if it is blanked
	if (blanked[pidx / shift] == 1.)
	{
		cfm[pidx / shift] = cfmi;
		return;
	}

	/* Will BE CHANGED MANUALLY */

	const int sh_size = 256;
	if (sh_size != blockDim.x)
		assert(0);
	__shared__ float3_t sh_sum[sh_size];
	__shared__ int sh_count_sum[sh_size];

	sh_sum[th] = make_float3_t(0., 0., 0.);
	sh_count_sum[th] = 0;
	__syncthreads();

	if (th >= 27 && th < 32)
		return; // need to be adjusted to be not hard coded
	// Set the buffers to zero
	/* sum_buff[pidx] = cfmi;
	count_buff[pidx] = 0; */

	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	int gd = geometry.gd;

	// load particle data at pidx
	float4_t pi = texfetch4(pos_tex, pidx / shift);
	float_t hi = texfetch1(h_tex, pidx / shift);

	int hashi = tex1Dfetch(hashes_tex, pidx / shift);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	// shifting to the left and start from it
	int sub_value = gd / 2;
	// number of cells in 1d
	int gd_idx = gd + 1;
	int i = gi - sub_value + (pidx % gd_idx);
	int j = gj - sub_value + (pidx / gd_idx) % gd_idx;
	int k = gk - sub_value + (pidx / (gd_idx * gd_idx)) % gd_idx;

	if ((0 <= i <= nx) && (0 <= j <= ny) && (0 <= k <= nz))
	{
		int idx;
		hash(i, j, k, idx);

		int c_start = tex1Dfetch(cells_start_tex, idx);
		if (c_start != 0xffffffff)
		{

			int c_end = tex1Dfetch(cells_end_tex, idx);
			for (int iter = c_start; iter < c_end; iter++)
			{

				float3_t cfmj = cfm[iter];
				if (cfmj.x == 0. && cfmj.y == 0. && cfmj.z == 0.)
					continue;

				// load vars at neighbor particle
				float4_t pj = texfetch4(pos_tex, iter);
				float_t xij = pi.x - pj.x;
				float_t yij = pi.y - pj.y;
				float_t zij = pi.z - pj.z;
				float_t r = sqrtf(xij * xij + yij * yij + zij * zij);
				if (r < hi)
				{
					/* sum_buff  [pidx].x += cfmj.x;
					sum_buff  [pidx].y += cfmj.y;
					sum_buff  [pidx].z += cfmj.z;
					count_buff[pidx]++;  */

					sh_sum[th].x += cfmj.x;
					sh_sum[th].y += cfmj.y;
					sh_sum[th].z += cfmj.z;
					sh_count_sum[th]++;
				}
			}
		}
	}

	/* 	sh_sum       [th] = sum_buff  [pidx];
		sh_count_sum [th] = count_buff[pidx];  */
	__syncthreads();

	for (unsigned int s = 1; s < shift; s *= 2)
	{
		if (th % (2 * s) == 0)
		{

			sh_sum[th].x += sh_sum[th + s].x;
			sh_sum[th].y += sh_sum[th + s].y;
			sh_sum[th].z += sh_sum[th + s].z;

			sh_count_sum[th] += sh_count_sum[th + s];
		}

		__syncthreads();
	}

	if (th % shift == 0 && sh_count_sum[th] != 0)
	{

		cfm[pidx / shift].x = sh_sum[th].x / sh_count_sum[th];
		cfm[pidx / shift].y = sh_sum[th].y / sh_count_sum[th];
		cfm[pidx / shift].z = sh_sum[th].z / sh_count_sum[th];
	}
}

void interactions_ni_smoothing_nested_independent_threading(particle_gpu *particles, const int *cells_start, const int *cells_end, int num_cell,
															float3_t *sum_buff, int *count_buff, int *start_array, int *end_array, int gd)
{

	int total_gd = (gd + 1) * (gd + 1) * (gd + 1);
	if (total_gd % 32 != 0)
		total_gd = (total_gd / 32 + 1) * 32;

	const unsigned int block_size = 256;

	dim3 dG((particles->N * total_gd + block_size - 1) / block_size);
	dim3 dB(block_size);
	particle_gpu *p = particles;
	int N = p->N;

	cudaBindTexture(0, pos_tex, p->pos, sizeof(float_t) * N * 4);
	cudaBindTexture(0, h_tex, p->h, sizeof(float_t) * N);
	cudaBindTexture(0, hashes_tex, p->hash, sizeof(int) * N);
	cudaBindTexture(0, cells_start_tex, cells_start, sizeof(int) * num_cell);
	cudaBindTexture(0, cells_end_tex, cells_end, sizeof(int) * num_cell);
	check_cuda_error("before interactions smoothing nested\n");

	do_ni_smoothing_nested_independent_threading<<<dG, dB>>>(particles->cfm, particles->surface, particles->blanked, p->N * total_gd, p->tool_particle,
															 sum_buff, count_buff, total_gd);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(hashes_tex);
	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);

	check_cuda_error("interactions smoothing nested\n");

	/*  	int* h_start_array = new int[N *3*3*3];
		int* h_end_array = new int[N *3*3*3];
		cudaMemcpy(h_start_array, start_array, sizeof(int) * N *3*3*3, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_end_array, end_array, sizeof(int) * N *3*3*3, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		for(int i= 0; i < N *3*3*3; i++){
				printf("%d, %d, %d\n",i, h_start_array[i], h_end_array[i]);

		}
		printf("*********************************************************************************\n"); */
}
/*
void interactions_monaghan_independent_threading(particle_gpu *particles, const int *cells_start, const int *cells_end, int num_cell, int gd) {

	int total_gd = (2*gd+1) * (2*gd+1) * (2*gd+1);
	if(total_gd%32 != 0)	total_gd = (total_gd/32+1)*32;


	const unsigned int block_size = 256;
	dim3 dG((particles->N * total_gd + block_size-1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	cudaBindTexture(0, pos_tex,    p->pos,    sizeof(float_t)*N*4);
	cudaBindTexture(0, vel_tex,    p->vel,    sizeof(float_t)*N*4);
	cudaBindTexture(0, h_tex,      p->h,      sizeof(float_t)*N);
	cudaBindTexture(0, rho_tex,    p->rho,    sizeof(float_t)*N);
	cudaBindTexture(0, p_tex,      p->p,      sizeof(float_t)*N);
	cudaBindTexture(0, hashes_tex, p->hash,   sizeof(int)*N);
#ifdef Thermal_Conduction_Brookshaw
	cudaBindTexture(0, T_tex,		p->T,		sizeof(float_t)*N);
#endif

	cudaBindTexture(0, tool_particle_tex, p->tool_particle, sizeof(float_t)*N);

	cudaBindTexture(0, cells_start_tex, cells_start,   sizeof(int)*num_cell);
	cudaBindTexture(0, cells_end_tex,   cells_end,     sizeof(int)*num_cell);

	check_cuda_error("before interactions monaghan\n");

	do_interactions_monaghan_independent_threading<<<dG,dB>>>(particles->cfm, particles->blanked, particles->S, particles->R, p->v_der,
															  p->S_der, p->T_t, p->pos_t, p->vel_t, p->N * total_gd, p->tool_particle, total_gd, particles->fixed);

	cudaUnbindTexture(pos_tex);
	cudaUnbindTexture(h_tex);
	cudaUnbindTexture(rho_tex);
	cudaUnbindTexture(p_tex);
	cudaUnbindTexture(hashes_tex);
#ifdef Thermal_Conduction_Brookshaw
	cudaUnbindTexture(T_tex);
#endif

	cudaUnbindTexture(tool_particle_tex);

	cudaUnbindTexture(cells_start_tex);
	cudaUnbindTexture(cells_end_tex);

	check_cuda_error("interactions monaghan\n");
} */
