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

// leap frog time integration. This file contains the basic simulation loop

#ifndef LEAP_FROG_H_
#define LEAP_FROG_H_

#include "grid_gpu_base.h"
#include "particle_gpu.h"

#include "actions_gpu.h"
#include "blanking.h"
#include "interactions_gpu.h"
#include "tool_3d_gpu.h"
#include "ForcePIDController.h"
#include "types.h"

extern std::vector<tool_3d_gpu *> global_tool;
extern tool_forces *global_tool_forces;
extern blanking *global_blanking;
extern float_t global_Vsf;
extern float_t global_wz;
extern float_t global_die_velocity;
extern float_t global_time_dt;


class leap_frog
{
private:
	float4_t *pos_init;
	float4_t *vel_init;
	mat3x3_t *S_init;
	float_t *rho_init;
	float_t *T_init;

	int *cell_start;
	int *cell_end;

	float3_t *sum_buff;
	int *count_buff;

	int *start_array;
	int *end_array;
	float3 *forces;
	float_t fz_prev = 0.0;


	int *joined_count;
	float_t *joined_mean;

	FILE *m_fp = 0;
	bool m_verbose = true;

	std::vector<double> ts;
	std::vector<double> signal;

public:
	void step(particle_gpu *particles, grid_base *g, bool record_forces, int s);
	leap_frog(unsigned int num_part, unsigned int num_cell, int gd, float_t p_mass);
	~leap_frog();
	void rest_force();
	void report_foce(int) const;
	//void report_internal_kinetic_energy(particle_gpu *particles);
	void read_csv(std::string path);
	void interpolate(double value, double default_v);
	void update_wz();

	long _num_part;
	float_t _p_mass;

};

#endif /* LEAP_FROG_H_ */
