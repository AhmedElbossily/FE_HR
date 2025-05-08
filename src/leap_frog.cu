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

#include "leap_frog.h"

#include "grid_gpu_green.h"
struct inistate_struct
{
	float4_t *pos_init;
	float4_t *vel_init;
	mat3x3_t *S_init;
	float_t *rho_init;
	float_t *T_init;
	float_t *T_init_tool;
};

__global__ void init(particle_gpu particles, inistate_struct inistate)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init)
		return;
	if (particles.blanked[pidx] == 1.)
		return;

	int part = particles.tool_particle[pidx];
	// die return or container
	if (part == 0 || part == 2)
	{
		inistate.T_init[pidx] = particles.T[pidx];
		return;
	}

	inistate.pos_init[pidx] = particles.pos[pidx];
	inistate.vel_init[pidx] = particles.vel[pidx];
	inistate.S_init[pidx] = particles.S[pidx];
	inistate.rho_init[pidx] = particles.rho[pidx];
	inistate.T_init[pidx] = particles.T[pidx];
}

__global__ void predict(particle_gpu particles, inistate_struct inistate, float_t dt)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init)
		return;
	if (particles.blanked[pidx] == 1.)
		return;

	int part = particles.tool_particle[pidx];
	// die return or container
	if (part == 0 || part == 2)
	{
		particles.T[pidx] = inistate.T_init[pidx] + 0.5 * dt * particles.T_t[pidx];
		return;
	}

	particles.pos[pidx].x = inistate.pos_init[pidx].x + 0.5 * dt * particles.pos_t[pidx].x;
	particles.pos[pidx].y = inistate.pos_init[pidx].y + 0.5 * dt * particles.pos_t[pidx].y;
	particles.pos[pidx].z = inistate.pos_init[pidx].z + 0.5 * dt * particles.pos_t[pidx].z;

	particles.vel[pidx].x = inistate.vel_init[pidx].x + 0.5 * dt * particles.vel_t[pidx].x;
	particles.vel[pidx].y = inistate.vel_init[pidx].y + 0.5 * dt * particles.vel_t[pidx].y;
	particles.vel[pidx].z = inistate.vel_init[pidx].z + 0.5 * dt * particles.vel_t[pidx].z;

	particles.S[pidx][0][0] = inistate.S_init[pidx][0][0] + 0.5 * dt * particles.S_t[pidx][0][0];
	particles.S[pidx][0][1] = inistate.S_init[pidx][0][1] + 0.5 * dt * particles.S_t[pidx][0][1];
	particles.S[pidx][0][2] = inistate.S_init[pidx][0][2] + 0.5 * dt * particles.S_t[pidx][0][2];

	particles.S[pidx][1][0] = inistate.S_init[pidx][1][0] + 0.5 * dt * particles.S_t[pidx][1][0];
	particles.S[pidx][1][1] = inistate.S_init[pidx][1][1] + 0.5 * dt * particles.S_t[pidx][1][1];
	particles.S[pidx][1][2] = inistate.S_init[pidx][1][2] + 0.5 * dt * particles.S_t[pidx][1][2];

	particles.S[pidx][2][0] = inistate.S_init[pidx][2][0] + 0.5 * dt * particles.S_t[pidx][2][0];
	particles.S[pidx][2][1] = inistate.S_init[pidx][2][1] + 0.5 * dt * particles.S_t[pidx][2][1];
	particles.S[pidx][2][2] = inistate.S_init[pidx][2][2] + 0.5 * dt * particles.S_t[pidx][2][2];

	particles.rho[pidx] = inistate.rho_init[pidx] + 0.5 * dt * particles.rho_t[pidx];

	particles.T[pidx] = inistate.T_init[pidx] + 0.5 * dt * particles.T_t[pidx];

	// printf("particles.pos_t[pidx].x: %.20lf \n", particles.pos_t[pidx].x);
}

__global__ void correct(particle_gpu particles, inistate_struct inistate, float_t dt)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init)
		return;
	if (particles.blanked[pidx] == 1.)
		return;

	int part = particles.tool_particle[pidx];
	if (part == 0 || part == 2)
	{
		particles.T[pidx] = inistate.T_init[pidx] + dt * particles.T_t[pidx];
		return;
	}

	particles.pos[pidx].x = inistate.pos_init[pidx].x + dt * particles.pos_t[pidx].x;
	particles.pos[pidx].y = inistate.pos_init[pidx].y + dt * particles.pos_t[pidx].y;
	particles.pos[pidx].z = inistate.pos_init[pidx].z + dt * particles.pos_t[pidx].z;

	particles.vel[pidx].x = inistate.vel_init[pidx].x + dt * particles.vel_t[pidx].x;
	particles.vel[pidx].y = inistate.vel_init[pidx].y + dt * particles.vel_t[pidx].y;
	particles.vel[pidx].z = inistate.vel_init[pidx].z + dt * particles.vel_t[pidx].z;

	particles.S[pidx][0][0] = inistate.S_init[pidx][0][0] + dt * particles.S_t[pidx][0][0];
	particles.S[pidx][0][1] = inistate.S_init[pidx][0][1] + dt * particles.S_t[pidx][0][1];
	particles.S[pidx][0][2] = inistate.S_init[pidx][0][2] + dt * particles.S_t[pidx][0][2];

	particles.S[pidx][1][0] = inistate.S_init[pidx][1][0] + dt * particles.S_t[pidx][1][0];
	particles.S[pidx][1][1] = inistate.S_init[pidx][1][1] + dt * particles.S_t[pidx][1][1];
	particles.S[pidx][1][2] = inistate.S_init[pidx][1][2] + dt * particles.S_t[pidx][1][2];

	particles.S[pidx][2][0] = inistate.S_init[pidx][2][0] + dt * particles.S_t[pidx][2][0];
	particles.S[pidx][2][1] = inistate.S_init[pidx][2][1] + dt * particles.S_t[pidx][2][1];
	particles.S[pidx][2][2] = inistate.S_init[pidx][2][2] + dt * particles.S_t[pidx][2][2];

	particles.rho[pidx] = inistate.rho_init[pidx] + dt * particles.rho_t[pidx];

	particles.T[pidx] = inistate.T_init[pidx] + dt * particles.T_t[pidx];
}

__global__ void do_reset_contact_forces(particle_gpu particles)
{
	const unsigned int N = particles.N;

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;

	// reset contact forces
	particles.fc[pidx].x = 0.;
	particles.fc[pidx].y = 0.;
	particles.fc[pidx].z = 0.;
	particles.ft[pidx].x = 0.;
	particles.ft[pidx].y = 0.;
	particles.ft[pidx].z = 0.;
	particles.n[pidx].x = 0.;
	particles.n[pidx].y = 0.;
	particles.n[pidx].z = 0.;
}

void leap_frog::step(particle_gpu *particles, grid_base *g, bool record_forces, int s)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N_init + block_size - 1) / block_size);
	dim3 dB(block_size);

	inistate_struct inistate;
	inistate.pos_init = pos_init;
	inistate.vel_init = vel_init;
	inistate.S_init = S_init;
	inistate.rho_init = rho_init;
	inistate.T_init = T_init;

	if (/* s%10 == 0 || s==1 */ true)
	{
		if (global_blanking)
		{
			global_blanking->update();
			perform_blanking(particles, global_blanking);
		}

		// spatial sorting
		g->update_geometry(particles);
		g->assign_hashes(particles);
		if (global_blanking)
		{
			g->adapt_particle_number(particles);
		}
		g->sort(particles);
		g->get_cells(particles, cell_start, cell_end);
	}

	init<<<dG, dB>>>(*particles, inistate);
	predict<<<dG, dB>>>(*particles, inistate, global_time_dt);

	material_eos(particles);
	corrector_artificial_stress(particles);

	interactions_setup_geometry_constants(g);
	interactions_monaghan(particles, cell_start, cell_end, g->num_cell(), s);
	// calculate_center_of_mass(particles, s);
	// interactions_free_surface_detection(particles, cell_start, cell_end, g->num_cell(), s) ;
	// interactions_ni_smoothing(particles, cell_start, cell_end, g->num_cell(), sum_buff, count_buff,s);

#ifdef Thermal_Conduction_PSE
	// if (s % 1000 == 0)
	interactions_heat_pse(particles, cell_start, cell_end, g->num_cell());
#endif

	material_stress_rate_jaumann(particles);
	contmech_continuity(particles);
	contmech_momentum(particles);
	contmech_advection(particles);

#ifndef NDEBUG
	debug_invalidate(particles);
#endif

	correct<<<dG, dB>>>(*particles, inistate, global_time_dt);

	plasticity_johnson_cook(particles);

	do_reset_contact_forces<<<dG, dB>>>(*particles);

	if (record_forces)
	{
		global_tool_forces->reset();
	}

	rest_force();
	interactions_rod_force_Songwon(particles, cell_start, cell_end, g->num_cell(),
								   joined_count, joined_mean, forces);

	// material_fric_heat_gen(particles, global_tool[0]->get_vel());
	// perform_merge_conditions_thermal(particles, joined_count);
	update_cp_and_k(particles);
	// global_tool[0]->free_fall(particles);

	interpolate(global_time_current * global_Vsf, 19.0e-3);
	update_wz();
	perform_boundary_conditions(particles, joined_count, joined_mean);
	perform_boundary_conditions_thermal(particles);

	// actions_move_tool_particles(particles, global_tool[0]);

#ifndef NDEBUG
	debug_check_valid_full(particles);
#endif
	check_cuda_error();
}

leap_frog::leap_frog(unsigned int num_part, unsigned int num_cell, int gd, float_t p_mass)
{
	std::cout << "Loading the signal\n";
	read_csv("../Data/filtered_velocity.csv");
	std::cout << "signal Loaded successfully\n";

	_num_part = num_part;
	_p_mass = p_mass;
	printf("p_mass = %.20lf\n", p_mass);

	cudaMalloc((void **)&pos_init, sizeof(float4_t) * num_part);
	cudaMalloc((void **)&vel_init, sizeof(float4_t) * num_part);
	cudaMalloc((void **)&S_init, sizeof(mat3x3_t) * num_part);
	cudaMalloc((void **)&rho_init, sizeof(float_t) * num_part);
	cudaMalloc((void **)&T_init, sizeof(float_t) * num_part);

	cudaMalloc((void **)&cell_start, sizeof(int) * num_cell);
	cudaMalloc((void **)&cell_end, sizeof(int) * num_cell);

	int total_gd = (gd + 1) * (gd + 1) * (gd + 1);

	if (total_gd % 32 != 0)
		total_gd = (total_gd / 32 + 1) * 32;

	cudaMalloc((void **)&sum_buff, sizeof(float3_t) * num_part * total_gd);
	cudaMemset(sum_buff, 0, sizeof(float3_t) * num_part * total_gd);

	cudaMalloc((void **)&count_buff, sizeof(int) * num_part * total_gd);
	cudaMemset(count_buff, 0, sizeof(int) * num_part * total_gd);

	cudaMalloc((void **)&joined_count, sizeof(int));
	cudaMemset(joined_count, 0, sizeof(int));

	cudaMalloc((void **)&joined_mean, sizeof(float_t));
	cudaMemset(joined_mean, 0., sizeof(float_t));

	/* 	cudaMalloc((void **) &start_array,sizeof(int) *num_part *total_gd);
		cudaMalloc((void **) &end_array,   sizeof(int) *num_part *total_gd);

		cudaMemset(start_array, 0, sizeof(int) *num_part *total_gd);
		cudaMemset(end_array, 0, sizeof(int) *num_part *total_gd); */

	cudaMalloc((void **)&forces, sizeof(float3));
	cudaMemset(forces, 0., sizeof(float3));

	m_fp = fopen("Results_sammary/Rforces.csv", "w+");
}

void leap_frog::report_foce(int s) const
{

	float3 *h_force = new float3();
	cudaMemcpy((void *)h_force, forces, sizeof(float3), cudaMemcpyDeviceToHost);

	h_force->x /= 1000.;
	h_force->y /= 1000.;
	h_force->z /= 1000.;

	if (m_verbose)
	{
		printf("Tool Forces:\n");
		printf("%.1f %.1f %.1f ", (1.e-6) * h_force->x, (1.e-6) * h_force->y, (1.e-6) * h_force->z);
		printf("\n");
	}

	fprintf(m_fp, "%f,%f", s * global_time_dt * global_Vsf, h_force->z);
	fprintf(m_fp, "\n");

	fflush(m_fp);
	delete h_force;
}
void leap_frog::rest_force()
{
	cudaMemset(forces, 0, sizeof(float3));
}

leap_frog::~leap_frog()
{
	fclose(m_fp);
}

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

void leap_frog::read_csv(std::string path)
{
	// std::string filename = "../Data/filtered_velocity.csv";
	std::string filename = path;
	// Open the CSV file
	std::ifstream file(filename);
	if (!file.is_open())
	{
		std::cout << "Error: Could not open file " << filename << std::endl;
		exit(-1);
	}

	// Read header line (optional)
	std::string line;

	// Read data lines
	while (std::getline(file, line))
	{
		std::stringstream line_stream(line);
		std::string time_str, velocity_str;

		// Extract time and velocity from each line
		if (std::getline(line_stream, time_str, ',') && std::getline(line_stream, velocity_str))
		{
			// Add data to vectors
			ts.push_back(std::stod(time_str));
			signal.push_back(std::stod(velocity_str));
		}
		else
		{
			std::cout << "Error: Invalid data format in line: " << line << std::endl;
			exit(-1);
		}
	}

	// Close the file
	file.close();
}

void leap_frog::interpolate(double value, double default_v)
{

	float_t ct = global_time_current * global_Vsf;
	float_t dt_control =global_time_dt;
	//const float_t epsilon = 1e-6; // Small tolerance
	//if (fmod(ct, dt_control) < epsilon)
	{
		float3 *h_force = new float3();
		cudaMemcpy((void *)h_force, forces, sizeof(float3), cudaMemcpyDeviceToHost);
		float_t fz =  (1.e-9) * h_force->z; // convert to KN/mm^2 
		
		// ForcePIDController pid(0.5f, 0.1f, 0.05f, dt_control, -0.001f, 0.4f); // Tune Kp, Ki, Kd
		
/* 		if (ct < 30){
			ForcePIDController pid(0.5f, 0.1f, 0.05f, dt_control, -0.4 * global_Vsf * 3., 0.4 * global_Vsf * 3.); // Tune Kp, Ki, Kd
			global_die_velocity = pid.computeVelocity(ct, fz);
		}
			
		else
		{
			ForcePIDController pid(0.5f, 0.1f, 0.05f, dt_control, -2. * global_Vsf * 3., -0.01); // Tune Kp, Ki, Kd
			global_die_velocity = pid.computeVelocity(ct, fz);
		
		} */

		ForcePIDController pid(0.5f, 0.1f, 0.05f, dt_control, -2. * global_Vsf * 3., -0.01); // Tune Kp, Ki, Kd
		global_die_velocity = pid.computeVelocity(ct, fz);
			
		

		delete h_force;
	}

}


void leap_frog::update_wz()
{
	float_t ct = global_time_current * global_Vsf;
	float_t wz = 90.;

/* 	if (ct < 30.)
		wz = 300.;
	else if (ct > 40.) 
		wz = 90.;
	else
		wz = -21 * ct + 930; */


	global_wz = wz * 0.104719755 * global_Vsf;
}
