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

#include "benchmarks_single_grain.h"

template <typename Out>
static void split(const std::string &s, char delim, Out &result)
{
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim))
	{
		result.push_back(item);
	}
}

particle_gpu *setup_single_grain_5tool(grid_base **grid)
{

	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	joco_constants joco = make_joco_constants();

	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();

	bool sample_tools = true;
	bool interp_steady_temp = true;
	bool separate = false;
	bool thermal = true;

	float_t target_feed = 20 * 1e-6 * 100;
	int depth_fac = 3;
	float_t depth_wp = depth_fac * target_feed;

	float_t num_particles_in_feed = 12; // 12 = 1.5M is resolution used in paper
										//  if initial positions are loaded from a result file, make sure that mass and smoothing length
										//  are the same as the ones used to produce the result file. For ``init_hires.vtk'' this is 12

	int nz = depth_fac * num_particles_in_feed;

	float_t dz = depth_wp / (nz - 1);

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t> positions;
	ldynak_read_triangles_from_tetmesh("pin25_reduziert_20062018.k", triangles, positions);

	vec3_t bbmin_tool, bbmax_tool;
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);
	geometry_translate(triangles, positions, vec3_t(0., 0., -bbmin_tool.z)); // lift tool to zero level
	geometry_translate(triangles, positions, vec3_t(0., 0., -target_feed));	 // push down to target feed again

	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	geometry_translate(triangles, positions, vec3_t(0., -bbmin_tool.y - ly_tool / 2., 0.)); // center tool to y zero

	float_t nudge = 6.4e-4;
	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	geometry_translate(triangles, positions, vec3_t(-bbmin_tool.x - lx_tool + nudge, 0., 0.)); // move tool in x direction such that bottom surface almost touches wp

	// update bb
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	std::vector<mesh_triangle> triangles_l_l = triangles; // leading left
	std::vector<vec3_t> positions_l_l = positions;

	std::vector<mesh_triangle> triangles_l_r = triangles; // leading right
	std::vector<vec3_t> positions_l_r = positions;

	std::vector<mesh_triangle> triangles_t_l = triangles; // trailing left
	std::vector<vec3_t> positions_t_l = positions;

	std::vector<mesh_triangle> triangles_t_r = triangles; // trailing right
	std::vector<vec3_t> positions_t_r = positions;

	float_t edge_diamond = 80 * 1e-6 * 100;
	float_t lx = 3 * edge_diamond;
	float_t ly = 3 * edge_diamond + 2 * edge_diamond + 1 * edge_diamond;
	float_t lz = depth_wp;

	float_t sp_length = (separate) ? 0.5 * edge_diamond : 0;

	geometry_translate(triangles_l_l, positions_l_l, vec3_t(0., -1.5 * edge_diamond - sp_length, 0.));
	geometry_translate(triangles_l_r, positions_l_r, vec3_t(0., +1.5 * edge_diamond + sp_length, 0.));
	geometry_translate(triangles_t_l, positions_t_l, vec3_t(-edge_diamond, -0.5 * edge_diamond - sp_length, 0.));
	geometry_translate(triangles_t_r, positions_t_r, vec3_t(-edge_diamond, +0.5 * edge_diamond + sp_length, 0.));

	//	geometry_translate(triangles,     positions,     vec3_t(lx*0.75, 0., 0.));
	//	geometry_translate(triangles_l_l, positions_l_l, vec3_t(lx*0.75, 0., 0.));
	//	geometry_translate(triangles_l_r, positions_l_r, vec3_t(lx*0.75, 0., 0.));
	//	geometry_translate(triangles_t_l, positions_t_l, vec3_t(lx*0.75, 0., 0.));
	//	geometry_translate(triangles_t_r, positions_t_r, vec3_t(lx*0.75, 0., 0.));

	//	geometry_translate(triangles_t_l, positions_t_l, vec3_t(-lx, 0., 0.));
	//	geometry_translate(triangles_t_r, positions_t_r, vec3_t(-lx, 0., 0.));

	int nx = lx / dz; // equi particle spacing
	int ny = ly / dz;

	int n = nx * ny * nz;

	int part_iter = 0;
	float4_t *pos = new float4_t[n];
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int k = 0; k < nz; k++)
			{
				float_t px = i * dz;
				float_t py = j * dz - ly / 2.;
				float_t pz = -k * dz;
				float4_t cur_pos;

				cur_pos.x = px;
				cur_pos.y = py;
				cur_pos.z = pz;

				pos[part_iter] = cur_pos;

				part_iter++;
			}
		}
	}

	//	// read result file for positions if desired
	//	auto vec_pos = vtk_read_init_pos("init_hires.vtk");
	//	n = vec_pos.size();
	//	float4_t* pos = new float4_t[n];
	//	unsigned int pos_it = 0;
	//	for (unsigned int i = 0; i < n; i++) {
	//		pos[pos_it] = vec_pos[i];
	//		pos_it++;
	//	}
	//	n = pos_it;

	printf("calculating with %d regular particles\n", n);

	float_t hdx = 1.7;
	phys.E = 1.1;
	phys.nu = 0.35;
	phys.rho0 = 4.43;
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dz * dz * dz * phys.rho0;

	bool use_khan = true;

	if (use_khan)
	{
		joco.A = 0.0110400; // Mansur Akbari 2016, Tabelle 1
		joco.B = 0.0103600; // Mansur Akbari 2016, Tabelle 1
		joco.C = 0.0139000; // Mansur Akbari 2016, Tabelle 1
		joco.m = 0.7794;	// Mansur Akbari 2016, Tabelle 1
		joco.n = 0.6349;	// Mansur Akbari 2016, Tabelle 1
		joco.Tref = 300.;
		joco.Tmelt = 1723.0000;	 // Mansur Akbari 2016, im Text S.65
		joco.eps_dot_ref = 1e-6; // Mansur Akbari 2016, im Text S.65
		joco.clamp_temp = 1.;
	}
	else
	{
		joco.A = 0.0086200;
		joco.B = 0.0033100;
		joco.C = 0.0100000;
		joco.m = 0.8;
		joco.n = 0.34;
		joco.Tref = 300.;
		joco.Tmelt = 1836.0000;
		joco.eps_dot_ref = 1e-6;
		joco.clamp_temp = 1.;
	}

	float_t rho0_tool = 3.5; // https://en.wikipedia.org/wiki/Diamond
	trml_wp.T_init = joco.Tref;
	if (thermal)
	{
		trml_wp.cp = 553 * 1e-8;							  // Heat Capacity
		trml_wp.tq = 0.9;									  // Taylor-Quinney Coefficient
		trml_wp.k = 7.1 * 1e-13;							  // Thermal Conduction
		trml_wp.alpha = trml_wp.k / (phys.rho0 * trml_wp.cp); // Thermal diffusivity
		trml_wp.eta = 0.9;

		trml_tool.cp = 520 * 1e-8;	// https://www.engineeringtoolbox.com/specific-heat-solids-d_154.html
		trml_tool.tq = 0.;			// no plastic deformation in diamond
		trml_tool.k = 2200 * 1e-13; // https://en.wikipedia.org/wiki/Material_properties_of_diamond
		trml_tool.alpha = trml_tool.k / (rho0_tool * trml_tool.cp);
	}

	float_t c0 = sqrt(phys.K / phys.rho0);

	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
	}

	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *fixed = new float_t[n];
	float_t *T = new float_t[n];
	float_t *tool_p = new float_t[n];

	for (int i = 0; i < n; i++)
	{ // initialize particles
		rho[i] = phys.rho0;
		h[i] = hdx * dz;
		vel[i].x = 0.;
		vel[i].y = 0.;
		vel[i].z = 0.;
		fixed[i] = 0.;
		T[i] = joco.Tref;
		tool_p[i] = 0.;
	}

	unsigned int Nr_BC_particles = 0; // Counter boundary particles
	for (unsigned int i = 0; i < n; i++)
	{ // define boundary particles
		// if (pos[i].z < dz/2) {
		if (pos[i].z < -lz + dz / 2.0)
		{
			fixed[i] = 1;
			Nr_BC_particles++;
		}
	}

	//---------------------------------------------------------------------

	//	float_t v_tool = 8e-5;	//speed akbari
	float_t v_tool = 3e-3; // speed KW
	//	float_t v_tool = 570*100/1e6; //reckless speed

	float_t friction_coeff = 0.35; // fric ruttiumann
	//	float_t friction_coeff = 0.6;	//fric akbari
	//	float_t friction_coeff = 0.8;	//fric akbari increased

	mesh_compute_vertex_normals(triangles, positions);
	mesh_compute_vertex_normals(triangles_l_l, positions_l_l);
	mesh_compute_vertex_normals(triangles_l_r, positions_l_r);
	mesh_compute_vertex_normals(triangles_t_l, positions_t_l);
	mesh_compute_vertex_normals(triangles_t_r, positions_t_r);

	float_t avg_h = mesh_average_edge_length(triangles);

	std::vector<tool_3d *> cpu_tools;

	// remove this grain for pattern 3
	auto cpu_tool_l_m = new tool_3d(triangles, positions, avg_h);
	cpu_tools.push_back(cpu_tool_l_m);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_l_m, vec3_t(v_tool, 0., 0.), phys));

	auto cpu_tool_l_l = new tool_3d(triangles_l_l, positions_l_l, avg_h);
	cpu_tools.push_back(cpu_tool_l_l);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_l_l, vec3_t(v_tool, 0., 0.), phys));
	auto cpu_tool_l_r = new tool_3d(triangles_l_r, positions_l_r, avg_h);
	cpu_tools.push_back(cpu_tool_l_r);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_l_r, vec3_t(v_tool, 0., 0.), phys));

	auto cpu_tool_t_l = new tool_3d(triangles_t_l, positions_t_l, avg_h);
	cpu_tools.push_back(cpu_tool_t_l);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_t_l, vec3_t(v_tool, 0., 0.), phys));
	auto cpu_tool_t_r = new tool_3d(triangles_t_r, positions_t_r, avg_h);
	cpu_tools.push_back(cpu_tool_t_r);
	global_tool.push_back(new tool_3d_gpu(cpu_tool_t_r, vec3_t(v_tool, 0., 0.), phys));

	for (auto &it : global_tool)
	{
		it->set_algorithm_type(tool_3d_gpu::contact_algorithm::spatial_hashing);
		it->set_mu(friction_coeff);
	}

	int n_tool = 0;
	std::vector<float4_t> steady_state_tool;
	float3_t bbmin_steady = make_float3_t(FLT_MAX, FLT_MAX, FLT_MAX);
	float_t dx_steady = 0.;

	if (interp_steady_temp)
	{
		std::string line;
		std::ifstream infile("thermal_steady_lowres.txt", std::ifstream::in);

		while (std::getline(infile, line))
		{
			std::vector<std::string> tokens;
			split(line, ' ', tokens);
			if (tokens.size() != 4)
			{
				continue;
			}

			float_t x = std::stod(tokens[0], NULL);
			float_t y = std::stod(tokens[1], NULL);
			float_t z = std::stod(tokens[2], NULL);
			float_t t = std::stod(tokens[3], NULL);

			bbmin_steady.x = std::min(x, bbmin_steady.x);
			bbmin_steady.y = std::min(y, bbmin_steady.y);
			bbmin_steady.z = std::min(z, bbmin_steady.z);

			steady_state_tool.push_back(make_float4_t(x, y, z, t));
		}

		float_t ddx = fabs(steady_state_tool[0].x - steady_state_tool[1].x);
		float_t ddy = fabs(steady_state_tool[0].y - steady_state_tool[1].y);
		float_t ddz = fabs(steady_state_tool[0].z - steady_state_tool[1].z);

		dx_steady = std::max(std::max(ddx, ddy), ddz);
	}

	if (sample_tools)
	{
		std::vector<float4_t> samples;
		for (auto it : cpu_tools)
		{
			std::vector<vec3_t> cur_samples = it->sample(dz);
			std::vector<float4_t> cur_samples_flt_4;
			std::vector<float4_t> shifted_steady_tool(steady_state_tool.size());
			float3_t bbmin_cur_tool = make_float3_t(FLT_MAX, FLT_MAX, FLT_MAX);

			for (auto jt : cur_samples)
			{
				cur_samples_flt_4.push_back(make_float4_t(jt.x, jt.y, jt.z, 0.));

				bbmin_cur_tool.x = std::min(jt.x, bbmin_cur_tool.x);
				bbmin_cur_tool.y = std::min(jt.y, bbmin_cur_tool.y);
				bbmin_cur_tool.z = std::min(jt.z, bbmin_cur_tool.z);
			}

			if (interp_steady_temp)
			{
				for (unsigned int i = 0; i < steady_state_tool.size(); i++)
				{
					shifted_steady_tool[i].x = steady_state_tool[i].x - bbmin_steady.x + bbmin_cur_tool.x;
					shifted_steady_tool[i].y = steady_state_tool[i].y - bbmin_steady.y + bbmin_cur_tool.y;
					shifted_steady_tool[i].z = steady_state_tool[i].z - bbmin_steady.z + bbmin_cur_tool.z;
					shifted_steady_tool[i].w = steady_state_tool[i].w;
				}
				interp_temps(shifted_steady_tool, dx_steady * dx_steady * dx_steady, std::max(float_t(1.5) * dx_steady, hdx * dz), cur_samples_flt_4);
			}

			samples.insert(samples.end(), cur_samples_flt_4.begin(), cur_samples_flt_4.end());
		}

		n_tool = samples.size();
		float4_t *tool_pos = new float4_t[n_tool];

		float3_t bbmin_cur_tool = make_float3_t(FLT_MAX, FLT_MAX, FLT_MAX);

		for (int i = 0; i < n_tool; i++)
		{
			tool_pos[i].x = samples[i].x;
			tool_pos[i].y = samples[i].y;
			tool_pos[i].z = samples[i].z;
			tool_pos[i].w = samples[i].w;
		}

		pos = (float4_t *)realloc(pos, sizeof(float4_t) * (n + n_tool));
		vel = (float4_t *)realloc(vel, sizeof(float4_t) * (n + n_tool));
		rho = (float_t *)realloc(rho, sizeof(float_t) * (n + n_tool));
		T = (float_t *)realloc(T, sizeof(float_t) * (n + n_tool));
		h = (float_t *)realloc(h, sizeof(float_t) * (n + n_tool));
		fixed = (float_t *)realloc(fixed, sizeof(float_t) * (n + n_tool));
		tool_p = (float_t *)realloc(tool_p, sizeof(float_t) * (n + n_tool));

		int tool_pos_iter = 0;
		for (int i = n; i < n + n_tool; i++)
		{
			pos[i] = tool_pos[tool_pos_iter];
			tool_pos_iter++;

			rho[i] = rho0_tool;
			h[i] = hdx * dz;
			vel[i].x = 0.;
			vel[i].y = 0.;
			vel[i].z = 0.;
			fixed[i] = 0.;
			T[i] = joco.Tref;
			if (interp_steady_temp)
			{
				T[i] = fmax(tool_pos[tool_pos_iter].w, joco.Tref);
			}
			tool_p[i] = 1.;
		}

		// fix back boundaries of grains
		for (int i = n; i < n + n_tool; i++)
		{
			// top
			if (pos[i].z > bbmax_tool.z - 1.5 * dz)
			{
				fixed[i] = 1.;
				Nr_BC_particles++;
			}

			// back plane leading
			// left
			vec3_t p1_l_l(-0.0082535, -0.0179347, -2.41e-05);
			vec3_t p2_l_l(-0.0067885, -0.019261, 0.0019517);
			vec3_t p3_l_l(-0.0013051, -0.011949, -2.41e-05);
			// middle
			vec3_t p1_l_m(-0.0082535, -0.00193475, -2.41e-05);
			vec3_t p2_l_m(-0.0067885, -0.00326105, 0.0019517);
			vec3_t p3_l_m(-0.0013051, 0.00405095, -2.41e-05);
			// right
			vec3_t p1_l_r(-0.0082535, 0.0140652, -2.41e-05);
			vec3_t p2_l_r(-0.0067885, 0.012739, 0.0019517);
			vec3_t p3_l_r(-0.0013051, 0.020051, -2.41e-05);

			// back plane trailing
			// left
			vec3_t p1_t_l(-0.0162535, -0.00993475, -2.41e-05);
			vec3_t p2_t_l(-0.0147885, -0.011261, 0.0019517);
			vec3_t p3_t_l(-0.0093051, -0.00394905, -2.41e-05);
			// right
			vec3_t p1_t_r(-0.0162535, 0.00606525, -2.41e-05);
			vec3_t p2_t_r(-0.0147885, 0.00473895, 0.0019517);
			vec3_t p3_t_r(-0.0093051, 0.0120509, -2.41e-05);

			vec3_t nrm_l_l = glm::cross(p2_l_l - p1_l_l, p2_l_l - p3_l_l);
			vec3_t nrm_l_m = glm::cross(p2_l_m - p1_l_m, p2_l_m - p3_l_m);
			vec3_t nrm_l_r = glm::cross(p2_l_r - p1_l_r, p2_l_r - p3_l_r);
			vec3_t nrm_t_l = glm::cross(p2_t_l - p1_t_l, p2_t_l - p3_t_l);
			vec3_t nrm_t_r = glm::cross(p2_t_r - p1_t_r, p2_t_r - p3_t_r);

			vec3_t qp(pos[i].x, pos[i].y, pos[i].z);
			float_t d_l_l = fabs(glm::dot(nrm_l_l, qp - p1_l_l) / glm::length(nrm_l_l));
			float_t d_l_m = fabs(glm::dot(nrm_l_m, qp - p1_l_m) / glm::length(nrm_l_m));
			float_t d_l_r = fabs(glm::dot(nrm_l_r, qp - p1_l_r) / glm::length(nrm_l_r));
			float_t d_t_l = fabs(glm::dot(nrm_t_l, qp - p1_t_l) / glm::length(nrm_t_l));
			float_t d_t_r = fabs(glm::dot(nrm_t_r, qp - p1_t_r) / glm::length(nrm_t_r));

			if (d_l_l < 2 * dz || d_l_m < 2 * dz || d_l_r < 2 * dz || d_t_l < 2 * dz || d_t_r < 2 * dz)
			{
				fixed[i] = 1.;
				Nr_BC_particles++;
			}
		}

		printf("calculating with %d tool particles\n", n_tool);
	}

	//---------------------------------------------------------------------

	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(v_tool) + c0);
	global_time_dt = 0.5 * delta_t_max;
	global_time_final = (lx + 2 * edge_diamond) / v_tool;

	printf("t final %f\n", global_time_final);

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//---------------------------------------------------------------------

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n + n_tool);
	assert(check_cuda_error());

	*grid = new grid_gpu_green(n + n_tool, make_float3_t(std::min(-15.0 * dz, bbmin_tool.x - 15.0 * dz - edge_diamond), -15.0 * dz - ly / 2, -1.5 * nz * dz), make_float3_t(lx + 15.0 * dz, ly + 15.0 * dz - ly / 2, 3.0 * bbmax_tool.z), hdx * dz);
	global_blanking = new blanking(vec3_t(std::min(-15.0 * dz, bbmin_tool.x - 15.0 * dz - edge_diamond), -15.0 * dz - ly / 2, -1.5 * nz * dz),
								   vec3_t(lx + 15.0 * dz, ly + 15.0 * dz - ly / 2, 3.0 * bbmax_tool.z), 2.0 * 2.0);

	//---------------------------------------------------------------------

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_substrate(trml_wp);
	actions_setup_thermal_constants_rod(trml_tool);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml_wp);
	interactions_setup_thermal_constants_rod(trml_tool, global_tool[0]);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_single_grain_1tool_realscale(grid_base **grid)
{
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml = make_trml_constants();
	joco_constants joco = make_joco_constants();

	float_t target_feed = 30 * 1e-6 * 100; // 30 mu in cm
	int depth_fac = 3;
	float_t depth_wp = depth_fac * target_feed;

	float_t num_particles_in_feed = 5; // 5 = 1.2M particles, which was used for the paper
	int nz = depth_fac * num_particles_in_feed;

	float_t dz = depth_wp / (nz - 1);

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t> positions;
	ldynak_read_triangles_from_tetmesh("Diamant_von_Mansur_04072018_SDB1125-2025-D851.k", triangles, positions);

	vec3_t bbmin_tool, bbmax_tool;
	geometry_scale(triangles, positions, vec3_t(1. / 10000.0, 1. / 10000.0, 1. / 10000.0)); // micron to cm
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);
	geometry_translate(triangles, positions, vec3_t(0., 0., -bbmin_tool.z)); // lift tool to zero level
	geometry_translate(triangles, positions, vec3_t(0., 0., -target_feed));	 // push down to target feed again

	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	geometry_translate(triangles, positions, vec3_t(0., -bbmin_tool.y - ly_tool / 2., 0.)); // center tool to y zero

	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	geometry_translate(triangles, positions, vec3_t(-bbmin_tool.x - lx_tool + 260 * 100 * 1e-6, 0., 0.)); // move tool in x direction such that bottom surface almost touches wp

	// tool is at final position, update bbox
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	float_t lx = 2 * 270 * 1e-6 * 100; // from measured base surface
	float_t ly = 2 * 320 * 1e-6 * 100;
	float_t lz = depth_wp;

	int nx = lx / dz; // equi particle spacing
	int ny = ly / dz;

	int n = nx * ny * nz;
	printf("calculating with %d\n", n);

	int part_iter = 0;
	float4_t *pos = new float4_t[n];
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int k = 0; k < nz; k++)
			{
				float_t px = i * dz;
				float_t py = j * dz - ly / 2.;
				float_t pz = -k * dz;
				float4_t cur_pos;

				cur_pos.x = px;
				cur_pos.y = py;
				cur_pos.z = pz;

				pos[part_iter] = cur_pos;

				part_iter++;
			}
		}
	}

	//---------------------------------------------------------------------

	float_t hdx = 1.7;
	phys.E = 1.1;
	phys.nu = 0.35;
	phys.rho0 = 4.43;
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dz * dz * dz * phys.rho0;

	bool use_khan = true;

	if (use_khan)
	{
		joco.A = 0.0110400; // Mansur Akbari 2016, Tabelle 1
		joco.B = 0.0103600; // Mansur Akbari 2016, Tabelle 1
		joco.C = 0.0139000; // Mansur Akbari 2016, Tabelle 1
		joco.m = 0.7794;	// Mansur Akbari 2016, Tabelle 1
		joco.n = 0.6349;	// Mansur Akbari 2016, Tabelle 1
		joco.Tref = 300.;
		joco.Tmelt = 1723.0000;	 // Mansur Akbari 2016, im Text S.65
		joco.eps_dot_ref = 1e-6; // Mansur Akbari 2016, im Text S.65
		joco.clamp_temp = 1.;
	}
	else
	{
		joco.A = 0.0086200;
		joco.B = 0.0033100;
		joco.C = 0.0100000;
		joco.m = 0.8;
		joco.n = 0.34;
		joco.Tref = 300.;
		joco.Tmelt = 1836.0000;
		joco.eps_dot_ref = 1e-6;
		joco.clamp_temp = 1.;
	}

	trml.cp = 553 * 1e-8;						 // Heat Capacity
	trml.tq = 0.9;								 // Taylor-Quinney Coefficient
	trml.k = 7.1 * 1e-13;						 // Thermal Conduction
	trml.alpha = trml.k / (phys.rho0 * trml.cp); // Thermal diffusivity
	trml.eta = 0.;
	trml.T_init = joco.Tref;

	float_t c0 = sqrt(phys.K / phys.rho0);

	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
	}

	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *fixed = new float_t[n];
	float_t *T = new float_t[n];

	for (int i = 0; i < n; i++)
	{ // initialize particles
		rho[i] = phys.rho0;
		h[i] = hdx * dz;
		vel[i].x = 0.;
		vel[i].y = 0.;
		vel[i].z = 0.;
		fixed[i] = 0;
		T[i] = joco.Tref;
	}

	unsigned int Nr_BC_particles = 0; // Counter boundary particles
	for (unsigned int i = 0; i < n; i++)
	{ // define boundary particles
		// if (pos[i].z < dz/2) {
		if (pos[i].z < -lz + dz / 2.0)
		{
			fixed[i] = 1;
			Nr_BC_particles++;
		}

		if (pos[i].x > lx - 5 * dz)
		{
			fixed[i] = 1;
			Nr_BC_particles++;
		}
	}

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	//---------------------------------------------------------------------

	//	float_t v_tool = 8e-5;	//speed akbari
	float_t v_tool = 3e-3;		   // speed KW
	float_t friction_coeff = 0.35; // fric ruttiumann
	//	float_t friction_coeff = 0.6;	//fric akbari
	//	float_t friction_coeff = 0.8;	//fric akbari increased

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, n);
	assert(check_cuda_error());

	mesh_compute_vertex_normals(triangles, positions);
	float_t avg_h = mesh_average_edge_length(triangles);

	auto cpu_tool = new tool_3d(triangles, positions, avg_h);
	global_tool.push_back(new tool_3d_gpu(cpu_tool, vec3_t(v_tool, 0., 0.), phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);

	//---------------------------------------------------------------------

	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(v_tool) + c0);
	global_time_dt = 0.5 * delta_t_max;
	global_time_final = lx / v_tool;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//---------------------------------------------------------------------

	*grid = new grid_gpu_green(n, make_float3_t(-15.0 * dz, -15.0 * dz - ly / 2, -1.5 * nz * dz), make_float3_t(lx + 15.0 * dz, ly + 15.0 * dz - ly / 2, 2.0 * bbmax_tool.z), hdx * dz);
	global_blanking = new blanking(vec3_t(-15.0 * dz, -15.0 * dz - ly / 2, -1.5 * nz * dz), vec3_t(lx + 15.0 * dz, ly + 15.0 * dz - ly / 2, 2.0 * bbmax_tool.z), 2.0 * 2.0);

	//---------------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_substrate(trml);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
particle_gpu *setup_single_grain_1tool(grid_base **grid)
{
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml = make_trml_constants();
	joco_constants joco = make_joco_constants();

	float_t target_feed = 20 * 1e-6 * 100;
	int depth_fac = 3;
	float_t depth_wp = depth_fac * target_feed;

	float_t num_particles_in_feed = 3; // 18 = ~2M reference simulation
	int nz = depth_fac * num_particles_in_feed;

	float_t dz = depth_wp / (nz - 1);

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t> positions;
	ldynak_read_triangles_from_tetmesh("pin25_reduziert_20062018.k", triangles, positions);
	// ldynak_read_triangles_from_tetmesh("../tri_ex.k", triangles, positions);

	vec3_t bbmin_tool, bbmax_tool;
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	// geometry_translate(triangles, positions, vec3_t(0., 0., -bbmin_tool.z));	//lift tool to zero level
	float_t nudge = 6.4e-4 / 6;
	geometry_translate(triangles, positions, vec3_t(0., 0., nudge)); // push down to target feed again

	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	geometry_translate(triangles, positions, vec3_t(0., -bbmin_tool.y - ly_tool / 2., 0.)); // center tool to y zero

	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;

	geometry_translate(triangles, positions, vec3_t((-1.0 * lx_tool) / 4.0, 0., 0.)); // move tool in x direction such that bottom surface almost touches wp

	// tool is at final position, update bbox
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);
	/* 	printf("max:%lf ,min:%lf \n",  bbmax_tool.x, bbmin_tool.x);
		exit(-1); */

	float_t lx = 2 * 80 * 1e-6 * 100; // full diamond measures
	float_t ly = 2 * 80 * 1e-6 * 100;
	float_t lz = depth_wp;

	int nx = lx / dz; // equi particle spacing
	int ny = ly / dz;

	int n = nx * ny * nz;
	printf("calculating with %d\n", n);

	int part_iter = 0;
	float4_t *pos = new float4_t[n];
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int k = 0; k < nz; k++)
			{
				float_t px = i * dz;
				float_t py = j * dz - ly / 2.;
				float_t pz = -k * dz;

				float sq = (px - lx / 2) * (px - lx / 2) + py * py;
				if (sq < (lx * lx) / 4.0)
				{

					float4_t cur_pos;
					cur_pos.x = px;
					cur_pos.y = py;
					cur_pos.z = pz;

					pos[part_iter] = cur_pos;

					part_iter++;
				}
			}
		}
	}

	n = part_iter;

	//---------------------------------------------------------------------

	float_t hdx = 1.7;
	phys.E = 1.1;
	phys.nu = 0.35;
	phys.rho0 = 4.43;
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dz * dz * dz * phys.rho0;

	bool use_khan = false;

	if (use_khan)
	{
		joco.A = 0.0110400; // Mansur Akbari 2016, Tabelle 1
		joco.B = 0.0103600; // Mansur Akbari 2016, Tabelle 1
		joco.C = 0.0139000; // Mansur Akbari 2016, Tabelle 1
		joco.m = 0.7794;	// Mansur Akbari 2016, Tabelle 1
		joco.n = 0.6349;	// Mansur Akbari 2016, Tabelle 1
		joco.Tref = 300.;
		joco.Tmelt = 1723.0000;	 // Mansur Akbari 2016, im Text S.65
		joco.eps_dot_ref = 1e-6; // Mansur Akbari 2016, im Text S.65
		joco.clamp_temp = 1.;
	}
	else
	{
		joco.A = 0.0086200;
		joco.B = 0.0033100;
		joco.C = 0.0100000;
		joco.m = 0.8;
		joco.n = 0.34;
		joco.Tref = 300.;
		joco.Tmelt = 1836.0000;
		joco.eps_dot_ref = 1e-6;
		joco.clamp_temp = 1.;
	}

	trml.cp = 553 * 1e-8;						 // Heat Capacity
	trml.tq = 0.9;								 // Taylor-Quinney Coefficient
	trml.k = 7.1 * 1e-13;						 // Thermal Conduction
	trml.alpha = trml.k / (phys.rho0 * trml.cp); // Thermal diffusivity
	trml.T_init = joco.Tref;

	float_t c0 = sqrt(phys.K / phys.rho0);

	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3; //<---!!!!
	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
	}

	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *fixed = new float_t[n];
	float_t *T = new float_t[n];

	for (int i = 0; i < n; i++)
	{ // initialize particles
		rho[i] = phys.rho0;
		h[i] = hdx * dz;
		vel[i].x = 0.;
		vel[i].y = 0.;
		vel[i].z = 0.;
		fixed[i] = 0;
		T[i] = joco.Tref;
	}

	unsigned int Nr_BC_particles = 0; // Counter boundary particles
	for (unsigned int i = 0; i < n; i++)
	{ // define boundary particles
		// if (pos[i].z < dz/2) {
		if (pos[i].z < -lz + dz / 2.0)
		{
			fixed[i] = 1;
			Nr_BC_particles++;
		}

		if (pos[i].x > lx - dz / 2.0)
		{
			fixed[i] = 1;
			Nr_BC_particles++;
		}
	}

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	//---------------------------------------------------------------------

	//	float_t v_tool = 8e-5;	//speed akbari
	float_t v_tool = 3e-3; // speed KW
	//	float_t v_tool = 570*100/1e6; //breakneck speed

	float_t friction_coeff = 0.35; // fric ruttiumann
	//	float_t friction_coeff = 0.6;	//fric akbari
	//	float_t friction_coeff = 0.8;	//fric akbari increased

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, n);
	assert(check_cuda_error());

	mesh_compute_vertex_normals(triangles, positions);
	float_t avg_h = mesh_average_edge_length(triangles);

	auto cpu_tool = new tool_3d(triangles, positions, avg_h);
	global_tool.push_back(new tool_3d_gpu(cpu_tool, vec3_t(0.0, 0., -v_tool), phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);

	//---------------------------------------------------------------------

	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(v_tool) + c0);
	global_time_dt = 0.5 * delta_t_max;
	global_time_final = (lz / 2) / v_tool;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//---------------------------------------------------------------------

	*grid = new grid_gpu_green(n, make_float3_t(-15.0 * dz, -15.0 * dz - ly / 2, -1.5 * nz * dz), make_float3_t(lx + 15.0 * dz, ly + 15.0 * dz - ly / 2, 1.5 * bbmax_tool.z), hdx * dz);
	global_blanking = new blanking(vec3_t(-15.0 * dz, -15.0 * dz - ly / 2, -1.5 * nz * dz), vec3_t(lx + 15.0 * dz, ly + 15.0 * dz - ly / 2, 1.5 * bbmax_tool.z), 2.0 * 2.0);

	global_blanking->set_max_fixed(true);
	global_blanking->set_min_fixed(true);

	//---------------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_substrate(trml);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

particle_gpu *setup_single_grain_1tool_trml_steady(grid_base **grid)
{
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();
	joco_constants joco = make_joco_constants();

	bool sample_tool = true;

	//	float_t v_tool = 8e-5;	//speed akbari
	float_t v_tool = 6e-6; // speed KW
	float_t vx = v_tool;   //  v_tool*5.0 with num_particles_in_feed = 5; was working fine

	// float_t v_tool = 1.0;	//speed KW
	float_t friction_coeff = 0.35; // fric ruttiumann
	//	float_t friction_coeff = 0.6;	//fric akbari
	//	float_t friction_coeff = 0.8;	//fric akbari increased

	float_t target_feed = 20 * 1e-6 * 100; // 20 mu in cm
	int depth_fac = 3;
	float_t depth_wp = depth_fac * target_feed;

	float_t num_particles_in_feed = 6;
	int nz = depth_fac * num_particles_in_feed;

	float_t dz = depth_wp / (nz - 1);
	printf("dz: %lf\n", dz);
	// exit(-1);

	float_t lx = 2 * 80 * 1e-6 * 100; // full diamond measures
	float_t ly = 2 * 80 * 1e-6 * 100;
	float_t lz = depth_wp;

	std::vector<mesh_triangle> triangles;
	std::vector<vec3_t> positions;

	// ldynak_read_triangles_from_tetmesh("pin25_reduziert_20062018.k", triangles, positions);
	// ldynak_read_triangles_from_tetmesh("../tri_ex.k", triangles, positions);
	ldynak_read_triangles_from_tetmesh("../tri_ex_longer.k", triangles, positions);

	vec3_t bbmin_tool, bbmax_tool;
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	geometry_translate(triangles, positions, vec3_t(0., 0., -bbmin_tool.z)); // lift tool to zero level
	// float_t nudge = 6.4e-4/6;
	// geometry_translate(triangles, positions, vec3_t(0., 0.,0.0 ));		//push down to target feed again

	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	geometry_translate(triangles, positions, vec3_t(0., -bbmin_tool.y - ly_tool / 2., 0.)); // center tool to y zero

	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	float_t lz_tool = bbmax_tool.z - bbmin_tool.z;

	geometry_translate(triangles, positions, vec3_t(-bbmin_tool.x - lx / 2., 0., 0.)); // move tool in x direction such that bottom surface almost touches wp

	ldynak_dump("translated_tool.k", triangles, positions);

	// tool is at final position, update bbox
	geometry_get_bb(positions, bbmin_tool, bbmax_tool);

	mesh_compute_vertex_normals(triangles, positions);

	float_t avg_h = mesh_average_edge_length(triangles);

	auto cpu_tool = new tool_3d(triangles, positions, avg_h);
	std::vector<vec3_t> samples;
	if (sample_tool)
		samples = cpu_tool->sample(dz, bbmin_tool, bbmax_tool);

	int nx = lx / dz; // equi particle spacing
	int ny = ly / dz;

	int n = nx * ny * nz;
	int n_tool = samples.size();
	printf("calculating with %d regular particles\n", n);
	printf("calculating with %d tool particles\n", n_tool);

	int part_iter = 0;
	float4_t *pos = new float4_t[n + n_tool];
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int k = 0; k < nz; k++)
			{
				float_t px = i * dz - lx / 2;
				float_t py = j * dz - ly / 2.;
				float_t pz = -k * dz;

				float sq = px * px + py * py;
				if (sq < (lx * lx) / 4.0)
				{

					float4_t cur_pos;
					cur_pos.x = px;
					cur_pos.y = py;
					cur_pos.z = pz - 2.0 * dz;

					pos[part_iter] = cur_pos;

					part_iter++;
				}
			}
		}
	}

	n = part_iter;

	for (auto &it : samples)
	{
		float4_t cur_pos;

		cur_pos.x = it.x;
		cur_pos.y = it.y;
		cur_pos.z = it.z;

		pos[part_iter] = cur_pos;

		part_iter++;
	}

	//---------------------------------------------------------------------

	float_t hdx = 1.7;
	phys.E = 1.1;
	phys.nu = 0.35;
	phys.rho0 = 4.43;
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dz * dz * dz * phys.rho0;

	bool use_khan = false;

	if (use_khan)
	{
		joco.A = 0.0110400; // Mansur Akbari 2016, Tabelle 1
		joco.B = 0.0103600; // Mansur Akbari 2016, Tabelle 1
		joco.C = 0.0139000; // Mansur Akbari 2016, Tabelle 1
		joco.m = 0.7794;	// Mansur Akbari 2016, Tabelle 1
		joco.n = 0.6349;	// Mansur Akbari 2016, Tabelle 1
		joco.Tref = 300.;
		joco.Tmelt = 1723.0000;	 // Mansur Akbari 2016, im Text S.65
		joco.eps_dot_ref = 1e-6; // Mansur Akbari 2016, im Text S.65
		joco.clamp_temp = 1.;
	}
	else
	{
		joco.A = 0.0086200;
		joco.B = 0.0033100;
		joco.C = 0.0100000;
		joco.m = 0.8;
		joco.n = 0.34;
		joco.Tref = 300.;
		joco.Tmelt = 1836.0000;
		joco.eps_dot_ref = 1e-6;
		joco.clamp_temp = 1.;
	}

	// Thermal properties Ti6Al4v from
	// https://www.azom.com/properties.aspx?ArticleID=1547
	trml_wp.cp = 553 * 1e-8;							  // Heat Capacity
	trml_wp.tq = 0.9;									  // Taylor-Quinney Coefficient
	trml_wp.k = 7.1 * 1e-13;							  // Thermal Conduction
	trml_wp.alpha = trml_wp.k / (phys.rho0 * trml_wp.cp); // Thermal diffusivity
	trml_wp.T_init = joco.Tref;
	trml_wp.eta = 0.9;

	trml_tool.cp = 553 * 1e-8; // https://www.engineeringtoolbox.com/specific-heat-solids-d_154.html
	trml_tool.tq = 0.9;		   // no plastic deformation in diamond
	trml_tool.k = 7.1 * 1e-13; // https://en.wikipedia.org/wiki/Material_properties_of_diamond
	float_t rho0_tool = 4.43;  // https://en.wikipedia.org/wiki/Diamond
	trml_tool.alpha = trml_tool.k / (rho0_tool * trml_tool.cp);
	trml_tool.T_init = joco.Tref;
	trml_tool.eta = 0.9;

	/* 	printf("trml_wp.alpha: %.20lf \n", trml_wp.alpha);
		printf("trml_tool.alpha: %.20lf \n", trml_tool.alpha);
		exit(-1); */

	float_t c0 = sqrt(phys.K / phys.rho0);

	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
	}

	float4_t *vel = new float4_t[n + n_tool];
	float_t *h = new float_t[n + n_tool];
	float_t *rho = new float_t[n + n_tool];
	float_t *fixed = new float_t[n + n_tool];
	float_t *T = new float_t[n + n_tool];
	float_t *tool_p = new float_t[n + n_tool];

	glm::vec3 w(0.0, 0.0, 1.25 * 2.0);

	for (int i = 0; i < n + n_tool; i++)
	{ // initialize particles
		rho[i] = phys.rho0;
		h[i] = hdx * dz;
		T[i] = joco.Tref;
		fixed[i] = 0;

		if (i >= n)
		{
			tool_p[i] = 1.;

			vel[i].x = -vx;
			vel[i].y = 0.;
			vel[i].z = 0.;
		}
		else
		{
			tool_p[i] = 0.;

			glm::vec3 r(pos[i].x, pos[i].y, 0.0);
			glm::vec3 v = glm::cross(w, r);

			vel[i].x = v.x;
			vel[i].y = v.y;
			vel[i].z = v_tool;
		}
	}

	unsigned int Nr_BC_particles = 0; // Counter boundary particles
	{
		vec3_t p1(-0.0082535, -0.00193475, 0.0004759);
		vec3_t p2(-0.0067885, -0.00326105, 0.0024517);
		vec3_t p3(-0.0013051, 0.00405095, 0.0004759);

		vec3_t nrm = glm::cross(p2 - p1, p2 - p3);

		for (unsigned int i = 0; i < n + n_tool; i++)
		{ // define boundary particles

			if (pos[i].z < -lz + dz)
			{
				fixed[i] = 1;
				Nr_BC_particles++;
			}

			if (pos[i].z > lz_tool - (dz / 2.0))
			{
				fixed[i] = 2;
				Nr_BC_particles++;
			}
			/* 			if (tool_p[i] == 1.) {	//thermal sinks at tool bounds

							if (pos[i].z > bbmax_tool.z - 1.5*dz) {
								fixed[i] = 1.;
								Nr_BC_particles++;
							}

							vec3_t qp(pos[i].x, pos[i].y, pos[i].z);
							float_t d = glm::dot(nrm, qp-p1)/glm::length(nrm);

							if (d < 1.5*dz) {
								fixed[i] = 1.;
								Nr_BC_particles++;
							}
						} */
		}
	}

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	//---------------------------------------------------------------------

	global_tool.push_back(new tool_3d_gpu(cpu_tool, vec3_t(-vx, 0., 0. /*  -v_tool */), phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n + n_tool);
	assert(check_cuda_error());

	//---------------------------------------------------------------------

	float_t max_vel = max(v_tool, w.z * lx / 2.);
	max_vel = max(max_vel, w.z * ly / 2.);
	max_vel = max(max_vel, vx);
	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(max_vel) + c0);
	global_time_dt = 0.5 * delta_t_max;
	/* global_time_final = (lx_tool-lx)/max_vel;
	global_time_final = global_time_final * 2.0; */

	global_time_final = (0.5 * lz) / v_tool;

	// global_time_dt = global_time_dt/1000;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);

	//---------------------------------------------------------------------

	*grid = new grid_gpu_green(n + n_tool, make_float3_t(bbmin_tool.x - 15.0 * dz, bbmin_tool.y - 15.0 * dz, -1.5 * nz * dz), make_float3_t(bbmax_tool.x + 15.0 * dz, bbmax_tool.y + 15.0 * dz, 3.0 * bbmax_tool.z), hdx * dz);
	(*grid)->set_bbox_vel(make_float3_t(-vx, 0., v_tool));
	global_blanking = new blanking(vec3_t(bbmin_tool.x - 15.0 * dz, bbmin_tool.y - 15.0 * dz, -1.5 * nz * dz), vec3_t(bbmax_tool.x + 15.0 * dz, bbmax_tool.y + 15.0 * dz, 3.0 * bbmax_tool.z),
								   vec3_t(-vx, 0., v_tool), 2.0 * 2.0);

	//---------------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_substrate(trml_wp);
	actions_setup_thermal_constants_rod(trml_tool);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml_wp);
	interactions_setup_thermal_constants_rod(trml_tool, global_tool[0]);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_single_grain_1tool_trml_steady_si_units(grid_base **grid)
{

	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();
	joco_constants joco = make_joco_constants();
	rod_constants rod = make_rod_constants();

	bool sample_tool = true;

	float_t scale = 1.0;

	float_t vx = -1 * 6e-3 * 1000;
	float_t v_rod = (1.83e-3) * 1000;
	glm::vec3 w(0.0, 0.0, 125.663706);
	float_t friction_coeff = 0.35; // fric ruttiumann

	float_t rod_length = 6e-3 * 6; //*(1./6.)
	float_t dz = (1e-3);
	int nz = (rod_length / dz) + 1;

	printf("dz: %lf\n", dz);

	float_t lx = (20e-3);
	float_t ly = (20e-3);
	float_t lz = rod_length;
	float_t hdx = 1.7;

	auto cpu_tool = new tool_3d();
	vec3_t bbmin_tool(-lx, -ly, 0.0);
	vec3_t bbmax_tool(1.639344 * (6.0e-3) * 6 + lx, ly, 2e-3);

	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	float_t lz_tool = bbmax_tool.z - bbmin_tool.z;

	std::vector<vec3_t> samples;
	if (sample_tool)
	{

		samples = cpu_tool->sample(dz, bbmin_tool, bbmax_tool);
		// samples = cpu_tool->sample(dz);
	}

	int nx = (lx / dz) + 1; // equi particle spacing
	int ny = (ly / dz) + 1;

	int n = nx * ny * nz;
	int n_tool = samples.size();

	int part_iter = 0;
	float4_t *pos = new float4_t[n + n_tool];
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int k = 0; k < nz; k++)
			{
				float_t px = i * dz - lx / 2.;
				float_t py = j * dz - ly / 2.;
				float_t pz = -k * dz;

				float sq = px * px + py * py;
				if (sq < (lx * lx) / 4.0)
				{

					float4_t cur_pos;
					cur_pos.x = px;
					cur_pos.y = py;
					cur_pos.z = pz - 0.1 * dz;
					cur_pos.w = 0.0; // unneeded channel, used for marking the joined particles

					pos[part_iter] = cur_pos;

					part_iter++;
				}
			}
		}
	}

	n = part_iter;
	printf("calculating with %d regular particles\n", n);
	printf("calculating with %d tool particles\n", n_tool);

	for (auto &it : samples)
	{
		float4_t cur_pos;

		cur_pos.x = it.x;
		cur_pos.y = it.y;
		cur_pos.z = it.z;
		cur_pos.w = 1.0; // unneeded channel, used for marking the joined particles

		pos[part_iter] = cur_pos;

		part_iter++;
	}

	//---------------------------------------------------------------------
	rod.vz = v_rod;
	rod.wz = w.z;
	rod.lz = rod_length;
	rod.radius = lx / 2;

	phys.E = 70.3e9;	// 70.3 GPa
	phys.nu = 0.33;		// 0.33
	phys.rho0 = 2830.0; // 2830
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dz * dz * dz * phys.rho0;

	joco.A = 450.821e6; //	450.821 MPa
	joco.B = 108.537e6; //	108.537 MPA
	joco.C = 0.027;		//	0.027
	joco.m = 0.981;		//	0.981
	joco.n = 0.045;		//	0.045
	joco.Tref = 293.0;	//	323
	joco.Tmelt = 903.0; //	903 K
	joco.eps_dot_ref = 1;
	joco.clamp_temp = 1.;

	trml_wp.cp = 860.;									  // 860 J/kgC
	trml_wp.tq = 0.9;									  // Note used
	trml_wp.k = 157.;									  // 157.
	trml_wp.alpha = trml_wp.k / (phys.rho0 * trml_wp.cp); // varify it 0.3
	trml_wp.T_init = joco.Tref;
	trml_wp.eta = 0.9;

	trml_tool.cp = 860.; // 860 J/kgC
	trml_tool.tq = 0.9;	 // Note used
	trml_tool.k = 157.;	 // 157

	float_t rho0_tool = 2830.0; // 2830
	trml_tool.alpha = trml_tool.k / (rho0_tool * trml_tool.cp);
	trml_tool.T_init = joco.Tref;
	trml_tool.eta = 0.9;

	float_t c0 = sqrt(phys.K / phys.rho0);

	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
	}

	float4_t *vel = new float4_t[n + n_tool];
	float_t *h = new float_t[n + n_tool];
	float_t *rho = new float_t[n + n_tool];
	float_t *fixed = new float_t[n + n_tool];
	float_t *T = new float_t[n + n_tool];
	float_t *tool_p = new float_t[n + n_tool];

	for (int i = 0; i < n + n_tool; i++)
	{ // initialize particles
		rho[i] = phys.rho0;
		h[i] = hdx * dz;
		T[i] = joco.Tref;
		fixed[i] = 0;

		if (i >= n)
		{
			tool_p[i] = 1.;

			vel[i].x = 0.0;
			vel[i].y = 0.;
			vel[i].z = 0.;
		}
		else
		{
			tool_p[i] = 0.;

			glm::vec3 r(pos[i].x, pos[i].y, 0.0);
			glm::vec3 v = glm::cross(w, r);

			vel[i].x = v.x;
			vel[i].y = v.y;
			vel[i].z = v_rod;
			// vel[i].z =   0;
		}
	}

	unsigned int Nr_BC_particles = 0; // Counter boundary particles
	{
		for (unsigned int i = 0; i < n + n_tool; i++)
		{ // define boundary particles

			if (pos[i].z < -lz + (dz))
			{
				fixed[i] = 1;
				Nr_BC_particles++;
			}

			if (pos[i].z > lz_tool - (2 * dz))
			{
				fixed[i] = 2;
				Nr_BC_particles++;
			}
		}
	}

	printf("Number of BC-particles: %u \n", Nr_BC_particles);

	//---------------------------------------------------------------------

	vec3_t cc(vx, 0., 0.);
	global_tool.push_back(new tool_3d_gpu(cc, phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n + n_tool);
	assert(check_cuda_error());

	//---------------------------------------------------------------------

	float_t max_vel = max(v_rod, w.z * lx / 2.);
	max_vel = max(max_vel, w.z * ly / 2.);
	max_vel = max(max_vel, vx);
	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(max_vel) + c0);
	global_time_dt = 0.5 * delta_t_max;

	// hard coding delta t
	// global_time_dt = 1e-7;
	// global_time_final = (lz)/v_rod;
	global_time_final = 3.5;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);
	printf("global_time_final %e \n", global_time_final);

	//---------------------------------------------------------------------

	*grid = new grid_gpu_green(n + n_tool, make_float3_t(bbmin_tool.x - 2.0 * dz, bbmin_tool.y - 2.0 * dz, -1.1 * nz * dz), make_float3_t(bbmax_tool.x + 2.0 * dz, bbmax_tool.y + 2.0 * dz, 1.1 * bbmax_tool.z), hdx * dz);
	(*grid)->set_bbox_vel(make_float3_t(vx, 0., v_rod));
	/* 	(*grid)->set_max_fixed(true);
		(*grid)->set_min_fixed(true); */
	global_blanking = new blanking(vec3_t(bbmin_tool.x - 2 * dz, bbmin_tool.y - 2.0 * dz, -1.1 * nz * dz), vec3_t(bbmax_tool.x + 2 * dz, bbmax_tool.y + 2.0 * dz, 1.1 * bbmax_tool.z),
								   vec3_t(vx, 0., v_rod), 200.0 * 200.0);

	/* global_blanking->set_max_fixed(true);
	global_blanking->set_min_fixed(true); */

	//---------------------------------------------------------------------

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco);
	actions_setup_thermal_constants_substrate(trml_wp);
	actions_setup_thermal_constants_rod(trml_tool);
	actions_setup_rod_constants(rod);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml_wp);
	interactions_setup_thermal_constants_rod(trml_tool, global_tool[0]);
	interactions_setup_geometry_constants(*grid);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_single_grain_1tool_trml_steady_si_units_scaling_down(grid_base **grid)
{

	printf("ATTENTION \n");
	printf("Did you Activate the Merge Method? \n");
	printf("Did you change the boundary condition method ?\n");
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_sub = make_trml_constants();
	trml_constants trml_rod = make_trml_constants();
	joco_constants joco_sub = make_joco_constants();
	joco_constants joco_rod = make_joco_constants();
	rod_constants rod = make_rod_constants();

	bool sample_tool = true;

	// scaling factors
	float_t Lsf = 1.0; // lenth scale factor
	float_t Vsf = 1.;  // velocity scake factor

	// Dimensions
	// Rod Dimensions
	float_t rod_diameter = 7.8e-3 * Lsf;
	// float_t rod_length   =  120.0e-3 * Lsf ;
	float_t rod_length = 0.5 * 23.0e-3 * Lsf;

	// Substrate Dimensions
	float_t lx = 2.0e-2 * Lsf;
	float_t ly = 2.0e-2 * Lsf;
	rod.lz_tool = 6.0e-3;
	float_t lz = 6.0e-3 * Lsf; // 12.5 for backing plate and 8 for the substrate 21.0e-3
	vec3_t bbmin_tool(-lx, -ly, 0.0);
	vec3_t bbmax_tool(lx, ly, lz);
	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	float_t lz_tool = bbmax_tool.z - bbmin_tool.z;

	// Velocities
	float_t vx = -1 * 0.0 * Vsf;
	float_t v_rod = (718) * Vsf;
	glm::vec3 w(0.0, 0.0, 0.0 * Vsf);

	// discretization
	float_t dz = 0.5 * (1e-3) * Lsf;
	float_t hdx = 1.7;
	int nz = (rod_length / dz) + 1;

	// Create a material
	// Al5083H116 al5083_Kath(Vsf,dz);
	Al5083_Katherine al5083_Kath(Vsf, dz);

	phys = al5083_Kath.phys; /////////////////////////////////////////////////////////////////

	trml_sub = al5083_Kath.trml;
	joco_sub = al5083_Kath.joco;

	trml_rod = al5083_Kath.trml;
	joco_rod = al5083_Kath.joco;

	// Phys constants
	float_t friction_coeff = 0.35; // fric ruttiumann
	float_t c0 = sqrt(phys.K / phys.rho0);

	// SPH correction factors
	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.01;
	corr.stresseps = 0.3;

	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
		printf("Cubic spline Kernel \n");
	}

	auto cpu_tool = new tool_3d();
	std::vector<vec3_t> samples;
	if (sample_tool)
	{
		samples = cpu_tool->sample(dz, bbmin_tool, bbmax_tool);
	}

	int nx = (rod_diameter / dz) + 1; // equi particle spacing
	int ny = (rod_diameter / dz) + 1;

	int n = nx * ny * nz;
	int n_tool = samples.size();

	int count = 0;
	int part_iter = 0;
	float4_t *pos = new float4_t[n + n_tool];
	for (int k = 0; k < nz; k++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{

				float_t px = i * dz - rod_diameter / 2.;
				float_t py = j * dz - rod_diameter / 2.;
				float_t pz = -k * dz;

				float sq = px * px + py * py;
				if (sq < (rod_diameter * rod_diameter) / 4.0)
				{

					float4_t cur_pos;
					cur_pos.x = px;
					cur_pos.y = py;
					cur_pos.z = pz - 5 * dz;
					cur_pos.w = 0.0; // unneeded channel, used for marking the joined particles

					pos[part_iter] = cur_pos;

					part_iter++;

					if (k == 0)
						count++;
				}
			}
		}
	}

	n = part_iter;
	printf("calculating with %d Rod particles\n", n);
	printf("calculating with %d Substrate particles\n", n_tool);
	printf("calculating with %d TOTAL particles\n", n + n_tool);

	// initiating global rod variable
	rod.vz = v_rod;
	rod.vx = vx;
	rod.wz = w.z;
	rod.lz = rod_length;
	rod.radius = rod_diameter / 2.;
	rod.dz = dz;
	rod.move_sub = 0.0;
	rod.Vsf = Vsf;

	for (auto &it : samples)
	{
		float4_t cur_pos;
		cur_pos.x = it.x;
		cur_pos.y = it.y;
		cur_pos.z = it.z;
		cur_pos.w = 1.0; // unneeded channel, used for marking the joined particles
		pos[part_iter] = cur_pos;
		part_iter++;
	}

	float4_t *vel = new float4_t[n + n_tool];
	float_t *h = new float_t[n + n_tool];
	float_t *rho = new float_t[n + n_tool];
	float_t *fixed = new float_t[n + n_tool];
	float_t *T = new float_t[n + n_tool];
	float_t *tool_p = new float_t[n + n_tool];
	float_t *cp = new float_t[n + n_tool];
	float_t *H = new float_t[n + n_tool];
	float_t *k = new float_t[n + n_tool];
	float_t Tc_sub = joco_sub.Tmelt - 273; // converting to C
	float_t Tc_rod = joco_rod.Tmelt - 273; // converting to C
	for (int i = 0; i < n + n_tool; i++)
	{ // initialize particles
		rho[i] = phys.rho0;

		fixed[i] = 0;
		h[i] = hdx * dz;

		if (i >= n)
		{
			tool_p[i] = 1.;

			cp[i] = trml_sub.cp0 + trml_sub.cp1 * Tc_sub + trml_sub.cp2 * Tc_sub * Tc_sub;
			k[i] = trml_sub.k0 + trml_sub.k1 * Tc_sub + trml_sub.k2 * Tc_sub * Tc_sub + trml_sub.k3 * Tc_sub * Tc_sub * Tc_sub;

			H[i] = cp[i] * (joco_sub.Tref - 273);
			T[i] = joco_sub.Tref;

			vel[i].x = 0.;
			vel[i].y = 0.;
			vel[i].z = 0.;

			float_t x = pos[i].x;
			float_t y = pos[i].y;
			float_t z = pos[i].z;
			float_t radius = rod.radius;
			fixed[i] = 0;
			if (abs(y) > ly_tool / 2. - 0.5 * dz || abs(x) > bbmax_tool.x - 0.5 * dz)
				fixed[i] = 3;
		}
		else
		{
			tool_p[i] = 0.;

			cp[i] = trml_rod.cp0 + trml_rod.cp1 * Tc_rod + trml_rod.cp2 * Tc_rod * Tc_rod;
			k[i] = trml_rod.k0 + trml_rod.k1 * Tc_rod + trml_rod.k2 * Tc_rod * Tc_rod + trml_rod.k3 * Tc_rod * Tc_rod * Tc_rod;

			H[i] = cp[i] * (joco_rod.Tref - 273);
			T[i] = joco_rod.Tref;

			glm::vec3 r(pos[i].x, pos[i].y, 0.0);
			glm::vec3 v = glm::cross(w, r);

			vel[i].x = v.x;
			vel[i].y = v.y;
			vel[i].z = v_rod;
			fixed[i] = 0;
		}
	}

	//---------------------------------------------------------------------

	vec3_t cc(vx, 0., 0.);
	global_tool.push_back(new tool_3d_gpu(cc, phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);
	global_tool[0]->set_rod(rod);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n + n_tool, cp, H, k);
	assert(check_cuda_error());

	//---------------------------------------------------------------------

	float_t max_vel = max(v_rod, w.z * (rod_diameter / 2.));
	max_vel = max(max_vel, v_rod);
	max_vel = max(max_vel, abs(vx));
	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(max_vel) + c0);
	global_time_dt = 0.5 * delta_t_max;

	global_time_final = 0.0002;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);
	printf("global_time_final %e \n", global_time_final);
	printf("C0 = %lf \n", c0);

	//------------------------------------------------------------------------

	*grid = new grid_gpu_green(n + n_tool, make_float3_t(bbmin_tool.x - 5.0 * dz, bbmin_tool.y - 5.0 * dz, -5.0 * nz * dz),
							   make_float3_t(bbmax_tool.x + 5.0 * dz,
											 bbmax_tool.y + 5.0 * dz,
											 5.0 * bbmax_tool.z),
							   hdx * dz);

	(*grid)->set_bbox_vel(make_float3_t(0., 0., 0.));

	global_blanking = new blanking(vec3_t(bbmin_tool.x - 5.0 * dz, bbmin_tool.y - 5.0 * dz, -5.0 * nz * dz),
								   vec3_t(bbmax_tool.x + 5.0 * dz, bbmax_tool.y + 5.0 * dz, 5.0 * bbmax_tool.z),
								   vec3_t(0., 0., 0.), 20000 * 20000);

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco_sub, joco_rod);
	actions_setup_thermal_constants_substrate(trml_sub);
	actions_setup_thermal_constants_rod(trml_rod);
	actions_setup_rod_constants(rod);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml_sub);
	interactions_setup_thermal_constants_rod(trml_rod, global_tool[0]);
	interactions_setup_geometry_constants(*grid);
	interactions_setup_rod_constants(rod);

	interactions_setup_johnson_cook_constants(joco_sub, joco_rod);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

void generate_circular_arrangement(float_t n, float_t r, float_t z, std::vector<Point> &points)
{

	if (r == 0)
	{
		Point point;
		point.x = r;
		point.y = r;
		point.z = z;
		points.push_back(point);
	}

	for (int i = 0; i < n; i++)
	{
		double angle = 2 * M_PI * i / n;
		Point point;
		point.x = r * cos(angle);
		point.y = r * sin(angle);
		point.z = z;
		points.push_back(point);
	}
}
void generate_square_arrangement(float_t spacing, float_t side_length, float_t z, std::vector<Point> &points)
{

	if (side_length == 0)
	{
		Point point;
		point.x = side_length;
		point.y = side_length;
		point.z = z;
		points.push_back(point);
	}

	int n = (side_length / spacing) + 1;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
			{
				Point point;
				point.x = i * spacing - side_length / 2;
				point.y = j * spacing - side_length / 2;
				point.z = z;
				points.push_back(point);
			}
		}
	}
}

float_t distance(Point a, Point b)
{

	float deff_x = a.x - b.x;
	float deff_y = a.y - b.y;
	return sqrt(deff_x * deff_x + deff_y * deff_y);
}

particle_gpu *setup_FSE_Cylinder(grid_base **grid, float_t &p_mass)
{

	printf("ATTENTION \n");
	printf("Did you Activate the Merge Method? \n");
	printf("Did you change the boundary condition method ?\n");
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_sub = make_trml_constants();
	trml_constants trml_rod = make_trml_constants();
	joco_constants joco_sub = make_joco_constants();
	joco_constants joco_rod = make_joco_constants();
	rod_constants rod = make_rod_constants();

	// scaling factors
	float_t Lsf = 1.0;		   // lenth scale factor
	float_t VsfM = 1.0;		   // 1000 failure occured at ~15%
	float_t Vsf = 15.0 * VsfM; // velocity scale factor

	// discretization
	float_t dz = (0.25e-3) * Lsf;
	float_t hdx = 1.7;

	float_t exp_scalling = 0.5;
	// Dimensions
	float_t top_die_diameter = 25.0e-3 * Lsf;
	float_t orifice = 2.5e-3 * Lsf;
	float_t die_length = 10.0e-3 * Lsf;
	float_t container_thickness = 1.0e-3 * Lsf;
	float_t container_length = 19.e-3 * Lsf * exp_scalling;
	float_t billet_length = 13.0e-3 * Lsf * exp_scalling;
	float_t billet_diameter = top_die_diameter;
	float_t free_length = container_length - billet_length - die_length;
	float_t disc_inner_dim = 12.7e-3 * Lsf;
	float_t sub_die = 1.0e-3 * Lsf;

	// Assymbly dimensions
	float_t lx = billet_diameter + 2.0 * container_thickness;
	float_t ly = billet_diameter + 2.0 * container_thickness;
	float_t lz = container_thickness + billet_length + die_length;
	vec3_t bbmin_tool(-lx / 2., -ly / 2., 0.0);
	vec3_t bbmax_tool(lx / 2., ly / 2., lz);
	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	float_t lz_tool = bbmax_tool.z - bbmin_tool.z;

	// Velocities
	float_t vx = 0.0 * Vsf;
	float_t v_die = (-8.0e-3) / 60. * Vsf;
	glm::vec3 w(0.0, 0.0, 31.41592653589793 * Vsf);

	int nz = (lz / dz) + 1;

	// Phys constants
	float_t friction_coeff = 0.35; // fric ruttiumann

	std::vector<Point> points;
	float_t zz = 0;
	while (zz < lz)
	{

		if (zz < container_length)
		{

			float_t r = 0.;
			while (r < lx / 2.)
			{
				if ((zz > container_thickness + billet_length) && r < orifice / 2.)
				{
					r += dz;
					continue;
				}
				int n = ((2 * M_PI * r) / dz);
				generate_circular_arrangement(n, r, zz, points);
				r += dz;
			}
		}
		else
		{

			float_t r = 0.;

			while (r < top_die_diameter / 2.)
			{

				if ((zz > container_thickness + billet_length + die_length / 2.) && r < disc_inner_dim / 2.)
				{
					r += dz;
					continue;
				}
				if (r < orifice / 2.)
				{
					r += dz;
					continue;
				}

				int n = ((2 * M_PI * r) / dz);
				generate_circular_arrangement(n, r, zz, points);
				r += dz;
			}
		}

		zz += dz;
	}

	// vtk_simple_write(points);

	// exit(-1);
	//********************************************************* Create a material ************************************************************/

	// AA1100 al_AA1100(Vsf,dz);
	AA1100 al_AA1100(Vsf, dz, 1);

	phys = al_AA1100.phys;
	float_t c0 = sqrt(phys.K / phys.rho0);

	trml_sub = al_AA1100.trml;
	joco_sub = al_AA1100.joco;

	trml_rod = al_AA1100.trml;
	joco_rod = al_AA1100.joco;

	p_mass = phys.mass;

	//********************************************************* SPH correction factors ************************************************************/
	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.01;
	corr.stresseps = 0.3;

	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
		printf("Cubic spline Kernel \n");
	}

	int n = points.size();

	float4_t *pos = new float4_t[n];
	for (int i = 0; i < n; i++)
	{
		pos[i].x = points[i].x;
		pos[i].y = points[i].y;
		pos[i].z = points[i].z;
	}

	printf("calculating with %d top die particles\n", n);

	// initiating global rod variable
	rod.vz = v_die;
	rod.vx = vx;
	rod.wz = w.z;
	rod.lz = container_thickness + billet_length;
	rod.radius = top_die_diameter / 2.;
	rod.dz = dz;
	rod.lz_tool = container_thickness;
	rod.Vsf = Vsf;

	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *fixed = new float_t[n];
	float_t *T = new float_t[n];
	float_t *tool_p = new float_t[n];
	float_t *cp = new float_t[n];
	float_t *H = new float_t[n];
	float_t *k = new float_t[n];
	float_t Tc_sub = joco_sub.Tmelt - 273; // converting to C
	float_t Tc_rod = joco_rod.Tmelt - 273; // converting to C

	for (int i = 0; i < n; i++)
	{ // initialize particles

		rho[i] = phys.rho0;
		h[i] = hdx * dz;

		float_t px = pos[i].x;
		float_t py = pos[i].y;
		float_t pz = pos[i].z;
		float sq = px * px + py * py;

		// draw the container only
		float_t inner_dimeter = lx - 2.5 * container_thickness;
		float_t outer_dimeter = lx;

		if (sq > (inner_dimeter * inner_dimeter) / 4.0 || abs(pz) < container_thickness)
		{
			// container
			tool_p[i] = 2.;
			fixed[i] = 4; // for deformable part

			cp[i] = trml_sub.cp0 + trml_sub.cp1 * Tc_sub + trml_sub.cp2 * Tc_sub * Tc_sub;
			k[i] = trml_sub.k0 + trml_sub.k1 * Tc_sub + trml_sub.k2 * Tc_sub * Tc_sub + trml_sub.k3 * Tc_sub * Tc_sub * Tc_sub;
			H[i] = cp[i] * (joco_sub.Tref - 273);
			T[i] = joco_sub.Tref;

			float_t x = pos[i].x;
			float_t y = pos[i].y;
			float_t z = pos[i].z;

			vel[i].x = 0.;
			vel[i].y = 0.;
			vel[i].z = 0.;
			float_t BCD = outer_dimeter - 2.5 * dz;
			if (sq >= (BCD * BCD) / 4.0 || abs(pz) < dz)
				fixed[i] = 7;
		}
		else
		{
			if (abs(pz) >= container_thickness + billet_length)
			{
				// Draw the die
				tool_p[i] = 0.; // for the top die
				if (abs(pz) < container_thickness + billet_length + 1e-3)
					fixed[i] = 5; // 4 or bigger is used for rigid body
				else
					fixed[i] = 6; // 5 or bigger is used for rigid body

				cp[i] = trml_rod.cp0 + trml_rod.cp1 * Tc_rod + trml_rod.cp2 * Tc_rod * Tc_rod;
				k[i] = trml_rod.k0 + trml_rod.k1 * Tc_rod + trml_rod.k2 * Tc_rod * Tc_rod + trml_rod.k3 * Tc_rod * Tc_rod * Tc_rod;

				H[i] = cp[i] * (joco_rod.Tref - 273);
				T[i] = joco_rod.Tref;

				glm::vec3 r(pos[i].x, pos[i].y, 0.0);
				glm::vec3 v = glm::cross(w, r);

				vel[i].x = v.x;
				vel[i].y = v.y;
				vel[i].z = v_die;
			}
			else
			{
				// billet
				tool_p[i] = 1.; // for the deformable billet
				fixed[i] = 0;	// for deformable part

				cp[i] = trml_sub.cp0 + trml_sub.cp1 * Tc_sub + trml_sub.cp2 * Tc_sub * Tc_sub;
				k[i] = trml_sub.k0 + trml_sub.k1 * Tc_sub + trml_sub.k2 * Tc_sub * Tc_sub + trml_sub.k3 * Tc_sub * Tc_sub * Tc_sub;
				H[i] = cp[i] * (joco_sub.Tref - 273);
				T[i] = joco_sub.Tref;

				float_t x = pos[i].x;
				float_t y = pos[i].y;
				float_t z = pos[i].z;

				vel[i].x = 0.;
				vel[i].y = 0.;
				vel[i].z = 0.;
			}
		}
	}

	// calculate total mass of the bullet

	/* int billet_count = 0;
	for(int i =0; i<n; i++){
		if(tool_p[i] == 1) billet_count++;
	} */
	// total_mass = phys.mass  * billet_count;

	/* printf("billet_count = %d\n", billet_count);
	printf("phys.mass = %.20lf\n", phys.mass);
	printf("total_mass = %lf\n", total_mass);
	float_t masss = 0.25 * M_PI * billet_diameter * billet_diameter * billet_length * phys.rho0;
	printf("mmmmmmmmmm = %lf\n", masss);
 */

	// Adding temperature gradient  for the billet
	// 600K, 560K, 520K, 480K and 440K, respectively

	/* 	for(int i=0; i<n; i++){
			if(tool_p[i] == 1){

				//T[i] = 823;
				 float_t pz =  points[i].z;

				if   (pz < 1*(billet_length/5) + container_thickness ) T[i] = 440;
				else if(pz < 2*(billet_length/5) + container_thickness ) T[i] = 480;
				else if(pz < 3*(billet_length/5) + container_thickness ) T[i] = 520;
				else if(pz < 4*(billet_length/5) + container_thickness ) T[i] = 560;
				else  T[i] = 600;

			}
		}     */

	//---------------------------------------------------------------------

	vec3_t cc(vx, 0., 0.);
	global_tool.push_back(new tool_3d_gpu(cc, phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);
	global_tool[0]->set_rod(rod);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n, cp, H, k);
	assert(check_cuda_error());
	//---------------------------------------------------------------------

	float_t max_vel = max(v_die, w.z * (top_die_diameter / 2.));
	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(max_vel) + c0);
	global_time_dt = 0.5 * delta_t_max;

	global_time_final = (1.0 / VsfM); ///////////////////////////////////////////////////////////////////////////////////////

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);
	printf("global_time_final %e \n", global_time_final);
	printf("C0 = %lf \n", c0);

	//------------------------------------------------------------------------

	*grid = new grid_gpu_green(n, make_float3_t(bbmin_tool.x - 2.0 * dz, bbmin_tool.y - 2.0 * dz, -2. * dz), make_float3_t(bbmax_tool.x + 2.0 * dz, bbmax_tool.y + 2.0 * dz, 2. * bbmax_tool.z), hdx * dz);
	(*grid)->set_bbox_vel(make_float3_t(vx, 0., v_die));

	global_blanking = new blanking(vec3_t(bbmin_tool.x - 2 * dz, bbmin_tool.y - 2.0 * dz, -2. * dz),
								   vec3_t(bbmax_tool.x + 2 * dz, bbmax_tool.y + 2.0 * dz, 2. * bbmax_tool.z),
								   vec3_t(vx, 0., v_die), 2000 * 2000);

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco_sub, joco_rod);
	actions_setup_thermal_constants_substrate(trml_sub);
	actions_setup_thermal_constants_rod(trml_rod);
	actions_setup_rod_constants(rod);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml_sub);
	interactions_setup_thermal_constants_rod(trml_rod, global_tool[0]);
	interactions_setup_geometry_constants(*grid);
	interactions_setup_rod_constants(rod);

	interactions_setup_johnson_cook_constants(joco_sub, joco_rod);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_FSE(grid_base **grid)
{

	printf("ATTENTION \n");
	printf("Did you Activate the Merge Method? \n");
	printf("Did you change the boundary condition method ?\n");
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_sub = make_trml_constants();
	trml_constants trml_rod = make_trml_constants();
	joco_constants joco_sub = make_joco_constants();
	joco_constants joco_rod = make_joco_constants();
	rod_constants rod = make_rod_constants();

	// scaling factors
	float_t Lsf = 1.0;		   // lenth scale factor
	float_t VsfM = 100.0;	   // 1000 failure occured at ~15%
	float_t Vsf = 15.0 * VsfM; // velocity scale factor

	// discretization
	float_t dz = (0.25e-3) * Lsf;
	float_t hdx = 1.7;

	float_t exp_scalling = 0.5;
	// Dimensions
	float_t top_die_diameter = 25.0e-3 * Lsf;
	float_t orifice = 2.5e-3 * Lsf;
	float_t die_length = 10.0e-3 * Lsf;
	float_t container_thickness = 1.0e-3 * Lsf;
	float_t container_length = 19.e-3 * Lsf * exp_scalling;
	float_t billet_length = 13.0e-3 * Lsf * exp_scalling;
	float_t billet_diameter = top_die_diameter;
	float_t free_length = container_length - billet_length - die_length;
	float_t disc_inner_dim = 12.7e-3 * Lsf;

	// Assymbly dimensions
	float_t lx = billet_diameter + 2.0 * container_thickness;
	float_t ly = billet_diameter + 2.0 * container_thickness;
	float_t lz = container_thickness + billet_length + die_length + dz;
	vec3_t bbmin_tool(-lx / 2., -ly / 2., 0.0);
	vec3_t bbmax_tool(lx / 2., ly / 2., lz);
	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	float_t lz_tool = bbmax_tool.z - bbmin_tool.z;

	// Velocities
	float_t vx = 0.0 * Vsf;
	float_t v_die = (-8.0e-3) / 60. * Vsf;
	glm::vec3 w(0.0, 0.0, 31.41592653589793 * Vsf);

	int nz = (lz / dz) + 1;

	// Phys constants
	float_t friction_coeff = 0.35; // fric ruttiumann

	//********************************************************* Create a material ************************************************************/

	AA1100 al_AA1100(Vsf, dz, 1.);

	phys = al_AA1100.phys;
	float_t c0 = sqrt(phys.K / phys.rho0);

	trml_sub = al_AA1100.trml;
	joco_sub = al_AA1100.joco;

	trml_rod = al_AA1100.trml;
	joco_rod = al_AA1100.joco;

	//********************************************************* SPH correction factors ************************************************************/
	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.01;
	corr.stresseps = 0.3;

	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
		printf("Cubic spline Kernel \n");
	}

	int nx = (lx / dz) + 1; // equi particle spacing
	int ny = (ly / dz) + 1;

	int n = nx * ny * nz;

	int part_iter = 0;
	float4_t *pos = new float4_t[n];

	for (int k = 0; k < nz; k++)
	{
		for (int j = -ny; j < ny; j++)
		{
			for (int i = -nx; i < nx; i++)
			{

				float_t px = i * dz;
				float_t py = j * dz;
				float_t pz = k * dz;

				float sq = px * px + py * py;

				// draw the container only
				float_t inner_dimeter = lx - 2 * container_thickness;
				float_t outer_dimeter = lx;

				if (pz > container_length)
				{

					if (sq < (inner_dimeter * inner_dimeter) / 4.0)
					{

						// container thickness and billet
						if (abs(pz) < container_thickness + billet_length)
						{
							float4_t cur_pos;
							cur_pos.x = px;
							cur_pos.y = py;
							cur_pos.z = pz;
							cur_pos.w = 0.0; // unneeded channel, used for marking the joined particles
							pos[part_iter] = cur_pos;
							part_iter++;
						}
						else
						{
							// die
							if (pz < container_thickness + billet_length + die_length / 2.)
							{
								if (sq > (orifice * orifice) / 4.0)
								{

									float4_t cur_pos;
									cur_pos.x = px;
									cur_pos.y = py;
									cur_pos.z = pz;
									cur_pos.w = 0.0; // unneeded channel, used for marking the joined particles

									pos[part_iter] = cur_pos;

									part_iter++;
								}
							}
							else
							{
								if (sq > (disc_inner_dim * disc_inner_dim) / 4.0)
								{

									float4_t cur_pos;
									cur_pos.x = px;
									cur_pos.y = py;
									cur_pos.z = pz;
									cur_pos.w = 0.0; // unneeded channel, used for marking the joined particles

									pos[part_iter] = cur_pos;

									part_iter++;
								}
							}
						}
					}
				}
				else
				{

					if (sq < (outer_dimeter * outer_dimeter) / 4.0)
					{

						// container thickness and billet
						if (pz < container_thickness + billet_length)
						{
							float4_t cur_pos;
							cur_pos.x = px;
							cur_pos.y = py;
							cur_pos.z = pz;
							/* if(sq< (inner_dimeter * inner_dimeter)/4.0 ) cur_pos.z = pz -dz;
							else 										 cur_pos.z = pz ;
							if( sq < (inner_dimeter * inner_dimeter)/4.0  &&
								sq > ((inner_dimeter-2*dz )* (inner_dimeter-2*dz))/4.0
								&& pz > container_thickness) continue;
							if (cur_pos.z < 0) continue; */
							cur_pos.w = 0.0; // unneeded channel, used for marking the joined particles
							pos[part_iter] = cur_pos;
							part_iter++;
						}
						else
						{
							// die
							if (pz < container_thickness + billet_length + die_length / 2.)
							{
								if (sq > (orifice * orifice) / 4.0)
								{

									float4_t cur_pos;
									cur_pos.x = px;
									cur_pos.y = py;
									cur_pos.z = pz;
									cur_pos.w = 0.0; // unneeded channel, used for marking the joined particles

									pos[part_iter] = cur_pos;

									part_iter++;
								}
							}
							else
							{
								if (sq > (disc_inner_dim * disc_inner_dim) / 4.0)
								{

									float4_t cur_pos;
									cur_pos.x = px;
									cur_pos.y = py;
									cur_pos.z = pz;
									cur_pos.w = 0.0; // unneeded channel, used for marking the joined particles

									pos[part_iter] = cur_pos;

									part_iter++;
								}
							}
						}
					}
				}
			}
		}
	}

	n = part_iter;
	printf("calculating with %d top die particles\n", n);

	// initiating global rod variable
	rod.vz = v_die;
	rod.vx = vx;
	rod.wz = w.z;
	rod.lz = container_thickness + billet_length;
	rod.radius = top_die_diameter / 2.;
	rod.dz = dz;
	rod.lz_tool = container_thickness;
	rod.Vsf = Vsf;

	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *fixed = new float_t[n];
	float_t *T = new float_t[n];
	float_t *tool_p = new float_t[n];
	float_t *cp = new float_t[n];
	float_t *H = new float_t[n];
	float_t *k = new float_t[n];
	float_t Tc_sub = joco_sub.Tmelt - 273; // converting to C
	float_t Tc_rod = joco_rod.Tmelt - 273; // converting to C

	for (int i = 0; i < n; i++)
	{ // initialize particles

		rho[i] = phys.rho0;
		h[i] = hdx * dz;

		float_t px = pos[i].x;
		float_t py = pos[i].y;
		float_t pz = pos[i].z;
		float sq = px * px + py * py;

		// draw the container only
		float_t inner_dimeter = lx - 2 * container_thickness;
		float_t outer_dimeter = lx;

		if (sq >= (inner_dimeter * inner_dimeter) / 4.0 || abs(pz) < container_thickness)
		{
			// container
			tool_p[i] = 2.;
			fixed[i] = 4; // for deformable part

			cp[i] = trml_sub.cp0 + trml_sub.cp1 * Tc_sub + trml_sub.cp2 * Tc_sub * Tc_sub;
			k[i] = trml_sub.k0 + trml_sub.k1 * Tc_sub + trml_sub.k2 * Tc_sub * Tc_sub + trml_sub.k3 * Tc_sub * Tc_sub * Tc_sub;
			H[i] = cp[i] * (joco_sub.Tref - 273);
			T[i] = joco_sub.Tref;

			float_t x = pos[i].x;
			float_t y = pos[i].y;
			float_t z = pos[i].z;

			vel[i].x = 0.;
			vel[i].y = 0.;
			vel[i].z = 0.;
			float_t BCD = outer_dimeter - 2. * dz;
			if (sq >= (BCD * BCD) / 4.0 || abs(pz) < dz)
				fixed[i] = 7;
		}
		else
		{
			if (abs(pz) >= container_thickness + billet_length)
			{
				// Draw the die
				tool_p[i] = 0.; // for the top die
				if (abs(pz) < container_thickness + billet_length + 1e-3)
					fixed[i] = 5; // 4 or bigger is used for rigid body
				else
					fixed[i] = 6; // 5 or bigger is used for rigid body

				cp[i] = trml_rod.cp0 + trml_rod.cp1 * Tc_rod + trml_rod.cp2 * Tc_rod * Tc_rod;
				k[i] = trml_rod.k0 + trml_rod.k1 * Tc_rod + trml_rod.k2 * Tc_rod * Tc_rod + trml_rod.k3 * Tc_rod * Tc_rod * Tc_rod;

				H[i] = cp[i] * (joco_rod.Tref - 273);
				T[i] = joco_rod.Tref;

				glm::vec3 r(pos[i].x, pos[i].y, 0.0);
				glm::vec3 v = glm::cross(w, r);

				vel[i].x = v.x;
				vel[i].y = v.y;
				vel[i].z = v_die;
			}
			else
			{
				// billet
				tool_p[i] = 1.; // for the deformable billet
				fixed[i] = 0;	// for deformable part

				cp[i] = trml_sub.cp0 + trml_sub.cp1 * Tc_sub + trml_sub.cp2 * Tc_sub * Tc_sub;
				k[i] = trml_sub.k0 + trml_sub.k1 * Tc_sub + trml_sub.k2 * Tc_sub * Tc_sub + trml_sub.k3 * Tc_sub * Tc_sub * Tc_sub;
				H[i] = cp[i] * (joco_sub.Tref - 273);
				T[i] = joco_sub.Tref;

				float_t x = pos[i].x;
				float_t y = pos[i].y;
				float_t z = pos[i].z;

				vel[i].x = 0.;
				vel[i].y = 0.;
				vel[i].z = 0.;
			}
		}
	}

	// Adding temperature gradient  for the billet
	// 600K, 560K, 520K, 480K and 440K, respectively
	for (int i = 0; i < n; i++)
	{
		if (tool_p[i] == 1)
		{

			float_t pz = pos[i].z;

			if (pz < 1 * (billet_length / 5) + container_thickness)
				T[i] = 440;
			else if (pz < 2 * (billet_length / 5) + container_thickness)
				T[i] = 480;
			else if (pz < 3 * (billet_length / 5) + container_thickness)
				T[i] = 520;
			else if (pz < 4 * (billet_length / 5) + container_thickness)
				T[i] = 560;
			else
				T[i] = 600;
		}
	}

	int billet_count = 0;
	for (int i = 0; i < n; i++)
	{
		if (tool_p[i] == 1)
			billet_count++;
	}

	float_t total_mass = dz * dz * dz * phys.rho0 * billet_count;
	printf("billet_count = %d\n", billet_count);
	printf("phys.mass = %.20lf\n", phys.mass);
	printf("total_mass = %.20lf\n", total_mass);

	float_t masss = 0.25 * 3.14 * billet_diameter * billet_diameter * billet_length * phys.rho0;
	float_t volume = 0.25 * 3.14 * billet_diameter * billet_diameter * billet_length;
	printf("mmmmmmmmmm = %lf\n", masss);
	/* printf("volume_PI = %lf\n", volume);
	printf("volume_dz = %lf\n", billet_count* dz*dz*dz); */

	//---------------------------------------------------------------------

	vec3_t cc(vx, 0., 0.);
	global_tool.push_back(new tool_3d_gpu(cc, phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);
	global_tool[0]->set_rod(rod);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n, cp, H, k);
	assert(check_cuda_error());
	//---------------------------------------------------------------------

	float_t max_vel = max(v_die, w.z * (top_die_diameter / 2.));
	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(max_vel) + c0);
	global_time_dt = 0.5 * delta_t_max;

	global_time_final = (1.0 / VsfM) * 0.1;

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);
	printf("global_time_final %e \n", global_time_final);
	printf("C0 = %lf \n", c0);

	//------------------------------------------------------------------------

	*grid = new grid_gpu_green(n, make_float3_t(bbmin_tool.x - 2.0 * dz, bbmin_tool.y - 2.0 * dz, -2. * dz), make_float3_t(bbmax_tool.x + 2.0 * dz, bbmax_tool.y + 2.0 * dz, 2. * bbmax_tool.z), hdx * dz);
	(*grid)->set_bbox_vel(make_float3_t(vx, 0., v_die));

	global_blanking = new blanking(vec3_t(bbmin_tool.x - 2 * dz, bbmin_tool.y - 2.0 * dz, -2. * dz),
								   vec3_t(bbmax_tool.x + 2 * dz, bbmax_tool.y + 2.0 * dz, 2. * bbmax_tool.z),
								   vec3_t(vx, 0., v_die), 2000 * 2000);

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco_sub, joco_rod);
	actions_setup_thermal_constants_substrate(trml_sub);
	actions_setup_thermal_constants_rod(trml_rod);
	actions_setup_rod_constants(rod);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml_sub);
	interactions_setup_thermal_constants_rod(trml_rod, global_tool[0]);
	interactions_setup_geometry_constants(*grid);
	interactions_setup_rod_constants(rod);

	interactions_setup_johnson_cook_constants(joco_sub, joco_rod);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_FSE_Cylinder_gcms(grid_base **grid, float_t &p_mass)
{

	printf("ATTENTION \n");
	printf("Did you Activate the Merge Method? \n");
	printf("Did you change the boundary condition method ?\n");
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_sub = make_trml_constants();
	trml_constants trml_rod = make_trml_constants();
	joco_constants joco_sub = make_joco_constants();
	joco_constants joco_rod = make_joco_constants();
	rod_constants rod = make_rod_constants();

	// scaling factors
	float_t Lsf = 1.; // lenth scale factor
	float_t ms = 3.9e3;
	float_t VsfM = 1.;		  // 1000 failure occured at ~15%
	float_t Vsf = 10. * VsfM; // velocity scale factor /////////////////////////////////////////////////
	global_Vsf = Vsf;

	// discretization
	// float_t dz = (1.e-3) * Lsf;
	float_t dz = (1.e-3) * Lsf;
	float_t hdx = 1.7; ////////////////////////////////////////////////////////////////////////////////////

	float_t exp_scalling = 1.0;
	// Dimensions
	float_t top_die_diameter = 50.0e-3 * Lsf;
	float_t orifice = 14.e-3 * Lsf;
	float_t die_length = 25.0e-3 * Lsf;
	float_t container_thickness = 25.e-3 * Lsf;
	float_t container_length = 110.e-3 * Lsf * exp_scalling; ////////////////////////////////////////////////////////////////////
	float_t billet_length = 70.e-3 * Lsf * exp_scalling;
	float_t billet_diameter = top_die_diameter;
	float_t free_length = container_length - billet_length - die_length;
	float_t disc_inner_dim = orifice;
	float_t sub_die = 25.0e-3 * Lsf;
	top_surface = container_thickness + billet_length - dz /* + (dz / 2.) */;

	// Assymbly dimensions
	float_t lx = billet_diameter + 2.0 * container_thickness;
	float_t ly = billet_diameter + 2.0 * container_thickness;
	float_t lz = container_thickness + billet_length + die_length;
	vec3_t bbmin_tool(-lx / 2., -ly / 2., 0.0);
	vec3_t bbmax_tool(lx / 2., ly / 2., lz);
	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	float_t lz_tool = bbmax_tool.z - bbmin_tool.z;

	// Velocities
	float_t vx = 0.0 * Vsf * Lsf;
	float_t v_die = ((-1.6e-3) / 60.) * Vsf * Lsf;
	glm::vec3 w(0.0, 0.0, 90. * 0.104719755 * Vsf); //////////////////////////////////////////////////////////////////////

	global_die_velocity = v_die;
	global_wz = w.z;

	int nz = (lz / dz) + 1;

	// Phys constants
	float_t friction_coeff = 0.35; // fric ruttiumann

	std::vector<Point> points;
	float_t zz = 0;
	while (zz < lz)
	{

		if (zz < container_length)
		{

			float_t r = 0.;
			while (r < lx / 2.)
			{
				if ((zz > container_thickness + billet_length) && r < orifice / 2.)
				{
					r += dz;
					continue;
				}
				int n = ((2 * M_PI * r) / dz);
				generate_circular_arrangement(n, r, zz, points);
				r += dz;
			}
		}
		else
		{

			float_t r = 0.;

			while (r < (top_die_diameter / 2.) + dz)
			{

				if ((zz > container_thickness + billet_length + die_length / 2.) && r < disc_inner_dim / 2.)
				{
					r += dz;
					continue;
				}
				if (r < orifice / 2.)
				{
					r += dz;
					continue;
				}

				int n = ((2 * M_PI * r) / dz);
				generate_circular_arrangement(n, r, zz, points);
				r += dz;
			}
		}

		zz += dz;
	}

	// vtk_simple_write(points);

	//********************************************************* Create a material ************************************************************/
	rod.Vsf = Vsf;				   /////////////////////////////////////////////////////////////////////////////////////////////////
	AA7075 al_AA(rod.Vsf, dz, ms); /////////////////////////////////////////////////////////////////////////////////////////////////

	phys = al_AA.phys;
	float_t c0 = sqrt(phys.K / phys.rho0);

	trml_sub = al_AA.trml;
	joco_sub = al_AA.joco;

	trml_rod = al_AA.trml;
	joco_rod = al_AA.joco;
	p_mass = phys.mass;

	//********************************************************* SPH correction factors ************************************************************/
	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;

	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1. - 0.5 * q));
		printf("Cubic spline Kernel \n");
	}

	/* {
		float_t hi = (hdx * dz);
		float_t q = dz / hi;
		float_t fac = 15 / (62 * M_PI * hi * hi * hi); // 3D*h1*h1;
		corr.wdeltap = fac * (q * q * q - 6 * q + 6);
		printf("hyperbolic_spline Kernel \n");
	} */

	int n = points.size();

	float4_t *pos = new float4_t[n];
	for (int i = 0; i < n; i++)
	{
		pos[i].x = points[i].x;
		pos[i].y = points[i].y;
		pos[i].z = points[i].z;
		pos[i].w = 0;
	}

	// initiating global rod variable
	rod.vz = v_die;
	rod.vx = vx;
	rod.wz = w.z;
	rod.lz = container_thickness + billet_length;
	rod.radius = (top_die_diameter / 2.);
	rod.ms = ms;
	rod.dz = dz;
	rod.lz_tool = container_thickness;
	rod.orifice_radius = (orifice / 2.);

	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *fixed = new float_t[n];
	float_t *T = new float_t[n];
	float_t *tool_p = new float_t[n];
	float_t *cp = new float_t[n];
	float_t *H = new float_t[n];
	float_t *k = new float_t[n];
	float_t *E = new float_t[n];

	int b, c, d;
	b = 0;
	c = 0;
	d = 0;

	for (int i = 0; i < n; i++)
	{ // initialize particles

		rho[i] = phys.rho0;
		h[i] = hdx * dz;
		vel[i].w = 0.;
		E[i] = 0.;

		float_t px = pos[i].x;
		float_t py = pos[i].y;
		float_t pz = pos[i].z;
		float sq = px * px + py * py;

		// draw the container only
		float_t inner_dimeter = lx - 2. * container_thickness;
		float_t outer_dimeter = lx;

		if (sq > (inner_dimeter * inner_dimeter) / 4.0 || abs(pz) < container_thickness)
		{
			// container
			c++;
			rho[i] = 7850. * rod.ms;
			tool_p[i] = 2.;
			fixed[i] = 4; // for deformable part

			cp[i] = 561. * (1. / rod.ms);
			k[i] = 33. * rod.Vsf;

			H[i] = cp[i] * (joco_sub.Tref - 273);
			T[i] = joco_sub.Tref;

			float_t x = pos[i].x;
			float_t y = pos[i].y;
			float_t z = pos[i].z;

			vel[i].x = 0.;
			vel[i].y = 0.;
			vel[i].z = 0.;
			float_t BCD = outer_dimeter - 3 * dz;
			if (sq > (BCD * BCD) / 4.0 || abs(pz) < dz)
				fixed[i] = 7;

			if (abs(pz) > container_length - dz)
				fixed[i] = 7;
		}
		else
		{
			if (abs(pz) > container_thickness + billet_length)
			{
				// Draw the die
				d++;
				rho[i] = 7850. * rod.ms;
				tool_p[i] = 0.; // for the top die
				fixed[i] = 5;	// 4 or bigger is used for rigid body
				// if (abs(pz) > container_thickness + billet_length + die_length - dz)
				//	fixed[i] = 7; // 5 or bigger is used for rigid body

				cp[i] = 561. * (1. / rod.ms);
				k[i] = 33. * rod.Vsf;

				H[i] = cp[i] * (joco_rod.Tref - 273);
				T[i] = joco_rod.Tref;

				glm::vec3 r(pos[i].x, pos[i].y, 0.0);
				glm::vec3 v = glm::cross(w, r);

				vel[i].x = v.x;
				vel[i].y = v.y;
				vel[i].z = v_die;
			}
			else
			{
				// billet
				b++;
				tool_p[i] = 1.; // for the deformable billet
				fixed[i] = 0;	// for deformable part
				if (abs(pz) < container_thickness + 2. * dz /* billet_length/2. */)
					fixed[i] = 8; // for deformable*dz part

				H[i] = cp[i] * (joco_sub.Tref - 273);
				T[i] = joco_sub.Tref;

				cp[i] = al_AA.trml.cp;
				k[i] = al_AA.trml.k;

				float_t x = pos[i].x;
				float_t y = pos[i].y;
				float_t z = pos[i].z;

				vel[i].x = 0.;
				vel[i].y = 0.;
				vel[i].z = 0.;

				E[i] = phys.E;
			}
		}
	}

	printf("------------Particles #------------\n");
	printf("Billet %d\n", b);
	printf("Container %d \n", c);
	printf("Die %d \n", d);
	printf("Total %d \n", b + c + d);
	printf("------------------------------------\n");

	// calculate total mass of the bullet

	/* int billet_count = 0;
	for(int i =0; i<n; i++){
		if(tool_p[i] == 1) billet_count++;
	} */
	// total_mass = phys.mass  * billet_count;

	/* printf("billet_count = %d\n", billet_count);
	printf("phys.mass = %.20lf\n", phys.mass);
	printf("total_mass = %lf\n", total_mass);
	float_t masss = 0.25 * M_PI * billet_diameter * billet_diameter * billet_length * phys.rho0;
	printf("mmmmmmmmmm = %lf\n", masss);
 */

	// Adding temperature gradient  for the billet
	// 600K, 560K, 520K, 480K and 440K, respectively

	/* 	 	for (int i = 0; i < n; i++)
			{
				if (tool_p[i] == 1)
				{

					// T[i] = 500;
					float_t pz = points[i].z;

					if (pz < 1 * (billet_length / 5) + container_thickness)
						T[i] = 230;
					else if (pz < 2 * (billet_length / 5) + container_thickness)
						T[i] = 270;
					else if (pz < 3 * (billet_length / 5) + container_thickness)
						T[i] = 310;
					else if (pz < 4 * (billet_length / 5) + container_thickness)
						T[i] = 350;
					else
						T[i] = 390;
				}
			}  */

	//---------------------------------------------------------------------

	vec3_t cc(vx, 0., 0.);
	global_tool.push_back(new tool_3d_gpu(cc, phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);
	global_tool[0]->set_rod(rod);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n, cp, E, k);
	assert(check_cuda_error());
	//---------------------------------------------------------------------

	float_t max_vel = max(v_die, w.z * (top_die_diameter / 2.));
	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(max_vel) + c0);
	// float_t delta_t_max = 1.957848e-07;
	global_time_dt = 0.5 * delta_t_max;

	global_time_final = ((100. / Vsf) / VsfM); ///////////////////////////////////////////////////////////////////////////////////////
	// global_time_final = ((5 / Vsf) / VsfM); ///////////////////////////////////////////////////////////////////////////////////////

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);
	printf("global_time_final %e \n", global_time_final);
	printf("C0 = %lf \n", c0);

	//------------------------------------------------------------------------

	*grid = new grid_gpu_green(n, make_float3_t(bbmin_tool.x - 2.0 * dz, bbmin_tool.y - 2.0 * dz, -2. * dz),
							   make_float3_t(bbmax_tool.x + 2.0 * dz, bbmax_tool.y + 2.0 * dz, 2.5 * bbmax_tool.z), hdx * dz);
	(*grid)->set_bbox_vel(make_float3_t(0, 0., 0));

	global_blanking = new blanking(vec3_t(bbmin_tool.x - 2 * dz, bbmin_tool.y - 2.0 * dz, -2. * dz),
								   vec3_t(bbmax_tool.x + 2 * dz, bbmax_tool.y + 2.0 * dz, 2.5 * bbmax_tool.z),
								   vec3_t(0, 0., 0), 2000 * 2000);

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco_sub, joco_rod);
	actions_setup_thermal_constants_substrate(trml_sub);
	actions_setup_thermal_constants_rod(trml_rod);
	actions_setup_rod_constants(rod);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml_sub);
	interactions_setup_thermal_constants_rod(trml_rod, global_tool[0]);
	interactions_setup_geometry_constants(*grid);
	interactions_setup_rod_constants(rod);

	interactions_setup_johnson_cook_constants(joco_sub, joco_rod);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}

particle_gpu *setup_FEx_g_mm_sec(grid_base **grid, float_t &p_mass)
{

	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_sub = make_trml_constants();
	trml_constants trml_rod = make_trml_constants();
	joco_constants joco_sub = make_joco_constants();
	joco_constants joco_rod = make_joco_constants();
	rod_constants rod = make_rod_constants();

	// scaling factors
	float_t Lsf = 1000.; // lenth scale factor
	float_t ms = 1.0e3;
	float_t VsfM = 1.;		 // 1000 failure occured at ~15%
	float_t Vsf = 10 * VsfM; // velocity scale factor /////////////////////////////////////////////////
	global_Vsf = Vsf;

	// discretization
	float_t dz = (1.0e-3) * Lsf;
	float_t hdx = 1.3; ////////////////////////////////////////////////////////////////////////////////////

	float_t exp_scalling = 1.0;
	// Dimensions
	float_t top_die_diameter = 49.0e-3 * Lsf;
	float_t orifice = 10.e-3 * Lsf + dz;
	float_t after_orifice = 16.e-3 * Lsf + dz;
	float_t die_length = 25.0e-3 * Lsf;
	float_t bearing_length = 9.e-3 * Lsf;
	float_t container_thickness = 25.e-3 * Lsf;
	float_t container_length = 110.e-3 * Lsf * exp_scalling; ////////////////////////////////////////////////////////////////////
	float_t billet_length = 70.e-3 * Lsf * exp_scalling;
	float_t billet_diameter = top_die_diameter + 1;
	float_t free_length = container_length - billet_length - die_length;
	float_t disc_inner_dim = orifice;
	float_t sub_die = 25.0e-3 * Lsf;
	top_surface = container_thickness + billet_length /* + (dz / 2.) */;
	printf("top_surface at:%lf\n", top_surface);

	// Assymbly dimensions
	float_t lx = billet_diameter + 2.0 * container_thickness;
	float_t ly = billet_diameter + 2.0 * container_thickness;
	float_t lz = container_thickness + billet_length + die_length;
	vec3_t bbmin_tool(-lx / 2., -ly / 2., 0.0);
	vec3_t bbmax_tool(lx / 2., ly / 2., lz);
	float_t lx_tool = bbmax_tool.x - bbmin_tool.x;
	float_t ly_tool = bbmax_tool.y - bbmin_tool.y;
	float_t lz_tool = bbmax_tool.z - bbmin_tool.z;

	// Velocities
	float_t vx = 0.0 * Vsf * Lsf;
	float_t v_die = ((-1.6e-3) / 60.) * Vsf * Lsf;
	glm::vec3 w(0.0, 0.0, 90. * 0.104719755 * Vsf); //////////////////////////////////////////////////////////////////////

	global_die_velocity = v_die;
	global_wz = w.z;

	int nz = (lz / dz) + 1;
	int nx = (lx / dz) + 1;
	int ny = (ly / dz) + 1;

	// Phys constants
	float_t friction_coeff = 0.35; // fric ruttiumann

	std::vector<Point> points;

	for(int i = 0; i < nx; i++)
	{
		for(int j = 0; j < ny; j++)
		{
			for(int k = 0; k < nz; k++)
			{
				Point p;
				p.x = i * dz - lx / 2.;
				p.y = j * dz - ly / 2.;
				p.z = k * dz;
				float_t r = sqrt(p.x * p.x + p.y * p.y);


				if (r >lx/2. || (((p.x > -orifice/2. && p.x < orifice/2.) && (p.y > -orifice/2. && p.y < orifice/2.)) && p.z > container_thickness + billet_length) 
				|| (r < after_orifice/2. && p.z > container_thickness + billet_length + bearing_length)
				|| (r > top_die_diameter/2. && p.z > container_length)
				|| (r > top_die_diameter/2. && r <= dz+(top_die_diameter/2.)&& p.z > container_thickness+billet_length) )
				//|| p.z > 94.5 && p.z < 95.5 &&r <billet_diameter/2. && ( i%2 == 0 && j%2 == 0))
				{
					continue;
				} 
				points.push_back(p);

				
			}
		}
	}

	printf("points size: %d\n", points.size());

	//********************************************************* Create a material ************************************************************/
	rod.Vsf = Vsf;							  /////////////////////////////////////////////////////////////////////////////////////////////////
	AA6082_T6511_gmms al_AA(rod.Vsf, dz, ms); /////////////////////////////////////////////////////////////////////////////////////////////////

	phys = al_AA.phys;
	float_t c0 = sqrt(phys.K / phys.rho0);

	trml_sub = al_AA.trml;
	joco_sub = al_AA.joco;

	trml_rod = al_AA.trml;
	joco_rod = al_AA.joco;
	p_mass = phys.mass;

	//********************************************************* SPH correction factors ************************************************************/
	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.01;
	corr.stresseps = 0.3; // 0.3

	{
		float_t h1 = 1. / (hdx * dz);
		float_t q = dz * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1. - 0.5 * q));
		printf("Cubic spline Kernel \n");
	}

	int n = points.size();

	float4_t *pos = new float4_t[n];
	for (int i = 0; i < n; i++)
	{
		pos[i].x = points[i].x;
		pos[i].y = points[i].y;
		pos[i].z = points[i].z;
		pos[i].w = 0;
	}

	// initiating global rod variablegWz
	rod.vz = v_die;
	rod.vx = vx;
	rod.wz = w.z;
	rod.lz = container_length;
	rod.radius = (billet_diameter / 2.);
	rod.ms = ms;
	rod.dz = dz;
	rod.lz_tool = container_thickness;
	rod.orifice_radius = (orifice / 2.);

	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *fixed = new float_t[n];
	float_t *T = new float_t[n];
	float_t *tool_p = new float_t[n];
	float_t *cp = new float_t[n];
	float_t *H = new float_t[n];
	float_t *k = new float_t[n];
	float_t *E = new float_t[n];

	int b, c, d;
	b = 0;
	c = 0;
	d = 0;

	for (int i = 0; i < n; i++)
	{ // initialize particles

		rho[i] = phys.rho0;
		h[i] = hdx * dz;
		vel[i].w = 0.;
		E[i] = 0.;

		float_t px = pos[i].x;
		float_t py = pos[i].y;
		float_t pz = pos[i].z;
		float sq = px * px + py * py;

		// draw the container only
		float_t inner_dimeter = lx - 2. * container_thickness;
		float_t outer_dimeter = lx;

		if (sq > (billet_diameter * billet_diameter) / 4.0 || abs(pz) < container_thickness)
		{
			// container
			c++;
			rho[i] = 7850. * 1.0e-6 * rod.ms;
			tool_p[i] = 2.;
			fixed[i] = 4; // for deformable part

			cp[i] = 473. * 1.0e6 * (1. / rod.ms);
			k[i] = 42.6 * 1.0e6 * rod.Vsf;

			H[i] = cp[i] * (joco_sub.Tref - 273);
			T[i] = joco_sub.Tref;

			float_t x = pos[i].x;
			float_t y = pos[i].y;
			float_t z = pos[i].z;

			vel[i].x = 0.;
			vel[i].y = 0.;
			vel[i].z = 0.;
			float_t BCD = outer_dimeter - 3 * dz;
			if (sq > (BCD * BCD) / 4.0 || abs(pz) < dz)
				fixed[i] = 7;

			if (abs(pz) > container_length - 1. * dz)
				fixed[i] = 7;
		}
		else
		{
			if (abs(pz) > container_thickness + billet_length)
			{
				// Draw the die
				d++;
				rho[i] = 7850. * 1.0e-6 * rod.ms;
				tool_p[i] = 0.; // for the top die
				fixed[i] = 5;	// 4 or bigger is used for rigid body
				if (abs(pz) > container_thickness + billet_length + die_length - dz)
					fixed[i] = 7; // 5 or bigger is used for rigid body

				cp[i] = 561. * 1.0e6 * (1. / rod.ms);
				k[i] = 33. * 1.0e6 * rod.Vsf;

				H[i] = cp[i] * (joco_rod.Tref - 273);
				T[i] = joco_rod.Tref;

				glm::vec3 r(pos[i].x, pos[i].y, 0.0);
				glm::vec3 v = glm::cross(w, r);

				vel[i].x = v.x;
				vel[i].y = v.y;
				vel[i].z = v_die;
			}
			else
			{
				// billet
				b++;
				tool_p[i] = 1.; // for the deformable billet
				fixed[i] = 0;	// for deformable part
				if (abs(pz) < container_thickness + 2. * dz /* billet_length/2. */)
					fixed[i] = 8; // for deformable*dz part

				H[i] = cp[i] * (joco_sub.Tref - 273);
				T[i] = joco_sub.Tref;

				cp[i] = al_AA.trml.cp;
				k[i] = al_AA.trml.k;

				float_t x = pos[i].x;
				float_t y = pos[i].y;
				float_t z = pos[i].z;

				vel[i].x = 0.;
				vel[i].y = 0.;
				vel[i].z = 0.;

				E[i] = phys.E;
			}
		}
	}

	printf("------------Particles #------------\n");
	printf("Billet %d\n", b);
	printf("Container %d \n", c);
	printf("Die %d \n", d);
	printf("Total %d \n", b + c + d);
	printf("------------------------------------\n");
	vec3_t cc(vx, 0., 0.);
	global_tool.push_back(new tool_3d_gpu(cc, phys));

	global_tool[0]->set_algorithm_type(tool_3d_gpu::contact_algorithm::exhaustive);
	global_tool[0]->set_mu(friction_coeff);
	global_tool[0]->set_rod(rod);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n, cp, E, k);
	assert(check_cuda_error());
	//---------------------------------------------------------------------

	float_t max_vel = max(v_die, w.z * (top_die_diameter / 2.));
	const float_t CFL = 0.3;
	float_t delta_t_max = CFL * hdx * dz / (sqrt(max_vel) + c0);
	global_time_dt = 0.5 * delta_t_max;

	global_time_final = ((100. / Vsf) / VsfM); ///////////////////////////////////////////////////////////////////////////////////////

	printf("max recommended dt %e, dt used %e\n", delta_t_max, global_time_dt);
	printf("global_time_final %e \n", global_time_final);
	printf("C0 = %lf \n", c0);

	//------------------------------------------------------------------------

	*grid = new grid_gpu_green(n, make_float3_t(bbmin_tool.x - 2.0 * dz, bbmin_tool.y - 2.0 * dz, -2. * dz),
							   make_float3_t(bbmax_tool.x + 2.0 * dz, bbmax_tool.y + 2.0 * dz, 2.5 * bbmax_tool.z), hdx * dz);
	(*grid)->set_bbox_vel(make_float3_t(0, 0., 0));

	global_blanking = new blanking(vec3_t(bbmin_tool.x - 2 * dz, bbmin_tool.y - 2.0 * dz, -2. * dz),
								   vec3_t(bbmax_tool.x + 2 * dz, bbmax_tool.y + 2.0 * dz, 2.5 * bbmax_tool.z),
								   vec3_t(0, 0., 0), 2000 * 2000 * Lsf * Lsf);

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_johnson_cook_constants(joco_sub, joco_rod);
	actions_setup_thermal_constants_substrate(trml_sub);
	actions_setup_thermal_constants_rod(trml_rod);
	actions_setup_rod_constants(rod);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_thermal_constants_substrate(trml_sub);
	interactions_setup_thermal_constants_rod(trml_rod, global_tool[0]);
	interactions_setup_geometry_constants(*grid);
	interactions_setup_rod_constants(rod);

	interactions_setup_johnson_cook_constants(joco_sub, joco_rod);

	global_tool_forces = new tool_forces(global_tool.size());

	return particles;
}
