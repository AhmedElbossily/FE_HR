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

#include "vtk_writer.h"

bool m_individual_tool_files = false;

void report_internal_kinetic_energy(float4_t *h_vel, float_t *h_p, float_t *h_rho, float_t *h_blanked, float_t *h_part, int _num_part, float_t _p_mass)
{
	FILE *m_energy = fopen("Results_sammary/energy.csv", "a+");
	float_t sum_kinetic = 0.;
	float_t sum_internal = 0.;
	for (int i = 0; i < _num_part; i++)
	{
		if (h_blanked[i] == 1. || h_part[i] != 1.)
			continue;

		float_t vel_mag_2 = h_vel[i].x * h_vel[i].x + h_vel[i].y * h_vel[i].y + h_vel[i].z * h_vel[i].z;
		sum_kinetic += 0.5 * _p_mass * vel_mag_2;
		sum_internal += h_vel[i].w * (_p_mass / h_rho[i]);
	}

	float_t ct = global_time_current * global_Vsf;

	fprintf(m_energy, "%lf, %.9g, %.9g, %.9g\n", ct, sum_kinetic, sum_internal, sum_kinetic / sum_internal);
	fflush(m_energy);
	fclose(m_energy);
}

void vtk_writer_write_blanking()
{
	char buf[256];
	sprintf(buf, "results/vtk_bbox_%012d.vtk", global_time_step);
	FILE *fp = fopen(buf, "w+");

	vec3_t bbmin, bbmax;
	global_blanking->get_bb(bbmin, bbmax);

	const int num_face = 6;
	const int num_corner = 8;

	// generate 8 corners of tool bbox
	vec3_t c000(bbmin.x, bbmin.y, bbmin.z);
	vec3_t c100(bbmax.x, bbmin.y, bbmin.z);
	vec3_t c110(bbmax.x, bbmax.y, bbmin.z);
	vec3_t c010(bbmin.x, bbmax.y, bbmin.z);

	vec3_t c001(bbmin.x, bbmin.y, bbmax.z);
	vec3_t c101(bbmax.x, bbmin.y, bbmax.z);
	vec3_t c111(bbmax.x, bbmax.y, bbmax.z);
	vec3_t c011(bbmin.x, bbmax.y, bbmax.z);

	std::vector<vec3_t> corners({c000, c100, c110, c010, c001, c101, c111, c011});

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n"); // Particle positions
	fprintf(fp, "POINTS %d float\n", num_corner);

	for (auto &it : corners)
	{
		fprintf(fp, "%f %f %f\n", it.x, it.y, it.z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_face);
	for (int i = 0; i < num_face; i++)
	{
		fprintf(fp, "%d\n", 9);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_face, 5 * num_face);
	fprintf(fp, "4 %d %d %d %d\n", 0, 1, 2, 3);
	fprintf(fp, "4 %d %d %d %d\n", 0, 1, 5, 4);
	fprintf(fp, "4 %d %d %d %d\n", 1, 2, 6, 5);
	fprintf(fp, "4 %d %d %d %d\n", 0, 3, 7, 4);
	fprintf(fp, "4 %d %d %d %d\n", 3, 2, 6, 7);
	fprintf(fp, "4 %d %d %d %d\n", 4, 5, 6, 7);

	fprintf(fp, "\n");

	fclose(fp);
}

void vtk_writer_write(particle_gpu *particles, float_t _p_mass)
{
	static int *h_idx = 0;
	static float4_t *h_pos = 0;
	static float4_t *h_vel = 0;
	static float4_t *h_vel_bc = 0;
	static float3_t *h_vel_t = 0;
	static float_t *h_rho = 0;
	static float_t *h_h = 0;
	static float_t *h_p = 0;
	static float_t *h_T = 0;
	static float_t *h_eps = 0;
	static float_t *h_eps_dot = 0;
	static float_t *h_cp = 0;
	static float_t *h_k = 0;

	static mat3x3_t *h_S = 0;
	static mat3x3_t *h_S_t = 0;
	static mat3x3_t *h_S_der = 0;
	static mat3x3_t *h_v_der = 0;

	static float_t *h_fixed = 0;
	static float_t *h_blanked = 0;
	static float_t *h_tool_p = 0;
	static float3_t *h_contact = 0;
	static float3_t *h_cfm = 0;
	static float3_t *h_n = 0;
	static int *h_surface = 0;
	static float_t *h_E = 0;

	if (h_idx == 0)
	{
		int n_init = particles->N_init;

		// Memory allocation only upon first call;
		h_idx = new int[n_init];
		h_pos = new float4_t[n_init];
		h_vel = new float4_t[n_init];
		h_vel_bc = new float4_t[n_init];
		h_vel_t = new float3_t[n_init];
		h_rho = new float_t[n_init];
		h_h = new float_t[n_init];
		h_p = new float_t[n_init];
		h_T = new float_t[n_init];
		h_eps = new float_t[n_init];
		h_cp = new float_t[n_init];
		h_k = new float_t[n_init];

		h_S = new mat3x3_t[n_init];
		h_S_t = new mat3x3_t[n_init];
		h_S_der = new mat3x3_t[n_init];
		h_v_der = new mat3x3_t[n_init];

		h_fixed = new float_t[n_init];
		h_blanked = new float_t[n_init];
		h_tool_p = new float_t[n_init];
		h_contact = new float3_t[n_init];
		h_cfm = new float3_t[n_init];
		h_surface = new int[n_init];
		h_eps_dot = new float_t[n_init];
		h_n = new float3_t[n_init];
		h_E = new float_t[n_init];
	}

	int n = particles->N;

	cudaMemcpy(h_idx, particles->unique_idx, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pos, particles->pos, sizeof(float4_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel, particles->vel, sizeof(float4_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel_bc, particles->vel_bc, sizeof(float4_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel_t, particles->vel_t, sizeof(float3_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rho, particles->rho, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_h, particles->h, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_p, particles->p, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_T, particles->T, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_eps, particles->eps_pl, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_eps_dot, particles->eps_pl_dot, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cp, particles->cp, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_k, particles->k, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_E, particles->E, sizeof(float_t) * n, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_S_der, particles->S_der, sizeof(mat3x3_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S_t, particles->S_t, sizeof(mat3x3_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_S, particles->S, sizeof(mat3x3_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v_der, particles->v_der, sizeof(mat3x3_t) * n, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_fixed, particles->fixed, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_blanked, particles->blanked, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tool_p, particles->tool_particle, sizeof(float_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_contact, particles->fc, sizeof(float3_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_n, particles->n, sizeof(float3_t) * n, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_cfm, particles->cfm, sizeof(float3_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_surface, particles->surface, sizeof(int) * n, cudaMemcpyDeviceToHost);

	int num_unblanked_part = 0;
	for (int i = 0; i < n; i++)
	{
		if (h_blanked[i] != 1.)
		{
			num_unblanked_part++;
		}
	}

	char buf[256];
	sprintf(buf, "results/vtk_out_%012d.vtk", global_time_step);
	FILE *fp = fopen(buf, "w+");

	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "mfree iwf\n");
	fprintf(fp, "ASCII\n");
	fprintf(fp, "\n");

	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n"); // Particle positions
	fprintf(fp, "POINTS %d float\n", num_unblanked_part);
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f %f %f\n", h_pos[i].x, h_pos[i].y, h_pos[i].z);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", num_unblanked_part, 2 * num_unblanked_part);
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%d %d\n", 1, i);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", num_unblanked_part);
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%d\n", 1);
	}
	fprintf(fp, "\n");

	fprintf(fp, "POINT_DATA %d\n", num_unblanked_part);

	FILE *m_tem_744851 = fopen("Results_sammary/tem_744851.csv", "a+");
	FILE *m_tem_491082 = fopen("Results_sammary/tem_491082.csv", "a+");
	float_t ct = global_time_current * global_Vsf;

	fprintf(fp, "SCALARS unique_idx int 1\n"); // Current particle density
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%d\n", h_idx[i]);

		if (h_idx[i] == 744851)
			fprintf(m_tem_744851, "%lf, %lf\n", ct, h_T[i]);

		if (h_idx[i] == 491082)
			fprintf(m_tem_491082, "%lf, %lf\n", ct, h_T[i]);
	}
	fprintf(fp, "\n");
	fflush(m_tem_744851);
	fflush(m_tem_491082);
	fclose(m_tem_744851);
	fclose(m_tem_491082);

	fprintf(fp, "SCALARS Parts float 1\n"); // Current particle density
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f\n", h_tool_p[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Fixed float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f\n", h_fixed[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS strain_rate float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f\n", h_vel_bc[i].w);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS pl_strain float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f\n", h_eps[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS pl_strain_rate float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f\n", h_eps_dot[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Temperature float 1\n"); // Current particle temperature
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f\n", h_T[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS extroded float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f\n", h_pos[i].w);
	}
	fprintf(fp, "\n");

	fprintf(fp, "VECTORS velocity float \n");
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		fprintf(fp, "%f %f %f\n", h_vel[i].x / global_Vsf, h_vel[i].y / global_Vsf, h_vel[i].z / global_Vsf);
	}
	fprintf(fp, "\n");

	fprintf(fp, "SCALARS Svm float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	float_t sum_svm = 0;
	int count_svm = 0;
	for (unsigned int i = 0; i < n; i++)
	{
		if (h_blanked[i] == 1.)
			continue;
		double sxx = h_S[i][0][0] - h_p[i];
		double sxy = h_S[i][0][1];
		double sxz = h_S[i][0][2];
		double syy = h_S[i][1][1] - h_p[i];
		double syz = h_S[i][1][2];
		double szz = h_S[i][2][2] - h_p[i];

		double svm2 = (sxx * sxx + syy * syy + szz * szz) - sxx * syy - sxx * szz - syy * szz + 3.0 * (sxy * sxy + syz * syz + sxz * sxz);
		double svm = (svm2 > 0) ? sqrt(svm2) : 0.;
		fprintf(fp, "%f\n", svm);

		/* double sxx = h_S[i][0][0];
		double sxy = h_S[i][0][1];
		double sxz = h_S[i][0][2];
		double syy = h_S[i][1][1];
		double syz = h_S[i][1][2];
		double szz = h_S[i][2][2];
		float_t svm = sqrt(3. / 2.) * 2. * sqrt_t(sxx * sxx + syy * syy + szz * szz + 2 * (sxy * sxy + syz * syz + sxz * sxz));
		fprintf(fp, "%f\n", svm); */

		if (h_tool_p[i] == 1 && h_pos[i].w == 0)
		{
			sum_svm += svm;
			count_svm++;
		}
	}
	fprintf(fp, "\n");

	fclose(fp);

	FILE *m_svm = fopen("Results_sammary/svm.csv", "a+");
	fprintf(m_svm, "%lf, %.9g\n", ct, sum_svm / count_svm);
	fflush(m_svm);
	fclose(m_svm);

	report_internal_kinetic_energy(h_vel, h_p, h_rho, h_blanked, h_tool_p, n, _p_mass);
}
void vtk_writer(particle_gpu *particles, float_t mass)
{

	vtk_writer_write(particles, mass);

	if (global_blanking)
	{
		vtk_writer_write_blanking();
	}
}
