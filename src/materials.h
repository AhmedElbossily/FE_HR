#include "constants_structs.h"

class Al5083H116 // DOI: 10.1177/0021998315622982
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    Al5083H116(float_t Vsf, float_t dz)
    {

        // Phys
        phys.E = 70.e9;
        phys.nu = 0.3;
        phys.rho0 = 2700.0;
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants
        joco.A = 1.67e8;
        joco.B = 5.96e8;
        joco.C = 0.001;
        joco.m = 0.859;
        joco.n = 0.551; //.36
        // joco.n = 0.045;
        joco.Tref = 293.0;
        joco.Tmelt = 893.0;
        joco.eps_dot_ref = 1;
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 910.;
        trml.cp0 = 910.;
        trml.cp1 = 0.0;
        trml.cp2 = 0.;
        trml.tq = 0.9;

        trml.k = 117. * Vsf;
        trml.k0 = 117 * Vsf;
        trml.k1 = 0. * Vsf;
        trml.k2 = 0. * Vsf;
        trml.k3 = 0. * Vsf;
        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120;
    }
};

class Al5083_Katherine // https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=3767&context=etd
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    Al5083_Katherine(float_t Vsf, float_t dz)
    {

        // Phys
        phys.E = 70.e9;
        phys.nu = 0.3;
        phys.rho0 = 2700.0 * 3.9e5; /*///////////////////////////////////////////////////////////*/
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants
        joco.A = 92.0e6;
        joco.B = 475.0e6;
        joco.C = 0.035;
        joco.m = 0.85;
        joco.n = 0.087;
        joco.Tref = 297.15;
        joco.Tmelt = 847.15;

        joco.eps_dot_ref = 1;
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 910.;
        trml.cp0 = 910.;
        trml.cp1 = 0.0;
        trml.cp2 = 0.;
        trml.tq = 0.9;

        trml.k = 117. * Vsf;
        trml.k0 = 117 * Vsf;
        trml.k1 = 0. * Vsf;
        trml.k2 = 0. * Vsf;
        trml.k3 = 0. * Vsf;
        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120;
    }
};

class Al7050 // https://doi.org/10.1115/1.4041915
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    Al7050(float_t Vsf, float_t dz)
    {

        // Phys
        phys.E = 70.3e9;
        phys.nu = 0.33;
        phys.rho0 = 2830.0;
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants substrate
        joco.A = 450.821e6; //	450.821 MPa
        joco.B = 108.537e6; //	108.537 MPA
        joco.C = 0.027;     //	0.027
        joco.m = 0.981;     //	0.981
        joco.n = 0.045;     //	0.045
        joco.Tref = 293.0;  //	323
        joco.Tmelt = 903.0; //	903 K
        joco.eps_dot_ref = 1;
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 896.;
        trml.cp0 = 896.;
        trml.cp1 = 0.;
        trml.cp2 = 0.;
        trml.tq = 0.9;

        trml.k = 167. * Vsf;
        trml.k0 = 167 * Vsf;
        trml.k1 = 0. * Vsf;
        trml.k2 = 0. * Vsf;
        trml.k3 = 0. * Vsf;

        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120;
    }
};

class Al7050_kirk
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    Al7050_kirk(float_t Vsf, float_t dz)
    {

        // Phys
        phys.E = 70.3e9;
        phys.nu = 0.33;
        phys.rho0 = 2830.0;
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants substrate
        joco.A = 450.821e6; //	450.821 MPa
        joco.B = 108.537e6; //	108.537 MPA
        joco.C = 0.027;     //	0.027
        joco.m = 0.981;     //	0.981
        joco.n = 0.045;     //	0.045
        joco.Tref = 293.0;  //	323
        joco.Tmelt = 903.0; //	903 K
        joco.eps_dot_ref = 1;
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 896.;
        trml.cp0 = 896.;
        trml.cp1 = 0.145;
        trml.cp2 = 1.18e-3;
        trml.tq = 0.9;

        trml.k = 167. * Vsf;
        trml.k0 = 167 * Vsf;
        trml.k1 = 0.084 * Vsf;
        trml.k2 = -4.14e-4 * Vsf;
        trml.k3 = 3.13e-7 * Vsf;

        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120;
    }
};

class AA1100
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    AA1100(float_t Vsf, float_t dz, float_t ms)
    {

        // Phys
        phys.E = 68.9e9;
        phys.nu = 0.33;
        phys.rho0 = 2700.0 * ms; /***********************************************************/
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants substrate
        /*         joco.A = 34.0e6;
                joco.B = 0.0;
                joco.C = 0.001;
                joco.m = 1.7;
                joco.n = 0.0;
                joco.Tref = 24.0;
                joco.Tmelt = 640.0;
                joco.eps_dot_ref = 1;
                joco.clamp_temp = 1.;  */

        joco.A = 17.e6;
        joco.B = 324.e6;
        // joco.B = 200.e6;
        joco.C = 0.2;
        joco.m = 0.7;
        joco.n = 0.25;
        // joco.n = 0.0;
        joco.Tref = 24.0;
        joco.Tmelt = 640.0;
        joco.eps_dot_ref = 1;
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 904. * (1. / ms);  // 904 623
        trml.cp0 = 904. * (1. / ms); // 904 623
        trml.cp1 = 0.0;
        trml.cp2 = 0.0;
        trml.tq = 0.0;

        trml.k = 222. * Vsf;  // 222 260
        trml.k0 = 222. * Vsf; // 222 260
        trml.k1 = 0. * Vsf;
        trml.k2 = 0. * Vsf;
        trml.k3 = 0. * Vsf;

        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120;
    }

    void print_properties()
    {

        FILE *m_fp = fopen("matrial_properties_kgCmSK.txt", "w+");

        fprintf(m_fp, "phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt\n");
        fprintf(m_fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt);
    }
};

class AA1100_gCmS
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    AA1100_gCmS(float_t Vsf, float_t dz)
    {

        float_t E_UC = 10;
        float_t rho0_UC = 0.001;
        float_t A_UC = 10;
        float_t cp_UC = 10000;
        float_t L_UC = 10000;
        float_t k_UC = 10000000;
        // Phys
        phys.E = 68.9e9 * E_UC;
        phys.nu = 0.33;
        phys.rho0 = 2700.0 * rho0_UC * 3.9e5; /* 3.9e5 **********************************************************/
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants substrate
        joco.A = 34.0e6 * A_UC;
        joco.B = 0.0 * A_UC;
        joco.C = 0.001;
        joco.m = 1.7;
        joco.n = 0.0;
        joco.Tref = 297.0;
        joco.Tmelt = 913;
        joco.eps_dot_ref = 1;
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 904. * cp_UC;
        trml.cp0 = 904. * cp_UC;
        trml.cp1 = 0.0 * cp_UC;
        trml.cp2 = 0.0 * cp_UC;
        trml.tq = 0.0 * cp_UC;

        trml.k = 222. * Vsf * k_UC;
        trml.k0 = 222 * Vsf * k_UC;
        trml.k1 = 0. * Vsf * k_UC;
        trml.k2 = 0. * Vsf * k_UC;
        trml.k3 = 0. * Vsf * k_UC;

        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000 * L_UC;
        trml.cpMax = 1120 * cp_UC;
    }

    void print_properties()
    {

        FILE *m_fp = fopen("matrial_properties_gcmsK.txt", "w+");

        fprintf(m_fp, "phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt\n");
        fprintf(m_fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt);
    }
};

class AA7075 // https://journals.sagepub.com/doi/full/10.1177/1464420715591860
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    AA7075(float_t Vsf, float_t dz, float_t ms)
    {

        // Phys
        phys.E = 71.7e9;
        // phys.E = 27.0e9;
        phys.nu = 0.33;
        phys.rho0 = 2700.0 * ms; /***********************************************************/
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants substrate
        joco.A = 317.37e6;
        // joco.B = 166.95e6;
        joco.B = 0;
        joco.C = 0.00736;
        joco.m = 1.57; ///////////////////////////////////////////////////////////////////////////////////////////////
        // joco.m = 0.5; ///////////////////////////////////////////////////////////////////////////////////////////////
        //  joco.n = 0.42;
        joco.n = 0;
        joco.Tref = 20.0;
        joco.Tmelt = 477.0;
        joco.eps_dot_ref = 1;
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 960. * (1. / ms);  // 904 623
        trml.cp0 = 960. * (1. / ms); // 904 623
        trml.cp1 = 0.0;
        trml.cp2 = 0.0;
        trml.tq = 0.0;

        trml.k = 130. * Vsf;  // 222 260
        trml.k0 = 130. * Vsf; // 222 260
        trml.k1 = 0. * Vsf;
        trml.k2 = 0. * Vsf;
        trml.k3 = 0. * Vsf;

        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120;
    }

    void print_properties()
    {

        FILE *m_fp = fopen("matrial_properties_kgCmSK.txt", "w+");

        fprintf(m_fp, "phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt\n");
        fprintf(m_fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt);
    }
};

class AA7075_gmms // https://pubs.aip.org/aip/acp/article-abstract/1195/1/945/836217/CONSTITUTIVE-MODEL-CONSTANTS-FOR-Al7075-T651-and?redirectedFrom=fulltext
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    AA7075_gmms(float_t Vsf, float_t dz, float_t ms)
    {

        // Phys
        phys.E = 71.7e9;
        phys.nu = 0.33;
        phys.rho0 = 2810.0 * 1.0e-6 * ms; /***********************************************************/
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants substrate
        // joco.A = 317.37e6;
        // joco.A = 455.e6;
        // joco.A = 480.e6;
        joco.A = 546.e6;
        joco.B = 0; // 678.e6;
        joco.C = 0.024;
        joco.m = 1.57; //////////////////////////////////////////////////////////////0.6
        joco.n = 0;    // 0.71
        joco.Tref = 22.0;
        joco.Tmelt = 477.0;
        // joco.Tmelt = 635.0;
        // joco.Tmelt = 520;
        joco.eps_dot_ref = 1;
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 960. * 1.0e6 * (1. / ms);  // 904 623
        trml.cp0 = 960. * 1.0e6 * (1. / ms); // 904 623
        trml.cp1 = 0.0;
        trml.cp2 = 0.0;
        trml.tq = 0.0;

        trml.k = 130. * 1.0e6 * Vsf;  // 222 260
        trml.k0 = 130. * 1.0e6 * Vsf; // 222 260
        trml.k1 = 0. * Vsf;
        trml.k2 = 0. * Vsf;
        trml.k3 = 0. * Vsf;

        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120 * 1.0e6;
    }

    void print_properties()
    {

        FILE *m_fp = fopen("matrial_properties_kgCmSK.txt", "w+");

        fprintf(m_fp, "phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt\n");
        fprintf(m_fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt);
    }
};

// https://www.researchgate.net/publication/275277268_3-Dimensional_Nonlinear_Finite_Element_Analysis_of_both_Thermal_and_Mechanical_Response_of_Friction_Stir_Welded_2024-T3_Aluminum_Plates
class AA2024_T3_gmms // https://www.brown.edu/Departments/Engineering/Courses/En2340/Projects/Projects_2017/Kim.pdf
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    AA2024_T3_gmms(float_t Vsf, float_t dz, float_t ms)
    {

        // Phys
        phys.E = 72.4e9;
        phys.nu = 0.33;
        phys.rho0 = 2710.0 * 1.0e-6 * ms; /***********************************************************/
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants substrate
        // joco.A = 317.37e6;
        // joco.A = 455.e6;
        // joco.A = 480.e6;
        joco.A = 265.e6;
        joco.B = 0; // 426.e6;
        joco.C = 0.018;
        joco.m = 1.;
        joco.n = 0; // 0.34
        joco.Tref = 20.0;
        joco.Tmelt = 502.0;
        // joco.Tmelt = 635.0;
        joco.eps_dot_ref = 1; //////////////////////////////////////////////////////////////////////////
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 881. * 1.0e6 * (1. / ms);  // 904 623
        trml.cp0 = 881. * 1.0e6 * (1. / ms); // 904 623
        trml.cp1 = 0.0;
        trml.cp2 = 0.0;
        trml.tq = 0.0;

        trml.k = 164. * 1.0e6 * Vsf;  // 222 260
        trml.k0 = 164. * 1.0e6 * Vsf; // 222 260
        trml.k1 = 0. * Vsf;
        trml.k2 = 0. * Vsf;
        trml.k3 = 0. * Vsf;

        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120 * 1.0e6;
    }

    void print_properties()
    {

        FILE *m_fp = fopen("matrial_properties_kgCmSK.txt", "w+");

        fprintf(m_fp, "phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt\n");
        fprintf(m_fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt);
    }
};

// https://www.researchgate.net/publication/364262406_Heat_generation_plastic_deformation_and_residual_stresses_in_friction_stir_welding_of_aluminium_alloy
class AA6082_T6511_gmms
{
public:
    phys_constants phys;
    trml_constants trml;
    joco_constants joco;

    AA6082_T6511_gmms(float_t Vsf, float_t dz, float_t ms)
    {

        // Phys
        phys.E = 70.0e9;
        phys.nu = 0.3;
        phys.rho0 = 2700.0 * 1.0e-6 * ms; /***********************************************************/
        phys.G = phys.E / (2. * (1. + phys.nu));
        phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
        phys.mass = dz * dz * dz * phys.rho0;

        // Johnson Cook Constants substrate

        joco.A = 250.e6;
        joco.B = 0; // 327.79e6;
        joco.C = 0.00747;
        joco.m = 1.31; // 1.31
        joco.n = 0;    // 1.008
        joco.Tref = 25.0;
        // joco.Tmelt = 582.0;
        joco.Tmelt = 540;
        joco.eps_dot_ref = 1.; //////////////////////////////////////////////////////////////////////////
        joco.clamp_temp = 1.;

        // Thetmal
        trml.cp = 881. * 1.0e6 * (1. / ms);  // 904 623
        trml.cp0 = 881. * 1.0e6 * (1. / ms); // 904 623
        trml.cp1 = 0.0;
        trml.cp2 = 0.0;
        trml.tq = 0.0;

        trml.k = 170. * 1.0e6 * Vsf;  // 222 260
        trml.k0 = 170. * 1.0e6 * Vsf; // 222 260
        trml.k1 = 0. * Vsf;
        trml.k2 = 0. * Vsf;
        trml.k3 = 0. * Vsf;

        trml.alpha = trml.k / (phys.rho0 * trml.cp);
        trml.T_init = joco.Tref;
        trml.eta = 0.9;
        trml.L = 380000;
        trml.cpMax = 1120 * 1.0e6;
    }

    void print_properties()
    {

        FILE *m_fp = fopen("matrial_properties_kgCmSK.txt", "w+");

        fprintf(m_fp, "phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt\n");
        fprintf(m_fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", phys.E, phys.nu, phys.rho0, trml.cp0, trml.k0, joco.A, joco.B, joco.n, joco.m, joco.C, joco.Tref, joco.Tmelt);
    }
};
