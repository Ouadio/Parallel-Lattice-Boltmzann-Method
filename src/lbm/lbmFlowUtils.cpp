#include <math.h> // for M_PI = 3.1415....

#include "lbmFlowUtils.h"
#include <iostream>
#include "writePNG/lodepng.h"
#include <sstream>
#include <vector>

// ======================================================
// ======================================================
#pragma acc routine seq
real_t compute_vel(int dir,
                   int i,
                   int j,
                   double uLB,
                   double ly)
{

  // flow is along X axis
  // X component is non-zero
  // Y component is always zero

  return (1 - dir) * uLB * (1 + 1e-4 * sin(j / ly * 2 * M_PI));

} // compute_vel

// ======================================================
void init_obstacle_mask(const LBMParams &params,
                        uint8_t *obstacle)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const real_t cx = params.cx;
  const real_t cy = params.cy;

  const real_t r = params.r;

#pragma acc parallel loop independent gang worker vector vector_length(32) num_workers(16) present(obstacle)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector
    for (int i = 0; i < nx; ++i)
    {

      int index = i + nx * j;

      real_t x = 1.0 * i;
      real_t y = 1.0 * j;

      obstacle[index] = (x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r ? 1 : 0;

    } // end for i
  }   // end for j
} // init_obstacle_mask

// ======================================================
void initialize_macroscopic_variables(const LBMParams &params,
                                      real_t *rho,
                                      real_t *ux,
                                      real_t *uy)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const double uLB = params.uLB;
  const double ly = params.ly;

#pragma acc parallel loop independent gang vector vector_length(128) num_workers(8) num_gangs(256) present(rho, ux, uy) async(1)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector
    for (int i = 0; i < nx; ++i)
    {
      ux[i + nx * j] = compute_vel(0, i, j, uLB, ly);
      rho[i + nx * j] = 1.0;
    } // end for i
  }   // end for j

#pragma acc parallel loop independent gang worker vector vector_length(64) num_workers(16) num_gangs(128) present(rho, ux, uy) async(1)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector
    for (int i = 0; i < nx; ++i)
    {
      uy[i + nx * j] = compute_vel(1, i, j, uLB, ly);
    } // end for i
  }   // end for j

} // initialize_macroscopic_variables

// ======================================================
void initialize_equilibrium(const LBMParams &params,
                            const real_t *v,
                            const real_t *t,
                            const real_t *rho,
                            const real_t *ux,
                            const real_t *uy,
                            real_t *fin)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

#pragma acc parallel loop independent gang worker vector num_gangs(256) vector_length(32) num_workers(16) present(ux, uy, fin, rho, t, v) async(1)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector cache(v [0:(2 * npop)], t [0:npop])
    for (int i = 0; i < nx; ++i)
    {
      double cu = 0.0;
      int index = i + nx * j;

      double uX = ux[index];
      double uY = uy[index];

      double usqr = 3.0 / 2 * (uX * uX + uY * uY);
#pragma acc loop seq
      for (int ipop = 0; ipop < npop; ++ipop)
      {
        cu = 3 * (v[ipop * 2] * uX +
                  v[ipop * 2 + 1] * uY);

        fin[index + ipop * nx * ny] = rho[index] * t[ipop] * (1 + cu + 0.5 * cu * cu - usqr);
      }

    } // end for i
  }   // end for j
} // init equilibrium

// ==========================      LOOP        ============================

// ======================================================
void border_outflow(const LBMParams &params,
                    real_t *fin)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const int nxny = nx * ny;

  const int i1 = nx - 1;
  const int i2 = nx - 2;
#pragma acc parallel loop independent present(fin) async(1)
  for (int j = 0; j < ny; ++j)
  {

    int index1 = i1 + nx * j;
    int index2 = i2 + nx * j;

    fin[index1 + 6 * nxny] = fin[index2 + 6 * nxny];
    fin[index1 + 7 * nxny] = fin[index2 + 7 * nxny];
    fin[index1 + 8 * nxny] = fin[index2 + 8 * nxny];

  } // end for j

} // border_outflow

// ======================================================
void macroscopic(const LBMParams &params,
                 const real_t *v,
                 const real_t *fin,
                 real_t *rho,
                 real_t *ux,
                 real_t *uy)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

#pragma acc parallel loop independent gang worker vector vector_length(128) num_workers(8) present(ux, uy, fin, rho, v) async(1)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector cache(v [0:(2 * npop)])
    for (int i = 1; i < nx; ++i) //Start from 1 since border inflow condition will overrite it anyways
    {
      int base_index = i + nx * j;

      double rho_tmp = 0;
      double ux_tmp = 0;
      double uy_tmp = 0;
      double tempFin = 0.0;

#pragma acc loop seq
      for (int ipop = 0; ipop < npop; ++ipop)
      {

        // int index = base_index + ipop * nx * ny;
        tempFin = fin[base_index + ipop * nx * ny];
        // Oth order moment
        rho_tmp += tempFin;

        // 1st order moment
        ux_tmp += v[ipop * 2] * tempFin;
        uy_tmp += v[ipop * 2 + 1] * tempFin;

      } // end for ipop

      rho[base_index] = rho_tmp;
      ux[base_index] = ux_tmp / rho_tmp;
      uy[base_index] = uy_tmp / rho_tmp;

    } // end for i
  }   // end for j

} // init macroscopic

// ======================================================
void border_inflow(const LBMParams &params,
                   const real_t *fin,
                   real_t *rho,
                   real_t *ux,
                   real_t *uy)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const double uLB = params.uLB;
  const double ly = params.ly;

  const int nxny = nx * ny;

  const int i = 0;
  int index = 0;
#pragma acc parallel loop independent gang worker vector num_gangs(5) vector_length(128) num_workers(4) present(ux, uy, rho, fin) async(1)
  for (int j = 0; j < ny; ++j)
  {

    index = nx * j;

    ux[index] = compute_vel(0, i, j, uLB, ly);
    uy[index] = compute_vel(1, i, j, uLB, ly);
    rho[index] = 1 / (1 - ux[index]) *
                 (fin[index + 3 * nxny] + fin[index + 4 * nxny] + fin[index + 5 * nxny] +
                  2 * (fin[index + 6 * nxny] + fin[index + 7 * nxny] + fin[index + 8 * nxny]));

  } // end for j

} // border_inflow

// ======================================================
void equilibrium(const LBMParams &params,
                 const real_t *v,
                 const real_t *t,
                 const real_t *rho,
                 const real_t *ux,
                 const real_t *uy,
                 real_t *feq)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

#pragma acc parallel loop independent gang vector vector_length(32) num_gangs(500) num_workers(16) present(ux, uy, feq, rho, t, v) async(3)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop worker vector cache(v [0:(2 * npop)], t [0:npop])
    for (int i = 0; i < nx; ++i)
    {

      int index = i + nx * j;

      double usqr = 3.0 / 2 * (ux[index] * ux[index] + uy[index] * uy[index]);
#pragma acc loop seq
      for (int ipop = 0; ipop < npop; ++ipop)
      {
        double cu = 3 * (v[ipop * 2] * ux[index] +
                         v[ipop * 2 + 1] * uy[index]);

        feq[index + ipop * nx * ny] = rho[index] * t[ipop] * (1 + cu + 0.5 * cu * cu - usqr);
      }

    } // end for i
  }   // end for j

} // equilibrium

// ======================================================
void update_fin_inflow(const LBMParams &params,
                       const real_t *feq,
                       real_t *fin)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const int nxny = nx * ny;

  //i = 0
#pragma acc parallel loop independent gang worker vector num_gangs(50) num_workers(4) vector_length(128) present(feq, fin) async(3)
  for (int j = 0; j < ny; ++j)
  {
    int index = nx * j;

    fin[index + 0 * nxny] = feq[index + 0 * nxny] + fin[index + 8 * nxny] - feq[index + 8 * nxny];
    fin[index + 1 * nxny] = feq[index + 1 * nxny] + fin[index + 7 * nxny] - feq[index + 7 * nxny];
    fin[index + 2 * nxny] = feq[index + 2 * nxny] + fin[index + 6 * nxny] - feq[index + 6 * nxny];

  } // end for j

} // update_fin_inflow

// ======================================================
void compute_collision(const LBMParams &params,
                       const real_t *fin,
                       const real_t *feq,
                       real_t *fout)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const int nxny = nx * ny;

  const int npop = LBMParams::npop;
  const double omega = params.omega;

#pragma acc parallel loop independent gang worker vector num_gangs(50) num_workers(8) vector_length(64) present(fin, feq, fout) async(3)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector
    for (int i = 0; i < nx; ++i)
    {
      int index = i + nx * j;
#pragma acc loop seq
      for (int ipop = 0; ipop < npop; ++ipop)
      {
        int index_f = index + ipop * nxny;

        fout[index_f] = fin[index_f] - omega * (fin[index_f] - feq[index_f]);
      } // end for ipop

    } // end for i
  }   // end for j

} // compute_collision

// ======================================================
void update_obstacle(const LBMParams &params,
                     const real_t *fin,
                     const uint8_t *obstacle,
                     real_t *fout)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nxny = nx * ny;
  const int npop = LBMParams::npop;
#pragma acc parallel loop independent gang worker vector num_workers(16) vector_length(64) num_gangs(80) present(fin, fout, obstacle) async(3)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector
    for (int i = 0; i < nx; ++i)
    {

      int index = i + nx * j;

      if (obstacle[index] == 1)
      {
#pragma acc loop seq
        for (int ipop = 0; ipop < npop; ++ipop)
        {

          int index_out = index + ipop * nxny;
          int index_in = index + (8 - ipop) * nxny;

          fout[index_out] = fin[index_in];

        } // end for ipop

      } // end inside obstacle

    } // end for i
  }   // end for j

} // update_obstacle

// ======================================================
void streaming(const LBMParams &params,
               const real_t *v,
               const real_t *fout,
               real_t *fin)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nxny = nx * ny;
  const int npop = LBMParams::npop;

#pragma acc parallel loop independent gang worker vector num_workers(4) vector_length(128) num_gangs(500) present(fin, fout, v) async(3)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector cache(v [0:(2 * npop)])
    for (int i = 0; i < nx; ++i)
    {
      int index = i + nx * j;
#pragma acc loop seq
      for (int ipop = 0; ipop < npop; ++ipop)
      {
        int index_in = index + ipop * nxny;
        int i_out = i - v[2 * ipop];
        if (i_out < 0)
          i_out += nx;
        if (i_out > nx - 1)
          i_out -= nx;

        int j_out = j - v[2 * ipop + 1];
        if (j_out < 0)
          j_out += ny;
        if (j_out > ny - 1)
          j_out -= ny;

        int index_out = i_out + nx * j_out + ipop * nxny;

        fin[index_in] = fout[index_out];

      } // end for ipop

    } // end for i

  } // end for j

} // streaming

//--------------------------------------
// ================================    Output PNG (GPU)    =========================

void prepare_png_gpu(const LBMParams &params,
                     const real_t *ux,
                     const real_t *uy,
                     real_t *u2,
                     unsigned char *img)
{

#pragma acc update self(ux[0:1], uy[0:1]) async(2)
  const int nx = params.nx;
  const int ny = params.ny;
#pragma acc wait(2)
  // compute velocity norm, as well as min and max values
  real_t min_value = sqrt(ux[0] * ux[0] + uy[0] * uy[0]);
  real_t max_value = min_value;
#pragma acc parallel num_gangs(100) vector_length(32) num_workers(4) present(u2, ux, uy) copy(min_value, max_value) async(2)
  {
#pragma acc loop gang worker vector collapse(2) reduction(max \
                                                   : max_value)
    for (int j = 0; j < ny; ++j)
    {
      for (int i = 0; i < nx; ++i)
      {
        int index = i + nx * j;

        real_t uX = ux[index];
        real_t uY = uy[index];

        real_t uu2 = sqrt(uX * uX + uY * uY);

        u2[index] = uu2;

        max_value = std::max(uu2, max_value);

      } // end for i

    } // end for j

#pragma acc loop gang vector collapse(2) reduction(min \
                                                   : min_value)
    for (int j = 0; j < ny; ++j)
    {
      for (int i = 0; i < nx; ++i)
      {
        int index = i + nx * j;

        min_value = std::min(u2[index], min_value);

      } // end for i

    } // end for j
  }

#pragma acc parallel loop gang worker vector num_workers(4) vector_length(128) num_gangs(100) present(u2, img) copyin(min_value, max_value) async(2)
  for (int j = 0; j < ny; ++j)
  {
#pragma acc loop vector
    for (int i = 0; i < nx; ++i)
    {
      int index = i + nx * j;

      // rescale velocity in 0-255 range
      unsigned char value = static_cast<unsigned char>((u2[index] - min_value) / (max_value - min_value) * 255);
      img[0 + 4 * i + 4 * nx * j] = value;
      img[1 + 4 * i + 4 * nx * j] = value;
      img[2 + 4 * i + 4 * nx * j] = value;
      img[3 + 4 * i + 4 * nx * j] = value;
    }
  }

} // LBMSolver::output_png

void output_png_gpu(int iTime,
                    int nx,
                    int ny,
                    const unsigned char *img)
{
  std::cout << "Output data (PNG) at time " << iTime << "\n";

  // create png image buff
  std::vector<unsigned char> image;
  image.resize(nx * ny * 4);
  image.assign(img, img + (nx * ny * 4));

  std::ostringstream iTimeNum;
  iTimeNum.width(7);
  iTimeNum.fill('0');
  iTimeNum << iTime;

  std::string filename = "vel_gpu_" + iTimeNum.str() + ".png";

  // encode the image
  unsigned error = lodepng::encode(filename, image, nx, ny);

  //if there's an error, display it
  if (error)
    std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}
