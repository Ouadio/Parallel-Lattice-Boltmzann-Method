#ifndef LBM_SOLVER_H
#define LBM_SOLVER_H

#include "real_type.h"
#include "LBMParams.h"
#include "lbmFlowUtils.h"

/**
 * class LBMSolver for D2Q9
 *
 * Adapted and translated to C++ from original python version
 * found here :
 * https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes
 *
 * LBM lattice : D2Q9
 *
 * 6   3   0
 *  \  |  /
 *   \ | /
 * 7---4---1
 *   / | \
 *  /  |  \
 * 8   5   2
 *
 */
class LBMSolver
{

public:
  // distribution functions
  real_t *fin{nullptr};
  real_t *fout{nullptr};
  real_t *feq{nullptr};

  // macroscopic variables
  real_t *rho{nullptr};
  real_t *ux{nullptr};
  real_t *uy{nullptr};

  real_t *u2{nullptr};

  unsigned char *img{nullptr};

  // obstacle
  uint8_t *obstacle{nullptr};

  LBMSolver(const LBMParams &params);
  ~LBMSolver();

  //! LBM weight for D2Q9
  const real_t t[9] = {1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 4.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36};

  // LBM lattive velocity (X and Y components) for D2Q9
  const real_t v[9 * 2]{
      1, 1,
      1, 0,
      1, -1,
      0, 1,
      0, 0,
      0, -1,
      -1, 1,
      -1, 0,
      -1, -1};

  const LBMParams &params;

  void initialize();
  void run();
  void output_png(int iTime);
  void output_vtk(int iTime);

}; // class LBMSolver

#endif // LBM_SOLVER_H
