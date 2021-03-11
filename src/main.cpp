#include <cstdlib>
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>
#include "lbm/LBMSolver.h"

int main(int argc, char *argv[])
{

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc > 1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  ConfigMap configMap(input_file);

  // test: create a LBMParams object
  auto start = std::chrono::high_resolution_clock::now();
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  LBMSolver *solver = new LBMSolver(params);

  solver->run();

  delete solver;

  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Slow calculations took "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

  int dimX, dimY;
  dimX = 1 << 10;
  dimY = 1 << 10;
  double *myArray = new double[dimY * dimX];

  start = std::chrono::high_resolution_clock::now();
#pragma acc data copy(myArray[:(dimX*dimY)])
#pragma acc parallel loop collapse(2)
  for (size_t i = 0; i != dimY; i++)
  {
    for (size_t j = 0; j != dimX; j++)
    {
      *(myArray + j + i * dimX) = (i + j) % 10;
    }
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout
      << "OpenACC calculations took "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

  delete[] myArray;
  return EXIT_SUCCESS;
}
