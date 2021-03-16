#include <cstdlib>
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>
#include "lbm/LBMSolver.h"
#include "utils/openacc_utils.h"

int main(int argc, char *argv[])
{
  typedef std::chrono::high_resolution_clock Time;

  std::string input_file = argc > 1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  // print OpenACC version / info
  print_openacc_version();
  init_openacc();

  ConfigMap configMap(input_file);

  // test: create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  params.print();

  // Instanciate solver class
  LBMSolver *mySolver = new LBMSolver(params);

  auto start = Time::now();

  // Run simulation
  mySolver->run();

  auto end = Time::now();

  double diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "Elapsed time : " << diff << std::endl;

  delete mySolver;

  return EXIT_SUCCESS;
}
