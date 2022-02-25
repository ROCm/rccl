#include <gtest/gtest.h>
#include "EnvVars.hpp"
#include "TestBed.hpp"
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  RcclUnitTesting::EnvVars::ShowConfig();
  int retCode = RUN_ALL_TESTS();
  printf("[ INFO     ] Total executed cases: %d\n", RcclUnitTesting::TestBed::NumTestsRun());
  return retCode;
}
