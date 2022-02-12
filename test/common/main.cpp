#include <gtest/gtest.h>
#include "EnvVars.hpp"
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  RcclUnitTesting::EnvVars::ShowConfig();
  return RUN_ALL_TESTS();
}
