/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include "EnvVars.hpp"
#include "TestBed.hpp"
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  RcclUnitTesting::EnvVars::ShowConfig();
  int retCode = RUN_ALL_TESTS();
  printf("[ INFO     ] Total executed cases: %d\n", RcclUnitTesting::TestBed::NumTestsRun());

  // Show timing information
  RcclUnitTesting::EnvVars ev;
  if (ev.showTiming)
  {
    printf("[ TIMING   ] %-20s: %-20s: %10s ms (%s)\n", "TEST SUITE", "TEST NAME", "TIME", "STATUS");
    auto unitTest = ::testing::UnitTest::GetInstance();
    for (int i = 0; i < unitTest->total_test_suite_count(); i++)
    {
      auto suiteInfo = unitTest->GetTestSuite(i);
      if (!suiteInfo->should_run()) continue;

      for (int j = 0; j < suiteInfo->total_test_count(); j++)
      {
        auto testInfo = suiteInfo->GetTestInfo(j);
        if (!testInfo->should_run()) continue;
        auto testResult = testInfo->result();
        if (testResult->Skipped()) continue;

        printf("[ TIMING   ] %-20s: %-20s: %10ld ms (%4s)\n", testInfo->test_suite_name(), testInfo->name(), testResult->elapsed_time(), testResult->Passed() ? "PASS" : "FAIL");
      }
      printf("[ TIMING   ] %-20s: %-20s: %10ld ms (%4s)\n", suiteInfo->name(), "TOTAL", suiteInfo->elapsed_time(), suiteInfo->Passed() ? "PASS" : "FAIL");
    }
  }

  return retCode;
}
