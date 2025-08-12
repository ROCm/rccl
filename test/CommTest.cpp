/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "comm.h"
#include "TestBed.hpp"



namespace RcclUnitTesting
{
  TEST(CommTest, Sorter)
  {
	TestBed testBed;
	INFO("[CommTest] starting the test with %d GPUs\n", testBed.numDevicesAvailable);
	// Configuration
	ncclTaskCollSorter* me_ptr = new ncclTaskCollSorter;
	me_ptr->head = nullptr;

	ASSERT_EQ(ncclTaskCollSorterEmpty(me_ptr), true);
	delete me_ptr;

	INFO("[CommTest] Completed the test\n");
	testBed.Finalize();

  }
}




