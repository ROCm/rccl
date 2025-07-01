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

	assert(true == ncclTaskCollSorterEmpty(me_ptr));
	delete me_ptr;

	INFO("[CommTest] Completed the test\n");
	testBed.Finalize();

  }
}




