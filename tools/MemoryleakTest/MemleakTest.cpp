#include <rccl/rccl.h>
#include "macros.hpp"
#include <iostream>
int main(int argc, char **argv){
    if (argc == 3)
    {
        int testfuncindex = atoi(argv[1]);
        if(testfuncindex == 0){
            int nranks = atoi(argv[2]);
            ncclComm_t comm[nranks];
            printf(" num ranks = %d \n",nranks);
            NCCL_CALL(ncclCommInitAll(comm, nranks, NULL));
            for(int r = 0; r < nranks; r++) {
                 NCCL_CALL(ncclCommDestroy(comm[r]));
            }
        }
    }
    return 0;
}
