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

            /*
            **
            */

            int cuDevice;
           

            int numRanks;
            NCCL_CALL(ncclCommCount(comm[0],&numRanks));
            printf("num ranks : %d \n",numRanks);

            for(int r=0; r< nranks; r++){
                int myRank;
                int myDev;
                NCCL_CALL(ncclCommCuDevice(comm[r],&myDev));
                NCCL_CALL(ncclCommUserRank(comm[r],&myRank));
                printf("r:%d, rank:%d, device:%d \n",r,myRank, myDev);
            }

           /*
           **
           */
            for(int r = 0; r < nranks; r++) {
                 NCCL_CALL(ncclCommDestroy(comm[r]));
            }


            // for(int r= 0; r< nranks; r++){
            //     if(comm[r]){
            //         printf(" Something wrong with comm[%d] \n",r);
            //     }
            // }
        }
    }
    return 0;
}