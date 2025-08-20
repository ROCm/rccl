#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "param.h"

namespace RcclUnitTesting
{
    TEST(ParamTests, initEnv_ParseValidConfFile){
        // This function call reads and opens the conf file from the path
        // which is set using env. variable NCCL_CONF_FILE
        initEnv();

        EXPECT_EQ(getenv("TEST_VAR_WITH_NO_VALUE"), nullptr);
        EXPECT_STREQ(getenv("TEST_VAR"), "12345"); 
        
        // Clean up
        unsetenv("TEST_VAR_WITH_NO_VALUE");
        unsetenv("TEST_VAR");
    }

    TEST(ParamTests, ncclLoadParam_InvalidParam){
        int64_t cache = -1;               
        const int64_t defaultVal = 12345; // Dummy input value

        // Force overflow: value exceeds int64_t max (9223372036854775807)
        setenv("TEST_INVALID_PARAM", "99999999999999999999", 1); // Dummy variable and value 
        ncclLoadParam("TEST_INVALID_PARAM", defaultVal, -1, &cache);
        unsetenv("TEST_INVALID_PARAM");

        EXPECT_EQ(cache, defaultVal);       // Cache should be set to default value
    }
}