#ifndef TESTCHECKS_HPP
#define TESTCHECKS_HPP

#define HIP_CALL(x) ASSERT_EQ(x, hipSuccess)
#define NCCL_CALL(x) ASSERT_EQ(x, ncclSuccess)

#define SYSCHECK_TEST(call, name) do { \
  int retval; \
  SYSCHECKVAL_TEST(call, name, retval); \
} while (false)

#define SYSCHECKVAL_TEST(call, name, retval) do { \
  SYSCHECKSYNC_TEST(call, name, retval); \
  if (retval == -1) { \
    printf("Call to %s failed : %s\n", name, strerror(errno)); \
    fflush(stdout); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECK_GOTO_TEST(call, name, label) do { \
  int retval; \
  SYSCHECKVAL_GOTO_TEST(call, name, retval, label); \
} while (false)

#define SYSCHECKVAL_GOTO_TEST(call, name, retval, label) do { \
  SYSCHECKSYNC_TEST(call, name, retval); \
  if (retval == -1) { \
    printf("Call to %s failed : %s\n", name, strerror(errno)); \
    fflush(stdout); \
    goto label; \
  } \
} while (false)

#define SYSCHECKSYNC_TEST(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
  } else { \
    break; \
  } \
} while(true)

#define NCCLCHECK_BARRIER_TEST(call, name, rank) do { \
  ncclResult_t retval; \
  retval = call; \
  if (retval != ncclSuccess) { \
        printf("Rank %d call to %s failed : %s\n", rank, name, strerror(errno)); \
        fflush(stdout); \
        return; \
  } \
} while (false)

#define NCCLCHECK_TEST(call, name) do { \
  ncclResult_t retval; \
  retval = call; \
  if (retval != ncclSuccess) { \
        printf("Call to %s failed : %s\n", name, strerror(errno)); \
        fflush(stdout); \
        return retval; \
  } \
} while (false)

#endif
