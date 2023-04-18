
#define __HIP_PLATFORM_SPIRV__ // It is required for CHIP-SPV to kick-in
#include "hipsolver.h"

typedef struct syclDevice_st *syclDevice_t;
typedef struct syclPlatform_st *syclPlatform_t;
typedef struct syclContext_st *syclContext_t;
typedef struct syclQueue_st *syclQueue_t;
typedef struct syclEvent_st *syclEvent_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct syclHandle* syclHandle_t;

// helper functions
hipsolverStatus_t sycl_create_handle(syclHandle_t* handle);
hipsolverStatus_t sycl_destroy_handle(syclHandle_t handle);
hipsolverStatus_t sycl_set_hipstream(syclHandle_t handle,
                                  unsigned long const* lzHandles,
                                  int                  nHandles,
                                   hipStream_t          stream,
                                   const char*          hipBlasBackendName);
hipsolverStatus_t sycl_get_hipstream(syclHandle_t handle, hipStream_t* pStream);

// solver functions

#ifdef __cplusplus
}
#endif
