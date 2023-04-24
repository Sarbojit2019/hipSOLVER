
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

typedef enum {
  ONEMKL_UPLO_UPPER,
  ONEMKL_UPLO_LOWER
} onemklUplo;

typedef enum {
  ONEMKL_JOB_NOVEC,
  ONEMKL_JOB_VEC
} onemklJob;

// helper functions
hipsolverStatus_t sycl_create_handle(syclHandle_t* handle);
hipsolverStatus_t sycl_destroy_handle(syclHandle_t handle);
hipsolverStatus_t sycl_set_hipstream(syclHandle_t handle,
                                  unsigned long const* lzHandles,
                                  int                  nHandles,
                                   hipStream_t          stream,
                                   const char*          hipBlasBackendName);
hipsolverStatus_t sycl_get_hipstream(syclHandle_t handle, hipStream_t* pStream);
syclQueue_t sycl_get_queue(syclHandle_t handle);

// solver functions
int64_t onemkl_Sgebrd_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);
int64_t onemkl_Dgebrd_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);
int64_t onemkl_Cgebrd_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);
int64_t onemkl_Zgebrd_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);
void onemkl_Sgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float* a, int64_t lda,
                   float* d, float* e, float* tauq, float* taup, float* scratchpad, int64_t scratchpad_size);
void onemkl_Dgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double* a, int64_t lda,
                   double* d, double* e, double* tauq, double* taup, double* scratchpad, int64_t scratchpad_size);
void onemkl_Cgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex* a, int64_t lda,
                   float* d, float* e, float _Complex* tauq, float _Complex* taup, float _Complex* scratchpad,
                   int64_t scratchpad_size);
void onemkl_Zgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex* a, int64_t lda,
                   double* d, double* e, double _Complex* tauq, double _Complex* taup, double _Complex* scratchpad,
                   int64_t scratchpad_size);

int64_t onemkl_Sgesvd_ScPadSz(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                             int64_t lda, int64_t ldu, int64_t ldvt);
int64_t onemkl_Dgesvd_ScPadSz(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                             int64_t lda, int64_t ldu, int64_t ldvt);
int64_t onemkl_Cgesvd_ScPadSz(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                             int64_t lda, int64_t ldu, int64_t ldvt);
int64_t onemkl_Zgesvd_ScPadSz(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                             int64_t lda, int64_t ldu, int64_t ldvt);
void onemkl_Sgesvd(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n, float* A, int64_t lda,
                    float* S, float* U, int64_t ldu, float* V, int64_t ldv, float* scratchpad, int64_t scratchpad_size);
void onemkl_Dgesvd(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n, double* A, int64_t lda,
                    double* S, double* U, int64_t ldu, double* V, int64_t ldv, double* scratchpad, int64_t scratchpad_size);
void onemkl_Cgesvd(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n, float _Complex* A, int64_t lda,
                    float* S, float _Complex* U, int64_t ldu, float _Complex* V, int64_t ldv, float _Complex* scratchpad, int64_t scratchpad_size);
void onemkl_Zgesvd(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n, double _Complex* A, int64_t lda,
                    double* S, double _Complex* U, int64_t ldu, double _Complex* V, int64_t ldv, double _Complex* scratchpad, int64_t scratchpad_size);

int64_t onemkl_Ssyevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda);
int64_t onemkl_Dsyevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda);
void onemkl_Ssyevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* w, float* scratchpad, int64_t scratchpad_size);
void onemkl_Dsyevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* w, double* scratchpad, int64_t scratchpad_size);

int64_t onemkl_Cheevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda);
int64_t onemkl_Zheevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda);
void onemkl_Cheevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float* w,
                   float _Complex* scratchpad, int64_t scratchpad_size);
void onemkl_Zheevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double* w,
                  double _Complex* scratchpad, int64_t scratchpad_size);
#ifdef __cplusplus
}
#endif
