#include <hip/hip_interop.h>
#include "hipsolver.h"
#include "exceptions.hpp"

#include <unordered_set>
#include "deps/sycl_solver.h"
#include <iostream>
std::unordered_set<hipsolverHandle_t*> solverHandleTbl;

bool isvalid(hipsolverEigType_t t) {
    if (t == HIPSOLVER_EIG_TYPE_1 || t == HIPSOLVER_EIG_TYPE_2 || t == HIPSOLVER_EIG_TYPE_3)
        return true;
    return false;
}
inline int64_t convert(hipsolverEigType_t t) {
  switch(t){
    case HIPSOLVER_EIG_TYPE_1:
      return 1;
    case HIPSOLVER_EIG_TYPE_2:
      return 2;
    case HIPSOLVER_EIG_TYPE_3:
      return 3;
    default:
      return -1; // error: Never come here
  }
}

bool isValidMode(hipsolverSideMode_t s) {
    if (s != HIPSOLVER_SIDE_LEFT && s != HIPSOLVER_SIDE_RIGHT){
        return false;
    }
    return true;
}
inline onemklGen convertToGen(hipsolverSideMode_t s) {
    std::cout<<"hipsolverSideMode_t :"<<s<<std::endl;
    switch(s){
        case HIPSOLVER_SIDE_LEFT: return ONEMKL_GEN_Q;
        case HIPSOLVER_SIDE_RIGHT: return ONEMKL_GEN_P;
    }
}

inline onemklSide convert(hipsolverSideMode_t s) {
    switch(s){
        case HIPSOLVER_SIDE_LEFT: return ONEMKL_SIDE_LEFT;
        case HIPSOLVER_SIDE_RIGHT: return ONEMKL_SIDE_RIGHT;
    }
}
bool isvalid(hipsolverEigMode_t j){
    if (j == HIPSOLVER_EIG_MODE_NOVECTOR || j == HIPSOLVER_EIG_MODE_VECTOR)
        return true;
    return false;
}
inline onemklJob convert(hipsolverEigMode_t job) {
  switch(job) {
    case HIPSOLVER_EIG_MODE_NOVECTOR: return ONEMKL_JOB_NOVEC;
    case HIPSOLVER_EIG_MODE_VECTOR: return ONEMKL_JOB_VEC;
  }
}

bool isvalid(hipsolverFillMode_t v){
    if (v == HIPSOLVER_FILL_MODE_UPPER || v == HIPSOLVER_FILL_MODE_LOWER)
        return true;
    return false;
}
inline onemklUplo convert(hipsolverFillMode_t val) {
    switch(val) {
        case HIPSOLVER_FILL_MODE_UPPER:
            return ONEMKL_UPLO_UPPER;
        case HIPSOLVER_FILL_MODE_LOWER:
            return ONEMKL_UPLO_LOWER;
    }
}
bool isvalid(hipsolverOperation_t t){
    if (t == HIPSOLVER_OP_T || t == HIPSOLVER_OP_C || t == HIPSOLVER_OP_N)
        return true;
    return false;
}
onemklTranspose convert(hipsolverOperation_t val) {
    switch(val) {
        case HIPSOLVER_OP_T:
            return ONEMKL_TRANSPOSE_TRANS;
        case HIPSOLVER_OP_C:
            return ONEMLK_TRANSPOSE_CONJTRANS;
        case HIPSOLVER_OP_N:
        default:
            return ONEMKL_TRANSPOSE_NONTRANS;
    }
}

// local functions
static hipsolverStatus_t updateSyclHandleToCrrStream(hipStream_t stream, syclHandle_t syclHandle)
{
    // Obtain the handles from HIP backend.
    unsigned long bkHandles[4];
    int           nHandles = 4;
    hipGetBackendNativeHandles((uintptr_t)stream, bkHandles, &nHandles);

    auto backendName = hipGetBackendName();

    //Fix-Me : Should Sycl know hipStream_t??
    sycl_set_hipstream(syclHandle, bkHandles, nHandles, stream, backendName);
    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle)
try
{
    // create syclHandle
    sycl_create_handle((syclHandle_t*)handle);

    hipStream_t nullStream = NULL; // default or null stream
    // set stream to default NULL stream
    auto status = updateSyclHandleToCrrStream(nullStream, (syclHandle_t)*handle);
    solverHandleTbl.insert(handle);
    return status;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle)
try
{
    return sycl_destroy_handle((syclHandle_t)handle);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle,
                                     hipStream_t       streamId)
try
{
    return updateSyclHandleToCrrStream(streamId, (syclHandle_t)handle);
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle,
                                     hipStream_t*      streamId)
try
{
    if(handle == nullptr)
    {
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    }
    return sycl_get_hipstream((syclHandle_t)handle, streamId);
}
catch(...)
{
	return exception2hip_status();
}

// gesvdj params
hipsolverStatus_t hipsolverCreateGesvdjInfo(hipsolverGesvdjInfo_t* info)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDestroyGesvdjInfo(hipsolverGesvdjInfo_t info)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjSetMaxSweeps(hipsolverGesvdjInfo_t info,
                                                                int                   max_sweeps)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjSetSortEig(hipsolverGesvdjInfo_t info,
                                                              int                   sort_eig)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjSetTolerance(hipsolverGesvdjInfo_t info,
                                                                double                tolerance)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjGetResidual(hipsolverHandle_t     handle,
                                                               hipsolverGesvdjInfo_t info,
                                                               double*               residual)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjGetSweeps(hipsolverHandle_t     handle,
                                                             hipsolverGesvdjInfo_t info,
                                                             int*                  executed_sweeps)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// syevj params
hipsolverStatus_t hipsolverCreateSyevjInfo(hipsolverSyevjInfo_t* info)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDestroySyevjInfo(hipsolverSyevjInfo_t info)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjSetMaxSweeps(hipsolverSyevjInfo_t info,
                                                               int                  max_sweeps)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjSetSortEig(hipsolverSyevjInfo_t info,
                                                             int                  sort_eig)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjSetTolerance(hipsolverSyevjInfo_t info,
                                                               double               tolerance)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjGetResidual(hipsolverHandle_t    handle,
                                                              hipsolverSyevjInfo_t info,
                                                              double*              residual)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjGetSweeps(hipsolverHandle_t    handle,
                                                            hipsolverSyevjInfo_t info,
                                                            int*                 executed_sweeps)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// orgbr/ungbr
hipsolverStatus_t hipsolverSorgbr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverSideMode_t side,
                                            int                 m,
                                            int                 n,
                                            int                 k,
                                            float*              A,
                                            int                 lda,
                                            float*              tau,
                                            int*                lwork)
try
{
    std::cout<<"handle :"<<handle<<std::endl;
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isValidMode(side))
        return HIPSOLVER_STATUS_INVALID_ENUM; 
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    std::cout<<m<<" |"<<n<<" |"<<k<<" |"<<lda<<" |"<<lwork<<std::endl;
    *lwork = (int)onemkl_Sorgbr_ScPadSz(queue, convertToGen(side), m,n,k,lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgbr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverSideMode_t side,
                                                              int                 m,
                                                              int                 n,
                                                              int                 k,
                                                              double*             A,
                                                              int                 lda,
                                                              double*             tau,
                                                              int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Dorgbr_ScPadSz(queue, convertToGen(side), m,n,k,lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCungbr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverSideMode_t side,
                                                              int                 m,
                                                              int                 n,
                                                              int                 k,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              hipFloatComplex*    tau,
                                                              int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Cungbr_ScPadSz(queue, convertToGen(side), m,n,k,lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZungbr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverSideMode_t side,
                                                              int                 m,
                                                              int                 n,
                                                              int                 k,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              hipDoubleComplex*   tau,
                                                              int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Zungbr_ScPadSz(queue, convertToGen(side), m, n, k, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSorgbr(hipsolverHandle_t   handle,
                                hipsolverSideMode_t side,
                                int                 m,
                                int                 n,
                                int                 k,
                                float*              A,
                                int                 lda,
                                float*              tau,
                                float*              work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isValidMode(side))
        return HIPSOLVER_STATUS_INVALID_ENUM; 
    if (A == nullptr || tau == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, &lwork);
    hipHostMalloc(&work, lwork, 0);

    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    std::cout<<m<<" |"<<n<<" |"<<k<<" |"<<lda<<" |"<<lwork<<std::endl;
    onemkl_Sorgbr(queue, convertToGen(side), m, n, k, A, lda, tau, work, lwork);
    hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgbr(hipsolverHandle_t   handle,
                                hipsolverSideMode_t side,
                                int                 m,
                                int                 n,
                                int                 k,
                                double*             A,
                                int                 lda,
                                double*             tau,
                                double*             work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dorgbr(queue, convertToGen(side), m, n, k, A, lda, tau, work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCungbr(hipsolverHandle_t   handle,
                                hipsolverSideMode_t side,
                                int                 m,
                                int                 n,
                                int                 k,
                                hipFloatComplex*    A,
                                int                 lda,
                                hipFloatComplex*    tau,
                                hipFloatComplex*    work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cungbr(queue, convertToGen(side), m, n, k,(float _Complex*)A, lda,
                (float _Complex*)tau, (float _Complex*)work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZungbr(hipsolverHandle_t   handle,
                                hipsolverSideMode_t side,
                                int                 m,
                                int                 n,
                                int                 k,
                                hipDoubleComplex*   A,
                                int                 lda,
                                hipDoubleComplex*   tau,
                                hipDoubleComplex*   work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zungbr(queue, convertToGen(side), m, n, k,(double _Complex*)A, lda,
                (double _Complex*)tau, (double _Complex*)work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// orgqr/ungqr
hipsolverStatus_t hipsolverSorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, float* A, int lda, float* tau, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Sorgqr_ScPadSz(queue, m, n, k, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Dorgqr_ScPadSz(queue, m, n, k, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCungqr_bufferSize(hipsolverHandle_t handle,
                                            int               m,
                                            int               n,
                                            int               k,
                                            hipFloatComplex*  A,
                                            int               lda,
                                            hipFloatComplex*  tau,
                                            int*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Cungqr_ScPadSz(queue, m, n, k, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZungqr_bufferSize(hipsolverHandle_t handle,
                                            int               m,
                                            int               n,
                                            int               k,
                                            hipDoubleComplex* A,
                                            int               lda,
                                            hipDoubleComplex* tau,
                                            int*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Zungqr_ScPadSz(queue, m, n, k, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSorgqr(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                int               k,
                                float*            A,
                                int               lda,
                                float*            tau,
                                float*            work,
                                int               lwork,
                                int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0) {
        lwork = 0;
        hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sorgqr(queue, m, n, k, A, lda, tau, work, lwork);
    if(allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgqr(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               k,
                                                   double*           A,
                                                   int               lda,
                                                   double*           tau,
                                                   double*           work,
                                                   int               lwork,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dorgqr(queue, m, n, k, A, lda, tau, work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCungqr(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               k,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   hipFloatComplex*  tau,
                                                   hipFloatComplex*  work,
                                                   int               lwork,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverCungqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cungqr(queue, m, n, k,(float _Complex*)A, lda,
                (float _Complex*)tau, (float _Complex*)work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZungqr(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               k,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   hipDoubleComplex* tau,
                                                   hipDoubleComplex* work,
                                                   int               lwork,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverZungqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zungqr(queue, m, n, k,(double _Complex*)A, lda,
                (double _Complex*)tau, (double _Complex*)work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// orgtr/ungtr
hipsolverStatus_t hipsolverSorgtr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            float*              A,
                                            int                 lda,
                                            float*              tau,
                                            int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Sorgtr_ScPadSz(queue, convert(uplo), n, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgtr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            double*             A,
                                            int                 lda,
                                            double*             tau,
                                            int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Dorgtr_ScPadSz(queue, convert(uplo), n, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCungtr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            hipFloatComplex*    A,
                                            int                 lda,
                                            hipFloatComplex*    tau,
                                            int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Cungtr_ScPadSz(queue, convert(uplo), n, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZungtr_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            hipDoubleComplex*   A,
                                            int                 lda,
                                            hipDoubleComplex*   tau,
                                            int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Zungtr_ScPadSz(queue, convert(uplo), n, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSorgtr(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                float*              A,
                                int                 lda,
                                float*              tau,
                                float*              work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || tau == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    auto status = hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS){
        return status;
    }
    auto hipStatus = hipHostMalloc(&work, lwork);
    hipStatus = hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sorgtr(queue, convert(uplo), n, A, lda, tau, work, lwork);
    hipStatus = hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgtr(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                double*             A,
                                int                 lda,
                                double*             tau,
                                double*             work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dorgtr(queue, convert(uplo), n, A, lda, tau, work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCungtr(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                hipFloatComplex*    A,
                                int                 lda,
                                hipFloatComplex*    tau,
                                hipFloatComplex*    work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverCungtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cungtr(queue, convert(uplo), n, (float _Complex*)A, lda,
                (float _Complex*)tau, (float _Complex*)work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZungtr(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                hipDoubleComplex*   A,
                                int                 lda,
                                hipDoubleComplex*   tau,
                                hipDoubleComplex*   work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverZungtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zungtr(queue, convert(uplo), n, (double _Complex*)A, lda,
                (double _Complex*)tau, (double _Complex*)work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// ormqr/unmqr
hipsolverStatus_t hipsolverSormqr_bufferSize(hipsolverHandle_t    handle,
                                            hipsolverSideMode_t  side,
                                            hipsolverOperation_t trans,
                                            int                  m,
                                            int                  n,
                                            int                  k,
                                            float*               A,
                                            int                  lda,
                                            float*               tau,
                                            float*               C,
                                            int                  ldc,
                                            int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans) || !isValidMode(side))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Sormqr_ScPadSz(queue, convert(side), convert(trans), m, n, k, lda, ldc);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDormqr_bufferSize(hipsolverHandle_t    handle,
                                            hipsolverSideMode_t  side,
                                            hipsolverOperation_t trans,
                                            int                  m,
                                            int                  n,
                                            int                  k,
                                            double*              A,
                                            int                  lda,
                                            double*              tau,
                                            double*              C,
                                            int                  ldc,
                                            int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans) || !isValidMode(side))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Dormqr_ScPadSz(queue, convert(side), convert(trans), m, n, k, lda, ldc);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCunmqr_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverSideMode_t  side,
                                                              hipsolverOperation_t trans,
                                                              int                  m,
                                                              int                  n,
                                                              int                  k,
                                                              hipFloatComplex*     A,
                                                              int                  lda,
                                                              hipFloatComplex*     tau,
                                                              hipFloatComplex*     C,
                                                              int                  ldc,
                                                              int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans) || !isValidMode(side))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Cunmqr_ScPadSz(queue, convert(side), convert(trans), m, n, k, lda, ldc);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZunmqr_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverSideMode_t  side,
                                                              hipsolverOperation_t trans,
                                                              int                  m,
                                                              int                  n,
                                                              int                  k,
                                                              hipDoubleComplex*    A,
                                                              int                  lda,
                                                              hipDoubleComplex*    tau,
                                                              hipDoubleComplex*    C,
                                                              int                  ldc,
                                                              int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans) || !isValidMode(side))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Zunmqr_ScPadSz(queue, convert(side), convert(trans), m, n, k, lda, ldc);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSormqr(hipsolverHandle_t    handle,
                                hipsolverSideMode_t  side,
                                hipsolverOperation_t trans,
                                int                  m,
                                int                  n,
                                int                  k,
                                float*               A,
                                int                  lda,
                                float*               tau,
                                float*               C,
                                int                  ldc,
                                float*               work,
                                int                  lwork,
                                int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isValidMode(side) || !isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    hipError_t hipStatus;
    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = 0;
        auto status = hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS){
            return status;
        }
        hipStatus = hipHostMalloc(&work, lwork);
        allocate = true;
    }
    hipStatus = hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sormqr(queue, convert(side), convert(trans), m, n, k, A, lda, tau, C, ldc, work, lwork);
    if (allocate)
        hipStatus = hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    std::cout<<"test error\n";
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDormqr(hipsolverHandle_t    handle,
                                                   hipsolverSideMode_t  side,
                                                   hipsolverOperation_t trans,
                                                   int                  m,
                                                   int                  n,
                                                   int                  k,
                                                   double*              A,
                                                   int                  lda,
                                                   double*              tau,
                                                   double*              C,
                                                   int                  ldc,
                                                   double*              work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isValidMode(side) || !isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    hipError_t hipStatus;
    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = 0;
        auto status = hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS){
            return status;
        }
        hipStatus = hipHostMalloc(&work, lwork);
        allocate = true;
    }
    hipStatus = hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dormqr(queue, convert(side), convert(trans), m, n, k, A, lda, tau, C, ldc, work, lwork);
    if (allocate)
        hipStatus = hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCunmqr(hipsolverHandle_t    handle,
                                                   hipsolverSideMode_t  side,
                                                   hipsolverOperation_t trans,
                                                   int                  m,
                                                   int                  n,
                                                   int                  k,
                                                   hipFloatComplex*     A,
                                                   int                  lda,
                                                   hipFloatComplex*     tau,
                                                   hipFloatComplex*     C,
                                                   int                  ldc,
                                                   hipFloatComplex*     work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isValidMode(side) || !isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    hipError_t hipStatus;
    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = 0;
        auto status = hipsolverCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS){
            return status;
        }
        hipStatus = hipHostMalloc(&work, lwork);
        allocate = true;
    }
    hipStatus = hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cunmqr(queue, convert(side), convert(trans), m, n, k, (float _Complex*)A, lda, (float _Complex*)tau, (float _Complex*)C, ldc, (float _Complex*)work, lwork);
    if (allocate)
        hipStatus = hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZunmqr(hipsolverHandle_t    handle,
                                                   hipsolverSideMode_t  side,
                                                   hipsolverOperation_t trans,
                                                   int                  m,
                                                   int                  n,
                                                   int                  k,
                                                   hipDoubleComplex*    A,
                                                   int                  lda,
                                                   hipDoubleComplex*    tau,
                                                   hipDoubleComplex*    C,
                                                   int                  ldc,
                                                   hipDoubleComplex*    work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isValidMode(side) || !isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    hipError_t hipStatus;
    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = 0;
        auto status = hipsolverZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS){
            return status;
        }
        hipStatus = hipHostMalloc(&work, lwork);
        allocate = true;
    }
    hipStatus = hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zunmqr(queue, convert(side), convert(trans), m, n, k, (double _Complex*)A, lda, (double _Complex*)tau, (double _Complex*)C, ldc, (double _Complex*)work, lwork);
    if (allocate)
        hipStatus = hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// ormtr/unmtr
hipsolverStatus_t hipsolverSormtr_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverSideMode_t  side,
                                                              hipsolverFillMode_t  uplo,
                                                              hipsolverOperation_t trans,
                                                              int                  m,
                                                              int                  n,
                                                              float*               A,
                                                              int                  lda,
                                                              float*               tau,
                                                              float*               C,
                                                              int                  ldc,
                                                              int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans) || !isValidMode(side) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Sormtr_ScPadSz(queue, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDormtr_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverSideMode_t  side,
                                                              hipsolverFillMode_t  uplo,
                                                              hipsolverOperation_t trans,
                                                              int                  m,
                                                              int                  n,
                                                              double*              A,
                                                              int                  lda,
                                                              double*              tau,
                                                              double*              C,
                                                              int                  ldc,
                                                              int*                 lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCunmtr_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverSideMode_t  side,
                                                              hipsolverFillMode_t  uplo,
                                                              hipsolverOperation_t trans,
                                                              int                  m,
                                                              int                  n,
                                                              hipFloatComplex*     A,
                                                              int                  lda,
                                                              hipFloatComplex*     tau,
                                                              hipFloatComplex*     C,
                                                              int                  ldc,
                                                              int*                 lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZunmtr_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverSideMode_t  side,
                                                              hipsolverFillMode_t  uplo,
                                                              hipsolverOperation_t trans,
                                                              int                  m,
                                                              int                  n,
                                                              hipDoubleComplex*    A,
                                                              int                  lda,
                                                              hipDoubleComplex*    tau,
                                                              hipDoubleComplex*    C,
                                                              int                  ldc,
                                                              int*                 lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSormtr(hipsolverHandle_t    handle,
                                                   hipsolverSideMode_t  side,
                                                   hipsolverFillMode_t  uplo,
                                                   hipsolverOperation_t trans,
                                                   int                  m,
                                                   int                  n,
                                                   float*               A,
                                                   int                  lda,
                                                   float*               tau,
                                                   float*               C,
                                                   int                  ldc,
                                                   float*               work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isValidMode(side) || !isvalid(trans) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || tau == nullptr  || C == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    hipError_t hipStatus;
    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = 0;
        auto status = hipsolverSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS){
            return status;
        }
        hipStatus = hipHostMalloc(&work, lwork);
        allocate = true;
    }
    hipStatus = hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sormtr(queue, convert(side), convert(uplo), convert(trans), m, n, A, lda, tau, C, ldc, work, lwork);
    if (allocate)
        hipStatus = hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;    
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDormtr(hipsolverHandle_t    handle,
                                                   hipsolverSideMode_t  side,
                                                   hipsolverFillMode_t  uplo,
                                                   hipsolverOperation_t trans,
                                                   int                  m,
                                                   int                  n,
                                                   double*              A,
                                                   int                  lda,
                                                   double*              tau,
                                                   double*              C,
                                                   int                  ldc,
                                                   double*              work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCunmtr(hipsolverHandle_t    handle,
                                                   hipsolverSideMode_t  side,
                                                   hipsolverFillMode_t  uplo,
                                                   hipsolverOperation_t trans,
                                                   int                  m,
                                                   int                  n,
                                                   hipFloatComplex*     A,
                                                   int                  lda,
                                                   hipFloatComplex*     tau,
                                                   hipFloatComplex*     C,
                                                   int                  ldc,
                                                   hipFloatComplex*     work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZunmtr(hipsolverHandle_t    handle,
                                                   hipsolverSideMode_t  side,
                                                   hipsolverFillMode_t  uplo,
                                                   hipsolverOperation_t trans,
                                                   int                  m,
                                                   int                  n,
                                                   hipDoubleComplex*    A,
                                                   int                  lda,
                                                   hipDoubleComplex*    tau,
                                                   hipDoubleComplex*    C,
                                                   int                  ldc,
                                                   hipDoubleComplex*    work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// gebrd
hipsolverStatus_t hipsolverSgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Sgebrd_ScPadSz(queue, m, n, *lwork);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dgebrd_ScPadSz(queue, m, n, *lwork);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Cgebrd_ScPadSz(queue, m, n, *lwork);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgebrd_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Zgebrd_ScPadSz(queue, m, n, *lwork);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSgebrd(hipsolverHandle_t handle,
                                 int               m,
                                 int               n,
                                 float*            A,
                                 int               lda,
                                 float*            D,
                                 float*            E,
                                 float*            tauq,
                                 float*            taup,
                                 float*            work,
                                 int               lwork,
                                 int*              devInfo)
try
{
    if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = lda;
        auto status = hipsolverSgebrd_bufferSize(handle, m, n, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS)
            return status;
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sgebrd(queue, m, n, A, lda, D, E, tauq, taup, work, lwork);
    
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgebrd(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                double*           A,
                                int               lda,
                                double*           D,
                                double*           E,
                                double*           tauq,
                                double*           taup,
                                double*           work,
                                int               lwork,
                                int*              devInfo)
try
{
    if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = lda;
        auto status = hipsolverDgebrd_bufferSize(handle, m, n, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS)
            return status;
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dgebrd(queue, m, n, A, lda, D, E, tauq, taup, work, lwork);
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgebrd(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                hipFloatComplex*  A,
                                int               lda,
                                float*            D,
                                float*            E,
                                hipFloatComplex*  tauq,
                                hipFloatComplex*  taup,
                                hipFloatComplex*  work,
                                int               lwork,
                                int*              devInfo)
try
{
    if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = lda;
        auto status = hipsolverCgebrd_bufferSize(handle, m, n, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS)
            return status;
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cgebrd(queue, m, n, (float _Complex*)A, lda, D, E,
                 (float _Complex*)tauq, (float _Complex*)taup, (float _Complex*)work, lwork);

    if (allocate) hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgebrd(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                hipDoubleComplex* A,
                                int               lda,
                                double*           D,
                                double*           E,
                                hipDoubleComplex* tauq,
                                hipDoubleComplex* taup,
                                hipDoubleComplex* work,
                                int               lwork,
                                int*              devInfo)
try
{
    if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = lda;
        auto status = hipsolverSgebrd_bufferSize(handle, m, n, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS)
            return status;
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zgebrd(queue, m, n, (double _Complex*)A, lda, D, E,
                 (double _Complex*)tauq, (double _Complex*)taup, (double _Complex*)work, lwork);
    if (allocate) hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// gels
hipsolverStatus_t hipsolverSSgels_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               nrhs,
                                                              float*            A,
                                                              int               lda,
                                                              float*            B,
                                                              int               ldb,
                                                              float*            X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDDgels_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               nrhs,
                                                              double*           A,
                                                              int               lda,
                                                              double*           B,
                                                              int               ldb,
                                                              double*           X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCCgels_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               nrhs,
                                                              hipFloatComplex*  A,
                                                              int               lda,
                                                              hipFloatComplex*  B,
                                                              int               ldb,
                                                              hipFloatComplex*  X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZZgels_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               nrhs,
                                                              hipDoubleComplex* A,
                                                              int               lda,
                                                              hipDoubleComplex* B,
                                                              int               ldb,
                                                              hipDoubleComplex* X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSSgels(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   float*            A,
                                                   int               lda,
                                                   float*            B,
                                                   int               ldb,
                                                   float*            X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDDgels(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   double*           A,
                                                   int               lda,
                                                   double*           B,
                                                   int               ldb,
                                                   double*           X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCCgels(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   hipFloatComplex*  B,
                                                   int               ldb,
                                                   hipFloatComplex*  X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZZgels(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   hipDoubleComplex* B,
                                                   int               ldb,
                                                   hipDoubleComplex* X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// geqrf
hipsolverStatus_t hipsolverSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Sgeqrf_ScPadSz(queue, m, n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dgeqrf_ScPadSz(queue, m, n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Cgeqrf_ScPadSz(queue, m, n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dgeqrf_ScPadSz(queue, m, n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSgeqrf(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                float*            A,
                                int               lda,
                                float*            tau,
                                float*            work,
                                int               lwork,
                                int*              devInfo)
try
{
    if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = lda;
        auto status = hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS)
            return status;
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sgeqrf(queue, m, n, A, lda, tau, work, lwork);
    
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   double*           A,
                                                   int               lda,
                                                   double*           tau,
                                                   double*           work,
                                                   int               lwork,
                                                   int*              devInfo)
try
{
    if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = lda;
        auto status = hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS)
            return status;
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dgeqrf(queue, m, n, A, lda, tau, work, lwork);
    
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   hipFloatComplex*  tau,
                                                   hipFloatComplex*  work,
                                                   int               lwork,
                                                   int*              devInfo)
try
{
    if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = lda;
        auto status = hipsolverCgeqrf_bufferSize(handle, m, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS)
            return status;
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cgeqrf(queue, m, n, (float _Complex*)A, lda, (float _Complex*)tau, (float _Complex*)work, lwork);
    
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   hipDoubleComplex* tau,
                                                   hipDoubleComplex* work,
                                                   int               lwork,
                                                   int*              devInfo)
try
{
    if (!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || tau == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        lwork = lda;
        auto status = hipsolverZgeqrf_bufferSize(handle, m, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS)
            return status;
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }
    hipMemset(devInfo, 0, sizeof(int));
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zgeqrf(queue, m, n, (double _Complex*)A, lda, (double _Complex*)tau, (double _Complex*)work, lwork);
    
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// gesv
hipsolverStatus_t hipsolverSSgesv_bufferSize(hipsolverHandle_t handle,
                                                              int               n,
                                                              int               nrhs,
                                                              float*            A,
                                                              int               lda,
                                                              int*              devIpiv,
                                                              float*            B,
                                                              int               ldb,
                                                              float*            X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDDgesv_bufferSize(hipsolverHandle_t handle,
                                                              int               n,
                                                              int               nrhs,
                                                              double*           A,
                                                              int               lda,
                                                              int*              devIpiv,
                                                              double*           B,
                                                              int               ldb,
                                                              double*           X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCCgesv_bufferSize(hipsolverHandle_t handle,
                                                              int               n,
                                                              int               nrhs,
                                                              hipFloatComplex*  A,
                                                              int               lda,
                                                              int*              devIpiv,
                                                              hipFloatComplex*  B,
                                                              int               ldb,
                                                              hipFloatComplex*  X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZZgesv_bufferSize(hipsolverHandle_t handle,
                                                              int               n,
                                                              int               nrhs,
                                                              hipDoubleComplex* A,
                                                              int               lda,
                                                              int*              devIpiv,
                                                              hipDoubleComplex* B,
                                                              int               ldb,
                                                              hipDoubleComplex* X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSSgesv(hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   float*            A,
                                                   int               lda,
                                                   int*              devIpiv,
                                                   float*            B,
                                                   int               ldb,
                                                   float*            X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDDgesv(hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   double*           A,
                                                   int               lda,
                                                   int*              devIpiv,
                                                   double*           B,
                                                   int               ldb,
                                                   double*           X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCCgesv(hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   int*              devIpiv,
                                                   hipFloatComplex*  B,
                                                   int               ldb,
                                                   hipFloatComplex*  X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZZgesv(hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   int*              devIpiv,
                                                   hipDoubleComplex* B,
                                                   int               ldb,
                                                   hipDoubleComplex* X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// gesvd
hipsolverStatus_t hipsolverSgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork)
try
{
    // parameter mismatch
    lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSgesvd(hipsolverHandle_t handle,
                                                   signed char       jobu,
                                                   signed char       jobv,
                                                   int               m,
                                                   int               n,
                                                   float*            A,
                                                   int               lda,
                                                   float*            S,
                                                   float*            U,
                                                   int               ldu,
                                                   float*            V,
                                                   int               ldv,
                                                   float*            work,
                                                   int               lwork,
                                                   float*            rwork,
                                                   int*              devInfo)
try
{
    if (handle == nullptr) return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (A == nullptr || S == nullptr || m < 0 || n < 0) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    if (U == nullptr || V == nullptr )
        return HIPSOLVER_STATUS_SUCCESS;

    auto queue = sycl_get_queue((syclHandle_t)handle);
    bool allocate = false;
    if (work == nullptr || lwork == 0 || true){
        lwork = (int)onemkl_Sgesvd_ScPadSz(queue, jobu, jobv, m, n, lda, ldu, ldv);
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    hipMemset(devInfo, 0, sizeof(int));    

    onemkl_Sgesvd(queue, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork);
    if(allocate){
        hipFree(work);
    }

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvd(hipsolverHandle_t handle,
                                                   signed char       jobu,
                                                   signed char       jobv,
                                                   int               m,
                                                   int               n,
                                                   double*           A,
                                                   int               lda,
                                                   double*           S,
                                                   double*           U,
                                                   int               ldu,
                                                   double*           V,
                                                   int               ldv,
                                                   double*           work,
                                                   int               lwork,
                                                   double*           rwork,
                                                   int*              devInfo)
try
{
    if (A == nullptr || S == nullptr || U == nullptr || V == nullptr || work == nullptr || rwork == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    auto queue = sycl_get_queue((syclHandle_t)handle);
    lwork = (int)onemkl_Dgesvd_ScPadSz(queue, jobu, jobv, m, n, lda, ldu, ldv);
    onemkl_Dgesvd(queue, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvd(hipsolverHandle_t handle,
                                                   signed char       jobu,
                                                   signed char       jobv,
                                                   int               m,
                                                   int               n,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   float*            S,
                                                   hipFloatComplex*  U,
                                                   int               ldu,
                                                   hipFloatComplex*  V,
                                                   int               ldv,
                                                   hipFloatComplex*  work,
                                                   int               lwork,
                                                   float*            rwork,
                                                   int*              devInfo)
try
{
    if (A == nullptr || S == nullptr || U == nullptr || V == nullptr || work == nullptr || rwork == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    auto queue = sycl_get_queue((syclHandle_t)handle);
    lwork = (int)onemkl_Cgesvd_ScPadSz(queue, jobu, jobv, m, n, lda, ldu, ldv);
    onemkl_Cgesvd(queue, jobu, jobv, m, n, (float _Complex*)A, lda, S, (float _Complex*)U, ldu,
                 (float _Complex*)V, ldv, (float _Complex*)work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvd(hipsolverHandle_t handle,
                                                   signed char       jobu,
                                                   signed char       jobv,
                                                   int               m,
                                                   int               n,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   double*           S,
                                                   hipDoubleComplex* U,
                                                   int               ldu,
                                                   hipDoubleComplex* V,
                                                   int               ldv,
                                                   hipDoubleComplex* work,
                                                   int               lwork,
                                                   double*           rwork,
                                                   int*              devInfo)
try
{
    if (A == nullptr || S == nullptr || U == nullptr || V == nullptr || work == nullptr || rwork == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    auto queue = sycl_get_queue((syclHandle_t)handle);
    lwork = (int)onemkl_Zgesvd_ScPadSz(queue, jobu, jobv, m, n, lda, ldu, ldv);
    onemkl_Zgesvd(queue, jobu, jobv, m, n, (double _Complex*)A, lda, S, (double _Complex*)U, ldu,
                 (double _Complex*)V, ldv, (double _Complex*)work, lwork);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// gesvdj
hipsolverStatus_t hipsolverSgesvdj_bufferSize(hipsolverHandle_t     handle,
                                                               hipsolverEigMode_t    jobz,
                                                               int                   econ,
                                                               int                   m,
                                                               int                   n,
                                                               const float*          A,
                                                               int                   lda,
                                                               const float*          S,
                                                               const float*          U,
                                                               int                   ldu,
                                                               const float*          V,
                                                               int                   ldv,
                                                               int*                  lwork,
                                                               hipsolverGesvdjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvdj_bufferSize(hipsolverHandle_t     handle,
                                                               hipsolverEigMode_t    jobz,
                                                               int                   econ,
                                                               int                   m,
                                                               int                   n,
                                                               const double*         A,
                                                               int                   lda,
                                                               const double*         S,
                                                               const double*         U,
                                                               int                   ldu,
                                                               const double*         V,
                                                               int                   ldv,
                                                               int*                  lwork,
                                                               hipsolverGesvdjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvdj_bufferSize(hipsolverHandle_t      handle,
                                                               hipsolverEigMode_t     jobz,
                                                               int                    econ,
                                                               int                    m,
                                                               int                    n,
                                                               const hipFloatComplex* A,
                                                               int                    lda,
                                                               const float*           S,
                                                               const hipFloatComplex* U,
                                                               int                    ldu,
                                                               const hipFloatComplex* V,
                                                               int                    ldv,
                                                               int*                   lwork,
                                                               hipsolverGesvdjInfo_t  params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvdj_bufferSize(hipsolverHandle_t       handle,
                                                               hipsolverEigMode_t      jobz,
                                                               int                     econ,
                                                               int                     m,
                                                               int                     n,
                                                               const hipDoubleComplex* A,
                                                               int                     lda,
                                                               const double*           S,
                                                               const hipDoubleComplex* U,
                                                               int                     ldu,
                                                               const hipDoubleComplex* V,
                                                               int                     ldv,
                                                               int*                    lwork,
                                                               hipsolverGesvdjInfo_t   params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSgesvdj(hipsolverHandle_t     handle,
                                                    hipsolverEigMode_t    jobz,
                                                    int                   econ,
                                                    int                   m,
                                                    int                   n,
                                                    float*                A,
                                                    int                   lda,
                                                    float*                S,
                                                    float*                U,
                                                    int                   ldu,
                                                    float*                V,
                                                    int                   ldv,
                                                    float*                work,
                                                    int                   lwork,
                                                    int*                  devInfo,
                                                    hipsolverGesvdjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvdj(hipsolverHandle_t     handle,
                                                    hipsolverEigMode_t    jobz,
                                                    int                   econ,
                                                    int                   m,
                                                    int                   n,
                                                    double*               A,
                                                    int                   lda,
                                                    double*               S,
                                                    double*               U,
                                                    int                   ldu,
                                                    double*               V,
                                                    int                   ldv,
                                                    double*               work,
                                                    int                   lwork,
                                                    int*                  devInfo,
                                                    hipsolverGesvdjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvdj(hipsolverHandle_t     handle,
                                                    hipsolverEigMode_t    jobz,
                                                    int                   econ,
                                                    int                   m,
                                                    int                   n,
                                                    hipFloatComplex*      A,
                                                    int                   lda,
                                                    float*                S,
                                                    hipFloatComplex*      U,
                                                    int                   ldu,
                                                    hipFloatComplex*      V,
                                                    int                   ldv,
                                                    hipFloatComplex*      work,
                                                    int                   lwork,
                                                    int*                  devInfo,
                                                    hipsolverGesvdjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvdj(hipsolverHandle_t     handle,
                                                    hipsolverEigMode_t    jobz,
                                                    int                   econ,
                                                    int                   m,
                                                    int                   n,
                                                    hipDoubleComplex*     A,
                                                    int                   lda,
                                                    double*               S,
                                                    hipDoubleComplex*     U,
                                                    int                   ldu,
                                                    hipDoubleComplex*     V,
                                                    int                   ldv,
                                                    hipDoubleComplex*     work,
                                                    int                   lwork,
                                                    int*                  devInfo,
                                                    hipsolverGesvdjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// gesvdj_batched
hipsolverStatus_t hipsolverSgesvdjBatched_bufferSize(hipsolverHandle_t     handle,
                                                                      hipsolverEigMode_t    jobz,
                                                                      int                   m,
                                                                      int                   n,
                                                                      const float*          A,
                                                                      int                   lda,
                                                                      const float*          S,
                                                                      const float*          U,
                                                                      int                   ldu,
                                                                      const float*          V,
                                                                      int                   ldv,
                                                                      int*                  lwork,
                                                                      hipsolverGesvdjInfo_t params,
                                                                      int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvdjBatched_bufferSize(hipsolverHandle_t     handle,
                                                                      hipsolverEigMode_t    jobz,
                                                                      int                   m,
                                                                      int                   n,
                                                                      const double*         A,
                                                                      int                   lda,
                                                                      const double*         S,
                                                                      const double*         U,
                                                                      int                   ldu,
                                                                      const double*         V,
                                                                      int                   ldv,
                                                                      int*                  lwork,
                                                                      hipsolverGesvdjInfo_t params,
                                                                      int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvdjBatched_bufferSize(hipsolverHandle_t      handle,
                                                                      hipsolverEigMode_t     jobz,
                                                                      int                    m,
                                                                      int                    n,
                                                                      const hipFloatComplex* A,
                                                                      int                    lda,
                                                                      const float*           S,
                                                                      const hipFloatComplex* U,
                                                                      int                    ldu,
                                                                      const hipFloatComplex* V,
                                                                      int                    ldv,
                                                                      int*                   lwork,
                                                                      hipsolverGesvdjInfo_t  params,
                                                                      int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvdjBatched_bufferSize(hipsolverHandle_t  handle,
                                                                      hipsolverEigMode_t jobz,
                                                                      int                m,
                                                                      int                n,
                                                                      const hipDoubleComplex* A,
                                                                      int                     lda,
                                                                      const double*           S,
                                                                      const hipDoubleComplex* U,
                                                                      int                     ldu,
                                                                      const hipDoubleComplex* V,
                                                                      int                     ldv,
                                                                      int*                    lwork,
                                                                      hipsolverGesvdjInfo_t params,
                                                                      int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSgesvdjBatched(hipsolverHandle_t     handle,
                                                           hipsolverEigMode_t    jobz,
                                                           int                   m,
                                                           int                   n,
                                                           float*                A,
                                                           int                   lda,
                                                           float*                S,
                                                           float*                U,
                                                           int                   ldu,
                                                           float*                V,
                                                           int                   ldv,
                                                           float*                work,
                                                           int                   lwork,
                                                           int*                  devInfo,
                                                           hipsolverGesvdjInfo_t params,
                                                           int                   batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvdjBatched(hipsolverHandle_t     handle,
                                                           hipsolverEigMode_t    jobz,
                                                           int                   m,
                                                           int                   n,
                                                           double*               A,
                                                           int                   lda,
                                                           double*               S,
                                                           double*               U,
                                                           int                   ldu,
                                                           double*               V,
                                                           int                   ldv,
                                                           double*               work,
                                                           int                   lwork,
                                                           int*                  devInfo,
                                                           hipsolverGesvdjInfo_t params,
                                                           int                   batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvdjBatched(hipsolverHandle_t     handle,
                                                           hipsolverEigMode_t    jobz,
                                                           int                   m,
                                                           int                   n,
                                                           hipFloatComplex*      A,
                                                           int                   lda,
                                                           float*                S,
                                                           hipFloatComplex*      U,
                                                           int                   ldu,
                                                           hipFloatComplex*      V,
                                                           int                   ldv,
                                                           hipFloatComplex*      work,
                                                           int                   lwork,
                                                           int*                  devInfo,
                                                           hipsolverGesvdjInfo_t params,
                                                           int                   batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvdjBatched(hipsolverHandle_t     handle,
                                                           hipsolverEigMode_t    jobz,
                                                           int                   m,
                                                           int                   n,
                                                           hipDoubleComplex*     A,
                                                           int                   lda,
                                                           double*               S,
                                                           hipDoubleComplex*     U,
                                                           int                   ldu,
                                                           hipDoubleComplex*     V,
                                                           int                   ldv,
                                                           hipDoubleComplex*     work,
                                                           int                   lwork,
                                                           int*                  devInfo,
                                                           hipsolverGesvdjInfo_t params,
                                                           int                   batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// getrf
hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Sgetrf_ScPadSz(queue, m, n, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Dgetrf_ScPadSz(queue, m, n, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Cgetrf_ScPadSz(queue, m, n, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Zgetrf_ScPadSz(queue, m, n, lda);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSgetrf(hipsolverHandle_t handle,
                                int               m,
                                int               n,
                                float*            A,
                                int               lda,
                                float*            work,
                                int               lwork,
                                int*              devIpiv,
                                int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || devIpiv == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork == 0) {
        lwork = 0;
        hipsolverSgetrf_bufferSize(handle, m, n, A, lda, &lwork);
        hipHostMalloc(&work, lwork);
        allocate = true;
    }
    // WA: MKL does not use devInfo hence resetting it to zero
    hipMemset(devInfo, 0, sizeof(int));
    
    // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
    //     hence need special handling here. Force type cast was causing crash as 
    //     MKL's requirement is more.
    //     Note: It can have performance impact as there are extra copies and element wise copies are involved  
    int64_t* local_dIpiv;
    auto no_of_elements = max(1, min(m,n));
    // Allocating it on host with device access to avoid extra copy needed while accessing it from host
    hipHostMalloc(&local_dIpiv, sizeof(int64_t)* no_of_elements);

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sgetrf(queue, m, n, A, lda, local_dIpiv, work, lwork);

    int* local_hIpiv = (int*)malloc(sizeof(int)* no_of_elements);
    for(auto i=0; i< no_of_elements; ++i){
        local_hIpiv[i] = (int)local_dIpiv[i];
    }
    hipMemcpy(devIpiv, local_hIpiv, sizeof(int)* min(m,n), hipMemcpyHostToDevice);

    // release the memory allocated in the WA
    hipFree(local_dIpiv);
    free(local_hIpiv);

    if (allocate)
        hipFree(work);
    
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   double*           A,
                                                   int               lda,
                                                   double*           work,
                                                   int               lwork,
                                                   int*              devIpiv,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || devIpiv == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork == 0) {
        lwork = 0;
        hipsolverDgetrf_bufferSize(handle, m, n, A, lda, &lwork);
        hipHostMalloc(&work, lwork);
        allocate = true;
    }
    // WA: MKL does not use devInfo hence resetting it to zero
    hipMemset(devInfo, 0, sizeof(int));
    
    // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
    //     hence need special handling here. Force type cast was causing crash as 
    //     MKL's requirement is more.
    //     Note: It can have performance impact as there are extra copies and element wise copies are involved  
    int64_t* local_dIpiv;
    auto no_of_elements = max(1, min(m,n));
    // Allocating it on host with device access to avoid extra copy needed while accessing it from host
    hipHostMalloc(&local_dIpiv, sizeof(int64_t)* no_of_elements);

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dgetrf(queue, m, n, A, lda, local_dIpiv, work, lwork);

    int* local_hIpiv = (int*)malloc(sizeof(int)* no_of_elements);
    for(auto i=0; i< no_of_elements; ++i){
        local_hIpiv[i] = (int)local_dIpiv[i];
    }
    hipMemcpy(devIpiv, local_hIpiv, sizeof(int)* min(m,n), hipMemcpyHostToDevice);

    // release the memory allocated in the WA
    hipFree(local_dIpiv);
    free(local_hIpiv);
    
    if (allocate)
        hipFree(work);
    
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   hipFloatComplex*  work,
                                                   int               lwork,
                                                   int*              devIpiv,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || devIpiv == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork == 0) {
        lwork = 0;
        hipsolverCgetrf_bufferSize(handle, m, n, A, lda, &lwork);
        hipHostMalloc(&work, lwork);
        allocate = true;
    }
    // WA: MKL does not use devInfo hence resetting it to zero
    hipMemset(devInfo, 0, sizeof(int));
    
    // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
    //     hence need special handling here. Force type cast was causing crash as 
    //     MKL's requirement is more.
    //     Note: It can have performance impact as there are extra copies and element wise copies are involved  
    int64_t* local_dIpiv;
    auto no_of_elements = max(1, min(m,n));
    // Allocating it on host with device access to avoid extra copy needed while accessing it from host
    hipHostMalloc(&local_dIpiv, sizeof(int64_t)* no_of_elements);

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cgetrf(queue, m, n, (float _Complex*)A, lda, local_dIpiv, (float _Complex*)work, lwork);

    int* local_hIpiv = (int*)malloc(sizeof(int)* no_of_elements);
    for(auto i=0; i< no_of_elements; ++i){
        local_hIpiv[i] = (int)local_dIpiv[i];
    }
    hipMemcpy(devIpiv, local_hIpiv, sizeof(int)* min(m,n), hipMemcpyHostToDevice);

    // release the memory allocated in the WA
    hipFree(local_dIpiv);
    free(local_hIpiv);
    
    if (allocate)
        hipFree(work);
    
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   hipDoubleComplex* work,
                                                   int               lwork,
                                                   int*              devIpiv,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || devIpiv == nullptr || devInfo == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork == 0) {
        lwork = 0;
        hipsolverZgetrf_bufferSize(handle, m, n, A, lda, &lwork);
        hipHostMalloc(&work, lwork);
        allocate = true;
    }
    // WA: MKL does not use devInfo hence resetting it to zero
    hipMemset(devInfo, 0, sizeof(int));
    
    // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
    //     hence need special handling here. Force type cast was causing crash as 
    //     MKL's requirement is more.
    //     Note: It can have performance impact as there are extra copies and element wise copies are involved  
    int64_t* local_dIpiv;
    auto no_of_elements = max(1, min(m,n));
    // Allocating it on host with device access to avoid extra copy needed while accessing it from host
    hipHostMalloc(&local_dIpiv, sizeof(int64_t)* no_of_elements);

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zgetrf(queue, m, n, (double _Complex*)A, lda, local_dIpiv, (double _Complex*)work, lwork);

    int* local_hIpiv = (int*)malloc(sizeof(int)* no_of_elements);
    for(auto i=0; i< no_of_elements; ++i){
        local_hIpiv[i] = (int)local_dIpiv[i];
    }
    hipMemcpy(devIpiv, local_hIpiv, sizeof(int)* min(m,n), hipMemcpyHostToDevice);

    // release the memory allocated in the WA
    hipFree(local_dIpiv);
    free(local_hIpiv);
    
    if (allocate)
        hipFree(work);
    
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// getrs
hipsolverStatus_t hipsolverSgetrs_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverOperation_t trans,
                                                              int                  n,
                                                              int                  nrhs,
                                                              float*               A,
                                                              int                  lda,
                                                              int*                 devIpiv,
                                                              float*               B,
                                                              int                  ldb,
                                                              int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Sgetrs_ScPadSz(queue, convert(trans), n, nrhs, lda, ldb);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrs_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverOperation_t trans,
                                                              int                  n,
                                                              int                  nrhs,
                                                              double*              A,
                                                              int                  lda,
                                                              int*                 devIpiv,
                                                              double*              B,
                                                              int                  ldb,
                                                              int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Dgetrs_ScPadSz(queue, convert(trans), n, nrhs, lda, ldb);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrs_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverOperation_t trans,
                                                              int                  n,
                                                              int                  nrhs,
                                                              hipFloatComplex*     A,
                                                              int                  lda,
                                                              int*                 devIpiv,
                                                              hipFloatComplex*     B,
                                                              int                  ldb,
                                                              int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Cgetrs_ScPadSz(queue, convert(trans), n, nrhs, lda, ldb);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrs_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverOperation_t trans,
                                                              int                  n,
                                                              int                  nrhs,
                                                              hipDoubleComplex*    A,
                                                              int                  lda,
                                                              int*                 devIpiv,
                                                              hipDoubleComplex*    B,
                                                              int                  ldb,
                                                              int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

    *lwork = (int)onemkl_Zgetrs_ScPadSz(queue, convert(trans), n, nrhs, lda, ldb);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSgetrs(hipsolverHandle_t    handle,
                                hipsolverOperation_t trans,
                                int                  n,
                                int                  nrhs,
                                float*               A,
                                int                  lda,
                                int*                 devIpiv,
                                float*               B,
                                int                  ldb,
                                float*               work,
                                int                  lwork,
                                int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || B == nullptr || devIpiv == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork ==0) {
        lwork = 0;
        hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork);
        hipHostMalloc(&work, lwork);
        allocate = true;
    }
    // WA: MKL does not use devinfo hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));
    // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
    //     hence need special handling here. Force type cast was causing result mismatch
    //     Note: It can have performance impact as there are extra copies and 
    //           element wise copies are involved between Host <-> device memory
    auto no_of_elements = max(1, n);
    int* local_hIpiv = (int*)malloc(no_of_elements*sizeof(int));
    hipMemcpy(local_hIpiv, devIpiv, sizeof(int)*no_of_elements, hipMemcpyDeviceToHost);

    int64_t* dIpiv; hipHostMalloc(&dIpiv, sizeof(int64_t)*no_of_elements);

    for(auto i=0; i<no_of_elements; ++i) {
        dIpiv[i] = local_hIpiv[i];
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sgetrs(queue, convert(trans), n, nrhs, A, lda, (int64_t*)dIpiv, B, ldb, work, lwork);

    hipFree(dIpiv);
    free(local_hIpiv);

    if (allocate)
        hipFree(work);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrs(hipsolverHandle_t    handle,
                                                   hipsolverOperation_t trans,
                                                   int                  n,
                                                   int                  nrhs,
                                                   double*              A,
                                                   int                  lda,
                                                   int*                 devIpiv,
                                                   double*              B,
                                                   int                  ldb,
                                                   double*              work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || B == nullptr || devIpiv == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork ==0) {
        lwork = 0;
        hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork);
        hipHostMalloc(&work, lwork);
        allocate = true;
    }
    // WA: MKL does not use devinfo hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));
    // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
    //     hence need special handling here. Force type cast was causing result mismatch
    //     Note: It can have performance impact as there are extra copies and 
    //           element wise copies are involved between Host <-> device memory
    auto no_of_elements = max(1, n);
    int* local_hIpiv = (int*)malloc(no_of_elements*sizeof(int));
    hipMemcpy(local_hIpiv, devIpiv, sizeof(int)*no_of_elements, hipMemcpyDeviceToHost);

    int64_t* dIpiv; hipHostMalloc(&dIpiv, sizeof(int64_t)*no_of_elements);

    for(auto i=0; i<no_of_elements; ++i) {
        dIpiv[i] = local_hIpiv[i];
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dgetrs(queue, convert(trans), n, nrhs, A, lda, dIpiv, B, ldb, work, lwork);

    hipFree(dIpiv);
    free(local_hIpiv);

    if (allocate)
        hipFree(work);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrs(hipsolverHandle_t    handle,
                                                   hipsolverOperation_t trans,
                                                   int                  n,
                                                   int                  nrhs,
                                                   hipFloatComplex*     A,
                                                   int                  lda,
                                                   int*                 devIpiv,
                                                   hipFloatComplex*     B,
                                                   int                  ldb,
                                                   hipFloatComplex*     work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || B == nullptr || devIpiv == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork ==0) {
        lwork = 0;
        hipsolverCgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork);
        hipHostMalloc(&work, lwork);
        allocate = true;
    }
    // WA: MKL does not use devinfo hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));
    // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
    //     hence need special handling here. Force type cast was causing result mismatch
    //     Note: It can have performance impact as there are extra copies and 
    //           element wise copies are involved between Host <-> device memory
    auto no_of_elements = max(1, n);
    int* local_hIpiv = (int*)malloc(no_of_elements*sizeof(int));
    hipMemcpy(local_hIpiv, devIpiv, sizeof(int)*no_of_elements, hipMemcpyDeviceToHost);

    int64_t* dIpiv; hipHostMalloc(&dIpiv, sizeof(int64_t)*no_of_elements);

    for(auto i=0; i<no_of_elements; ++i) {
        dIpiv[i] = local_hIpiv[i];
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cgetrs(queue, convert(trans), n, nrhs, (float _Complex*)A, lda, dIpiv, (float _Complex*)B, ldb, (float _Complex*)work, lwork);

    hipFree(dIpiv);
    free(local_hIpiv);

    if (allocate)
        hipFree(work);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrs(hipsolverHandle_t    handle,
                                                   hipsolverOperation_t trans,
                                                   int                  n,
                                                   int                  nrhs,
                                                   hipDoubleComplex*    A,
                                                   int                  lda,
                                                   int*                 devIpiv,
                                                   hipDoubleComplex*    B,
                                                   int                  ldb,
                                                   hipDoubleComplex*    work,
                                                   int                  lwork,
                                                   int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(trans))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (A == nullptr || B == nullptr || devIpiv == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if (work == nullptr || lwork ==0) {
        lwork = 0;
        hipsolverZgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork);
        hipHostMalloc(&work, lwork);
        allocate = true;
    }
    // WA: MKL does not use devinfo hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));
    // WA: data type of devIpiv is different in MKL vs HIP and CUDA solver library
    //     hence need special handling here. Force type cast was causing result mismatch
    //     Note: It can have performance impact as there are extra copies and 
    //           element wise copies are involved between Host <-> device memory
    auto no_of_elements = max(1, n);
    int* local_hIpiv = (int*)malloc(no_of_elements*sizeof(int));
    hipMemcpy(local_hIpiv, devIpiv, sizeof(int)*no_of_elements, hipMemcpyDeviceToHost);

    int64_t* dIpiv; hipHostMalloc(&dIpiv, sizeof(int64_t)*no_of_elements);

    for(auto i=0; i<no_of_elements; ++i) {
        dIpiv[i] = local_hIpiv[i];
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zgetrs(queue, convert(trans), n, nrhs, (double _Complex*)A, lda, dIpiv, (double _Complex*)B, ldb, (double _Complex*)work, lwork);

    hipFree(dIpiv);
    free(local_hIpiv);

    if (allocate)
        hipFree(work);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// potrf
hipsolverStatus_t hipsolverSpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Spotrf_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dpotrf_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrf_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Cpotrf_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrf_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Zpotrf_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Spotrf(queue, convert(uplo), n, A, lda, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dpotrf(queue, convert(uplo), n, A, lda, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cpotrf(queue, convert(uplo), n, (float _Complex*)A, lda, (float _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipDoubleComplex*   A,
                                                   int                 lda,
                                                   hipDoubleComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zpotrf(queue, convert(uplo), n, (double _Complex*)A, lda, (double _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// potrf_batched
hipsolverStatus_t hipsolverSpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A[],
                                                                     int                 lda,
                                                                     int*                lwork,
                                                                     int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A[],
                                                                     int                 lda,
                                                                     int*                lwork,
                                                                     int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipFloatComplex*    A[],
                                                                     int                 lda,
                                                                     int*                lwork,
                                                                     int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipDoubleComplex*   A[],
                                                                     int                 lda,
                                                                     int*                lwork,
                                                                     int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrfBatched(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A[],
                                                          int                 lda,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo,
                                                          int                 batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrfBatched(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A[],
                                                          int                 lda,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo,
                                                          int                 batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrfBatched(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A[],
                                                          int                 lda,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo,
                                                          int                 batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrfBatched(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A[],
                                                          int                 lda,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo,
                                                          int                 batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// potri
hipsolverStatus_t hipsolverSpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Spotri_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dpotri_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotri_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Cpotri_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotri_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Zpotri_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotri(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Spotri(queue, convert(uplo), n, A, lda, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotri(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dpotri(queue, convert(uplo), n, A, lda, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotri(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverCpotri_bufferSize(handle, uplo, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cpotri(queue, convert(uplo), n, (float _Complex*)A, lda, (float _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotri(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipDoubleComplex*   A,
                                                   int                 lda,
                                                   hipDoubleComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverZpotri_bufferSize(handle, uplo, n, A, lda, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zpotri(queue, convert(uplo), n, (double _Complex*)A, lda, (double _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// potrs
hipsolverStatus_t hipsolverSpotrs_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            int                 nrhs,
                                            float*              A,
                                            int                 lda,
                                            float*              B,
                                            int                 ldb,
                                            int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Spotrs_ScPadSz(queue, convert(uplo), n, nrhs, lda, ldb);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrs_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              int                 nrhs,
                                                              double*             A,
                                                              int                 lda,
                                                              double*             B,
                                                              int                 ldb,
                                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dpotrs_ScPadSz(queue, convert(uplo), n, nrhs, lda, ldb);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrs_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              int                 nrhs,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              hipFloatComplex*    B,
                                                              int                 ldb,
                                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Cpotrs_ScPadSz(queue, convert(uplo), n, nrhs, lda, ldb);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrs_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              int                 nrhs,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              hipDoubleComplex*   B,
                                                              int                 ldb,
                                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Cpotrs_ScPadSz(queue, convert(uplo), n, nrhs, lda, ldb);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrs(hipsolverHandle_t   handle,
                                hipsolverFillMode_t uplo,
                                int                 n,
                                int                 nrhs,
                                float*              A,
                                int                 lda,
                                float*              B,
                                int                 ldb,
                                float*              work,
                                int                 lwork,
                                int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || B == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Spotrs(queue, convert(uplo), n, nrhs, A, lda, B, ldb, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrs(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   int                 nrhs,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             B,
                                                   int                 ldb,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || B == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dpotrs(queue, convert(uplo), n, nrhs, A, lda, B, ldb, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrs(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   int                 nrhs,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   hipFloatComplex*    B,
                                                   int                 ldb,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || B == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverCpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cpotrs(queue, convert(uplo), n, nrhs, (float _Complex*)A, lda, (float _Complex*)B, ldb, (float _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   B,
                                  int                 ldb,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || B == nullptr || devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverZpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zpotrs(queue, convert(uplo), n, nrhs, (double _Complex*)A, lda, (double _Complex*)B, ldb, (double _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// potrs_batched
hipsolverStatus_t hipsolverSpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    float*              A[],
                                                    int                 lda,
                                                    float*              B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    double*             A[],
                                                    int                 lda,
                                                    double*             B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipFloatComplex*    A[],
                                                    int                 lda,
                                                    hipFloatComplex*    B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipDoubleComplex*   A[],
                                                    int                 lda,
                                                    hipDoubleComplex*   B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrsBatched(hipsolverHandle_t   handle,
                                        hipsolverFillMode_t uplo,
                                        int                 n,
                                        int                 nrhs,
                                        float*              A[],
                                        int                 lda,
                                        float*              B[],
                                        int                 ldb,
                                        float*              work,
                                        int                 lwork,
                                        int*                devInfo,
                                        int                 batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrsBatched(hipsolverHandle_t   handle,
                                        hipsolverFillMode_t uplo,
                                        int                 n,
                                        int                 nrhs,
                                        double*             A[],
                                        int                 lda,
                                        double*             B[],
                                        int                 ldb,
                                        double*             work,
                                        int                 lwork,
                                        int*                devInfo,
                                        int                 batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrsBatched(hipsolverHandle_t   handle,
                                        hipsolverFillMode_t uplo,
                                        int                 n,
                                        int                 nrhs,
                                        hipFloatComplex*    A[],
                                        int                 lda,
                                        hipFloatComplex*    B[],
                                        int                 ldb,
                                        hipFloatComplex*    work,
                                        int                 lwork,
                                        int*                devInfo,
                                        int                 batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrsBatched(hipsolverHandle_t   handle,
                                          hipsolverFillMode_t uplo,
                                          int                 n,
                                          int                 nrhs,
                                          hipDoubleComplex*   A[],
                                          int                 lda,
                                          hipDoubleComplex*   B[],
                                          int                 ldb,
                                          hipDoubleComplex*   work,
                                          int                 lwork,
                                          int*                devInfo,
                                          int                 batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// syevd/heevd
hipsolverStatus_t hipsolverSsyevd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              float*              A,
                                              int                 lda,
                                              float*              D,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;
    if (lwork == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Ssyevd_ScPadSz(queue, convert(jobz), convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              double*             A,
                                              int                 lda,
                                              double*             D,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || D == nullptr || lwork == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dsyevd_ScPadSz(queue, convert(jobz), convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipFloatComplex*    A,
                                              int                 lda,
                                              float*              D,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || D == nullptr || lwork == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Cheevd_ScPadSz(queue, convert(jobz), convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevd_bufferSize(hipsolverHandle_t   handle,
                                            hipsolverEigMode_t  jobz,
                                            hipsolverFillMode_t uplo,
                                            int                 n,
                                            hipDoubleComplex*   A,
                                            int                 lda,
                                            double*             D,
                                            int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (A == nullptr || D == nullptr || lwork == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Zheevd_ScPadSz(queue, convert(jobz), convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSsyevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              D,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || D == nullptr ){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if(work == nullptr || lwork == 0) {
        hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Ssyevd(queue, convert(jobz), convert(uplo), n, A, lda, D, work, lwork);

    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             D,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || D == nullptr ){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if(work == nullptr || lwork == 0) {
        hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dsyevd(queue, convert(jobz), convert(uplo), n, A, lda, D, work, lwork);
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  float*              D,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || D == nullptr ){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if(work == nullptr || lwork == 0) {
        hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cheevd(queue, convert(jobz), convert(uplo), n, (float _Complex*)A, lda, D, (float _Complex*)work, lwork);
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  double*             D,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || D == nullptr ){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    bool allocate = false;
    if(work == nullptr || lwork == 0) {
        hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zheevd(queue, convert(jobz), convert(uplo), n, (double _Complex*)A, lda, D, (double _Complex*)work, lwork);
    if (allocate)
        hipFree(work);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// syevj/heevj
hipsolverStatus_t hipsolverSsyevj_bufferSize(hipsolverHandle_t    handle,
                                              hipsolverEigMode_t   jobz,
                                              hipsolverFillMode_t  uplo,
                                              int                  n,
                                              float*               A,
                                              int                  lda,
                                              float*               W,
                                              int*                 lwork,
                                              hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevj_bufferSize(hipsolverHandle_t    handle,
                                              hipsolverEigMode_t   jobz,
                                              hipsolverFillMode_t  uplo,
                                              int                  n,
                                              double*              A,
                                              int                  lda,
                                              double*              W,
                                              int*                 lwork,
                                              hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevj_bufferSize(hipsolverHandle_t    handle,
                                            hipsolverEigMode_t   jobz,
                                            hipsolverFillMode_t  uplo,
                                            int                  n,
                                            hipFloatComplex*     A,
                                            int                  lda,
                                            float*               W,
                                            int*                 lwork,
                                            hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevj_bufferSize(hipsolverHandle_t    handle,
                                              hipsolverEigMode_t   jobz,
                                              hipsolverFillMode_t  uplo,
                                              int                  n,
                                              hipDoubleComplex*    A,
                                              int                  lda,
                                              double*              W,
                                              int*                 lwork,
                                              hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSsyevj(hipsolverHandle_t    handle,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  float*               A,
                                  int                  lda,
                                  float*               W,
                                  float*               work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevj(hipsolverHandle_t    handle,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  double*              A,
                                  int                  lda,
                                  double*              W,
                                  double*              work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevj(hipsolverHandle_t    handle,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  hipFloatComplex*     A,
                                  int                  lda,
                                  float*               W,
                                  hipFloatComplex*     work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevj(hipsolverHandle_t    handle,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  hipDoubleComplex*    A,
                                  int                  lda,
                                  double*              W,
                                  hipDoubleComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// syevj_batched/heevj_batched
hipsolverStatus_t hipsolverSsyevjBatched_bufferSize(hipsolverHandle_t    handle,
                                                    hipsolverEigMode_t   jobz,
                                                    hipsolverFillMode_t  uplo,
                                                    int                  n,
                                                    float*               A,
                                                    int                  lda,
                                                    float*               W,
                                                    int*                 lwork,
                                                    hipsolverSyevjInfo_t params,
                                                    int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevjBatched_bufferSize(hipsolverHandle_t    handle,
                                                    hipsolverEigMode_t   jobz,
                                                    hipsolverFillMode_t  uplo,
                                                    int                  n,
                                                    double*              A,
                                                    int                  lda,
                                                    double*              W,
                                                    int*                 lwork,
                                                    hipsolverSyevjInfo_t params,
                                                    int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevjBatched_bufferSize(hipsolverHandle_t    handle,
                                                    hipsolverEigMode_t   jobz,
                                                    hipsolverFillMode_t  uplo,
                                                    int                  n,
                                                    hipFloatComplex*     A,
                                                    int                  lda,
                                                    float*               W,
                                                    int*                 lwork,
                                                    hipsolverSyevjInfo_t params,
                                                    int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevjBatched_bufferSize(hipsolverHandle_t    handle,
                                                    hipsolverEigMode_t   jobz,
                                                    hipsolverFillMode_t  uplo,
                                                    int                  n,
                                                    hipDoubleComplex*    A,
                                                    int                  lda,
                                                    double*              W,
                                                    int*                 lwork,
                                                    hipsolverSyevjInfo_t params,
                                                    int batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSsyevjBatched(hipsolverHandle_t    handle,
                                          hipsolverEigMode_t   jobz,
                                          hipsolverFillMode_t  uplo,
                                          int                  n,
                                          float*               A,
                                          int                  lda,
                                          float*               W,
                                          float*               work,
                                          int                  lwork,
                                          int*                 devInfo,
                                          hipsolverSyevjInfo_t params,
                                          int                  batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevjBatched(hipsolverHandle_t    handle,
                                          hipsolverEigMode_t   jobz,
                                          hipsolverFillMode_t  uplo,
                                          int                  n,
                                          double*              A,
                                          int                  lda,
                                          double*              W,
                                          double*              work,
                                          int                  lwork,
                                          int*                 devInfo,
                                          hipsolverSyevjInfo_t params,
                                          int                  batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevjBatched(hipsolverHandle_t    handle,
                                          hipsolverEigMode_t   jobz,
                                          hipsolverFillMode_t  uplo,
                                          int                  n,
                                          hipFloatComplex*     A,
                                          int                  lda,
                                          float*               W,
                                          hipFloatComplex*     work,
                                          int                  lwork,
                                          int*                 devInfo,
                                          hipsolverSyevjInfo_t params,
                                          int                  batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevjBatched(hipsolverHandle_t    handle,
                                          hipsolverEigMode_t   jobz,
                                          hipsolverFillMode_t  uplo,
                                          int                  n,
                                          hipDoubleComplex*    A,
                                          int                  lda,
                                          double*              W,
                                          hipDoubleComplex*    work,
                                          int                  lwork,
                                          int*                 devInfo,
                                          hipsolverSyevjInfo_t params,
                                          int                  batch_count)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// sygvd/hegvd
hipsolverStatus_t hipsolverSsygvd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigType_t  itype,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              float*              A,
                                              int                 lda,
                                              float*              B,
                                              int                 ldb,
                                              float*              W,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo) || !isvalid(jobz) || !isvalid(itype))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Ssygvd_ScPadSz(queue, convert(itype), convert(jobz), convert(uplo), n, lda, ldb);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsygvd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigType_t  itype,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              double*             A,
                                              int                 lda,
                                              double*             B,
                                              int                 ldb,
                                              double*             W,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo) || !isvalid(jobz) || !isvalid(itype))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dsygvd_ScPadSz(queue, convert(itype), convert(jobz), convert(uplo), n, lda, ldb);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverChegvd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigType_t  itype,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipFloatComplex*    A,
                                              int                 lda,
                                              hipFloatComplex*    B,
                                              int                 ldb,
                                              float*              W,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo) || !isvalid(jobz) || !isvalid(itype))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Chegvd_ScPadSz(queue, convert(itype), convert(jobz), convert(uplo), n, lda, ldb);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZhegvd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverEigType_t  itype,
                                              hipsolverEigMode_t  jobz,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipDoubleComplex*   A,
                                              int                 lda,
                                              hipDoubleComplex*   B,
                                              int                 ldb,
                                              double*             W,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo) || !isvalid(jobz) || !isvalid(itype))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Zhegvd_ScPadSz(queue, convert(itype), convert(jobz), convert(uplo), n, lda, ldb);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSsygvd(hipsolverHandle_t   handle,
                                  hipsolverEigType_t  itype,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              B,
                                  int                 ldb,
                                  float*              W,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(itype) || !isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || B == nullptr || W ==nullptr ||devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Ssygvd(queue, convert(itype), convert(jobz), convert(uplo), n, A, lda, B, ldb, W, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsygvd(hipsolverHandle_t   handle,
                                  hipsolverEigType_t  itype,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             B,
                                  int                 ldb,
                                  double*             W,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(itype) || !isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || B == nullptr || W ==nullptr ||devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dsygvd(queue, convert(itype), convert(jobz), convert(uplo), n, A, lda, B, ldb, W, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverChegvd(hipsolverHandle_t   handle,
                                  hipsolverEigType_t  itype,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    B,
                                  int                 ldb,
                                  float*              W,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(itype) || !isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || B == nullptr || W ==nullptr ||devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Chegvd(queue, convert(itype), convert(jobz), convert(uplo), n, (float _Complex*)A, lda, (float _Complex*)B, ldb, W, (float _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZhegvd(hipsolverHandle_t   handle,
                                  hipsolverEigType_t  itype,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   B,
                                  int                 ldb,
                                  double*             W,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(itype) || !isvalid(jobz) || !isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || B == nullptr || W ==nullptr ||devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zhegvd(queue, convert(itype), convert(jobz), convert(uplo), n, (double _Complex*)A, lda, (double _Complex*)B, ldb, W, (double _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// sygvj/hegvj
hipsolverStatus_t hipsolverSsygvj_bufferSize(hipsolverHandle_t    handle,
                                              hipsolverEigType_t   itype,
                                              hipsolverEigMode_t   jobz,
                                              hipsolverFillMode_t  uplo,
                                              int                  n,
                                              float*               A,
                                              int                  lda,
                                              float*               B,
                                              int                  ldb,
                                              float*               W,
                                              int*                 lwork,
                                              hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsygvj_bufferSize(hipsolverHandle_t    handle,
                                              hipsolverEigType_t   itype,
                                              hipsolverEigMode_t   jobz,
                                              hipsolverFillMode_t  uplo,
                                              int                  n,
                                              double*              A,
                                              int                  lda,
                                              double*              B,
                                              int                  ldb,
                                              double*              W,
                                              int*                 lwork,
                                              hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverChegvj_bufferSize(hipsolverHandle_t    handle,
                                              hipsolverEigType_t   itype,
                                              hipsolverEigMode_t   jobz,
                                              hipsolverFillMode_t  uplo,
                                              int                  n,
                                              hipFloatComplex*     A,
                                              int                  lda,
                                              hipFloatComplex*     B,
                                              int                  ldb,
                                              float*               W,
                                              int*                 lwork,
                                              hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZhegvj_bufferSize(hipsolverHandle_t    handle,
                                              hipsolverEigType_t   itype,
                                              hipsolverEigMode_t   jobz,
                                              hipsolverFillMode_t  uplo,
                                              int                  n,
                                              hipDoubleComplex*    A,
                                              int                  lda,
                                              hipDoubleComplex*    B,
                                              int                  ldb,
                                              double*              W,
                                              int*                 lwork,
                                              hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSsygvj(hipsolverHandle_t    handle,
                                  hipsolverEigType_t   itype,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  float*               A,
                                  int                  lda,
                                  float*               B,
                                  int                  ldb,
                                  float*               W,
                                  float*               work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsygvj(hipsolverHandle_t    handle,
                                  hipsolverEigType_t   itype,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  double*              A,
                                  int                  lda,
                                  double*              B,
                                  int                  ldb,
                                  double*              W,
                                  double*              work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverChegvj(hipsolverHandle_t    handle,
                                  hipsolverEigType_t   itype,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  hipFloatComplex*     A,
                                  int                  lda,
                                  hipFloatComplex*     B,
                                  int                  ldb,
                                  float*               W,
                                  hipFloatComplex*     work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZhegvj(hipsolverHandle_t    handle,
                                  hipsolverEigType_t   itype,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  hipDoubleComplex*    A,
                                  int                  lda,
                                  hipDoubleComplex*    B,
                                  int                  ldb,
                                  double*              W,
                                  hipDoubleComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t params)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

// sytrd/hetrd
hipsolverStatus_t hipsolverSsytrd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              float*              A,
                                              int                 lda,
                                              float*              D,
                                              float*              E,
                                              float*              tau,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Ssytrd_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsytrd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              double*             A,
                                              int                 lda,
                                              double*             D,
                                              double*             E,
                                              double*             tau,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Dsytrd_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverChetrd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipFloatComplex*    A,
                                              int                 lda,
                                              float*              D,
                                              float*              E,
                                              hipFloatComplex*    tau,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Chetrd_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZhetrd_bufferSize(hipsolverHandle_t   handle,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipDoubleComplex*   A,
                                              int                 lda,
                                              double*             D,
                                              double*             E,
                                              hipDoubleComplex*   tau,
                                              int*                lwork)
try
{
    if(!handle) return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if (!isvalid(uplo))
    {
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    if (lwork == nullptr ) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    auto queue = sycl_get_queue((syclHandle_t)handle);
    auto size = onemkl_Zhetrd_ScPadSz(queue, convert(uplo), n, lda);
    *lwork = (int)size;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSsytrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              D,
                                  float*              E,
                                  float*              tau,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || D == nullptr || E==nullptr||tau==nullptr||devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Ssytrd(queue, convert(uplo), n, A, lda, D, E, tau, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsytrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             D,
                                  double*             E,
                                  double*             tau,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || D == nullptr || E==nullptr||tau==nullptr||devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dsytrd(queue, convert(uplo), n, A, lda, D, E, tau, work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverChetrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  float*              D,
                                  float*              E,
                                  hipFloatComplex*    tau,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || D == nullptr || E==nullptr||tau==nullptr||devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Chetrd(queue, convert(uplo), n, (float _Complex*)A, lda, D, E, (float _Complex*)tau, (float _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZhetrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  double*             D,
                                  double*             E,
                                  hipDoubleComplex*   tau,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    if (!isvalid(uplo))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    if(A == nullptr || D == nullptr || E==nullptr||tau==nullptr||devInfo==nullptr){
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    bool allocate = false;
    if (work == nullptr || lwork == 0){
        auto status = hipsolverZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return status;
        }
        hipHostMalloc(&work, lwork, 0);
        allocate = true;
    }

    // WA: MKL does not use info hence setting it to zero
    hipMemset(devInfo, 0, sizeof(int));

    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zhetrd(queue, convert(uplo), n, (double _Complex*)A, lda, D, E, (double _Complex*)tau, (double _Complex*)work, lwork);

    if (allocate) {
        hipFree(work);
    }
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
	return exception2hip_status();
}

// sytrf
hipsolverStatus_t
    hipsolverSsytrf_bufferSize(hipsolverHandle_t handle, int n, float* A, int lda, int* lwork)
try
{

}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t
    hipsolverDsytrf_bufferSize(hipsolverHandle_t handle, int n, double* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverSsytrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  int*                ipiv,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDsytrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  int*                ipiv,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCsytrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  int*                ipiv,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZsytrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  int*                ipiv,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}
