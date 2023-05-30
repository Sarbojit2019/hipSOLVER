#include <hip/hip_interop.h>
#include "hipsolver.h"
#include "exceptions.hpp"

#include <unordered_set>
#include "deps/sycl_solver.h"

std::unordered_set<hipsolverHandle_t*> solverHandleTbl;

inline onemklGen convertToGen(hipsolverSideMode_t s) {
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

inline onemklJob convert(hipsolverEigMode_t job) {
  switch(job) {
    case HIPSOLVER_EIG_MODE_NOVECTOR: return ONEMKL_JOB_NOVEC;
    case HIPSOLVER_EIG_MODE_VECTOR: return ONEMKL_JOB_VEC;
  }
}

inline onemklUplo convert(hipsolverFillMode_t val) {
    switch(val) {
        case HIPSOLVER_FILL_MODE_UPPER:
            return ONEMKL_UPLO_UPPER;
        case HIPSOLVER_FILL_MODE_LOWER:
            return ONEMKL_UPLO_LOWER;
    }
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);

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
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sorgbr(queue, convertToGen(side), m, n, k, A, lda, tau, work, lwork);
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
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sorgqr(queue, m, n, k, A, lda, tau, work, lwork);
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
    if (A == nullptr || tau == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = 0;
    hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, &lwork);
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sorgtr(queue, convert(uplo), n, A, lda, tau, work, lwork);
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
    auto size = onemkl_Sgebrd_ScPadSz(queue, m, n, *lwork);
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
    auto size = onemkl_Sgebrd_ScPadSz(queue, m, n, *lwork);
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
    auto size = onemkl_Sgebrd_ScPadSz(queue, m, n, *lwork);
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
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = lda;
    auto status = hipsolverSgebrd_bufferSize(handle, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Sgebrd(queue, m, n, A, lda, D, E, tauq, taup, work, lwork);
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
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = lda;
    auto status = hipsolverDgebrd_bufferSize(handle, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Dgebrd(queue, m, n, A, lda, D, E, tauq, taup, work, lwork);
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
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = lda;
    auto status = hipsolverCgebrd_bufferSize(handle, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Cgebrd(queue, m, n, (float _Complex*)A, lda, D, E,
                 (float _Complex*)tauq, (float _Complex*)taup, (float _Complex*)work, lwork);
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
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr || work == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    lwork = lda;
    auto status = hipsolverZgebrd_bufferSize(handle, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    onemkl_Zgebrd(queue, m, n, (double _Complex*)A, lda, D, E,
                 (double _Complex*)tauq, (double _Complex*)taup, (double _Complex*)work, lwork);
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
    if (A == nullptr || S == nullptr || U == nullptr || V == nullptr || work == nullptr || rwork == nullptr) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    auto queue = sycl_get_queue((syclHandle_t)handle);
    lwork = (int)onemkl_Sgesvd_ScPadSz(queue, jobu, jobv, m, n, lda, ldu, ldv);
    onemkl_Sgesvd(queue, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork);
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
	return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
try
{
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
    if (A == nullptr || D == nullptr || lwork == nullptr) {
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
    if(A == nullptr || D == nullptr || work == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
    onemkl_Ssyevd(queue, convert(jobz), convert(uplo), n, A, lda, D, work, lwork);
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
    if(A == nullptr || D == nullptr || work == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
    onemkl_Dsyevd(queue, convert(jobz), convert(uplo), n, A, lda, D, work, lwork);
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
    if(A == nullptr || D == nullptr || work == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
    onemkl_Cheevd(queue, convert(jobz), convert(uplo), n, (float _Complex*)A, lda, D, (float _Complex*)work, lwork);
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
    if(A == nullptr || D == nullptr || work == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    auto queue = sycl_get_queue((syclHandle_t)handle);
    hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork);
    onemkl_Zheevd(queue, convert(jobz), convert(uplo), n, (double _Complex*)A, lda, D, (double _Complex*)work, lwork);
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
	return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
