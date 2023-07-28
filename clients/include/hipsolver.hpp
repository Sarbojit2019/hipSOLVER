/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"
#ifdef _WIN32
#include "hipsolver_no_fortran.hpp"
#else
#include "hipsolver_fortran.hpp"
#endif

// Most functions within this file exist to provide a consistent interface for our templated tests.
// Function overloading is used to select between the float, double, rocblas_float_complex
// and rocblas_double_complex variants, and to distinguish the batched case (T**) from the normal
// and strided_batched cases (T*).
//
// The normal and strided_batched cases are distinguished from each other by passing a boolean
// parameter, STRIDED. Variants such as the blocked and unblocked versions of algorithms, may be
// provided in similar ways.

typedef enum
{
    API_NORMAL,
    API_FORTRAN,
    API_COMPAT
} testAPI_t;

typedef enum
{
    C_NORMAL,
    C_NORMAL_ALT,
    C_STRIDED,
    C_STRIDED_ALT,
    FORTRAN_NORMAL,
    FORTRAN_NORMAL_ALT,
    FORTRAN_STRIDED,
    FORTRAN_STRIDED_ALT,
    COMPAT_NORMAL,
    COMPAT_NORMAL_ALT,
    COMPAT_STRIDED,
    COMPAT_STRIDED_ALT,
    INVALID_API_SPEC
} testMarshal_t;

inline testMarshal_t api2marshal(testAPI_t API, bool ALT)
{
    switch(API)
    {
    case API_NORMAL:
        if(!ALT)
            return C_NORMAL;
        else
            return C_NORMAL_ALT;
    case API_FORTRAN:
        if(!ALT)
            return FORTRAN_NORMAL;
        else
            return FORTRAN_NORMAL_ALT;
    case API_COMPAT:
        if(!ALT)
            return COMPAT_NORMAL;
        else
            return COMPAT_NORMAL_ALT;
    default:
        return INVALID_API_SPEC;
    }
}

/******************** ORGBR/UNGBR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          int*                lwork)
{

        return hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          int*                lwork)
{

        return hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{

        return hipsolverCungbr_bufferSize(
            handle, side, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{

        return hipsolverZungbr_bufferSize(
            handle, side, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               float*              A,
                                               int                 lda,
                                               float*              tau,
                                               float*              work,
                                               int                 lwork,
                                               int*                info)
{

        return hipsolverSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               double*             A,
                                               int                 lda,
                                               double*             tau,
                                               double*             work,
                                               int                 lwork,
                                               int*                info)
{

        return hipsolverDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               hipsolverComplex*   tau,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info)
{

        return hipsolverCungbr(handle,
                               side,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverSideMode_t     side,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{

        return hipsolverZungbr(handle,
                               side,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}

/******************** ORGQR/UNGQR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(bool              FORTRAN,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          float*            A,
                                                          int               lda,
                                                          float*            tau,
                                                          int*              lwork)
{

        return hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(bool              FORTRAN,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          double*           A,
                                                          int               lda,
                                                          double*           tau,
                                                          int*              lwork)
{

        return hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(bool              FORTRAN,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          hipsolverComplex* A,
                                                          int               lda,
                                                          hipsolverComplex* tau,
                                                          int*              lwork)
{

        return hipsolverCungqr_bufferSize(
            handle, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{

        return hipsolverZungqr_bufferSize(
            handle, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(bool              FORTRAN,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               float*            A,
                                               int               lda,
                                               float*            tau,
                                               float*            work,
                                               int               lwork,
                                               int*              info)
{

        return hipsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(bool              FORTRAN,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               double*           A,
                                               int               lda,
                                               double*           tau,
                                               double*           work,
                                               int               lwork,
                                               int*              info)
{

        return hipsolverDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(bool              FORTRAN,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               hipsolverComplex* A,
                                               int               lda,
                                               hipsolverComplex* tau,
                                               hipsolverComplex* work,
                                               int               lwork,
                                               int*              info)
{

        return hipsolverCungqr(handle,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{

        return hipsolverZungqr(handle,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** ORGTR/UNGTR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          int*                lwork)
{

        return hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          int*                lwork)
{

        return hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{

        return hipsolverCungtr_bufferSize(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{

        return hipsolverZungtr_bufferSize(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);

}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               float*              tau,
                                               float*              work,
                                               int                 lwork,
                                               int*                info)
{

        return hipsolverSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               double*             tau,
                                               double*             work,
                                               int                 lwork,
                                               int*                info)
{

        return hipsolverDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               hipsolverComplex*   tau,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info)
{

        return hipsolverCungtr(handle,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{

        return hipsolverZungtr(handle,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** ORMQR/UNMQR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
{

        return hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);

}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
{

        return hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);

}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          int                  k,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          hipsolverComplex*    tau,
                                                          hipsolverComplex*    C,
                                                          int                  ldc,
                                                          int*                 lwork)
{

        return hipsolverCunmqr_bufferSize(handle,
                                          side,
                                          trans,
                                          m,
                                          n,
                                          k,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)tau,
                                          (hipFloatComplex*)C,
                                          ldc,
                                          lwork);

}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          hipsolverOperation_t    trans,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* C,
                                                          int                     ldc,
                                                          int*                    lwork)
{

        return hipsolverZunmqr_bufferSize(handle,
                                          side,
                                          trans,
                                          m,
                                          n,
                                          k,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)tau,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          lwork);

}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{

        return hipsolverSormqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{

        return hipsolverDormqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
                                               hipsolverSideMode_t  side,
                                               hipsolverOperation_t trans,
                                               int                  m,
                                               int                  n,
                                               int                  k,
                                               hipsolverComplex*    A,
                                               int                  lda,
                                               hipsolverComplex*    tau,
                                               hipsolverComplex*    C,
                                               int                  ldc,
                                               hipsolverComplex*    work,
                                               int                  lwork,
                                               int*                 info)
{

        return hipsolverCunmqr(handle,
                               side,
                               trans,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)C,
                               ldc,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverSideMode_t     side,
                                               hipsolverOperation_t    trans,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* C,
                                               int                     ldc,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{

        return hipsolverZunmqr(handle,
                               side,
                               trans,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)C,
                               ldc,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** ORMTR/UNMTR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
{

        return hipsolverSormtr_bufferSize(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);

}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
{

        return hipsolverDormtr_bufferSize(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);

}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverFillMode_t  uplo,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          hipsolverComplex*    tau,
                                                          hipsolverComplex*    C,
                                                          int                  ldc,
                                                          int*                 lwork)
{

        return hipsolverCunmtr_bufferSize(handle,
                                          side,
                                          uplo,
                                          trans,
                                          m,
                                          n,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)tau,
                                          (hipFloatComplex*)C,
                                          ldc,
                                          lwork);

}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          hipsolverFillMode_t     uplo,
                                                          hipsolverOperation_t    trans,
                                                          int                     m,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* C,
                                                          int                     ldc,
                                                          int*                    lwork)
{

        return hipsolverZunmtr_bufferSize(handle,
                                          side,
                                          uplo,
                                          trans,
                                          m,
                                          n,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)tau,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          lwork);

}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{

        return hipsolverSormtr(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{

        return hipsolverDormtr(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
                                               hipsolverSideMode_t  side,
                                               hipsolverFillMode_t  uplo,
                                               hipsolverOperation_t trans,
                                               int                  m,
                                               int                  n,
                                               hipsolverComplex*    A,
                                               int                  lda,
                                               hipsolverComplex*    tau,
                                               hipsolverComplex*    C,
                                               int                  ldc,
                                               hipsolverComplex*    work,
                                               int                  lwork,
                                               int*                 info)
{

        return hipsolverCunmtr(handle,
                               side,
                               uplo,
                               trans,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)C,
                               ldc,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverSideMode_t     side,
                                               hipsolverFillMode_t     uplo,
                                               hipsolverOperation_t    trans,
                                               int                     m,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* C,
                                               int                     ldc,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{

        return hipsolverZunmtr(handle,
                               side,
                               uplo,
                               trans,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)C,
                               ldc,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** GEBRD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{

        return hipsolverSgebrd_bufferSize(handle, m, n, lwork);

}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{

        return hipsolverDgebrd_bufferSize(handle, m, n, lwork);

}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{

        return hipsolverCgebrd_bufferSize(handle, m, n, lwork);

}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{

        return hipsolverZgebrd_bufferSize(handle, m, n, lwork);

}

inline hipsolverStatus_t hipsolver_gebrd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            D,
                                         int               stD,
                                         float*            E,
                                         int               stE,
                                         float*            tauq,
                                         int               stQ,
                                         float*            taup,
                                         int               stP,
                                         float*            work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{

        return hipsolverSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_gebrd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           D,
                                         int               stD,
                                         double*           E,
                                         int               stE,
                                         double*           tauq,
                                         int               stQ,
                                         double*           taup,
                                         int               stP,
                                         double*           work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{

        return hipsolverDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_gebrd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         float*            D,
                                         int               stD,
                                         float*            E,
                                         int               stE,
                                         hipsolverComplex* tauq,
                                         int               stQ,
                                         hipsolverComplex* taup,
                                         int               stP,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{

        return hipsolverCgebrd(handle,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               D,
                               E,
                               (hipFloatComplex*)tauq,
                               (hipFloatComplex*)taup,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_gebrd(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         double*                 D,
                                         int                     stD,
                                         double*                 E,
                                         int                     stE,
                                         hipsolverDoubleComplex* tauq,
                                         int                     stQ,
                                         hipsolverDoubleComplex* taup,
                                         int                     stP,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{

        return hipsolverZgebrd(handle,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               D,
                               E,
                               (hipDoubleComplex*)tauq,
                               (hipDoubleComplex*)taup,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** GEQRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
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
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);


    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
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
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);


    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   hipsolverComplex* A,
                                                   int               lda,
                                                   hipsolverComplex* B,
                                                   int               ldb,
                                                   hipsolverComplex* X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCCgels_bufferSize(handle,
                                          m,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          (hipFloatComplex*)X,
                                          ldx,
                                          lwork);


    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t               API,
                                                   hipsolverHandle_t       handle,
                                                   int                     m,
                                                   int                     n,
                                                   int                     nrhs,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   hipsolverDoubleComplex* B,
                                                   int                     ldb,
                                                   hipsolverDoubleComplex* X,
                                                   int                     ldx,
                                                   size_t*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZZgels_bufferSize(handle,
                                          m,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          (hipDoubleComplex*)X,
                                          ldx,
                                          lwork);


    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        float*            A,
                                        int               lda,
                                        int               stA,
                                        float*            B,
                                        int               ldb,
                                        int               stB,
                                        float*            X,
                                        int               ldx,
                                        int               stX,
                                        float*            work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverSSgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        double*           A,
                                        int               lda,
                                        int               stA,
                                        double*           B,
                                        int               ldb,
                                        int               stB,
                                        double*           X,
                                        int               ldx,
                                        int               stX,
                                        double*           work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverDDgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        hipsolverComplex* A,
                                        int               lda,
                                        int               stA,
                                        hipsolverComplex* B,
                                        int               ldb,
                                        int               stB,
                                        hipsolverComplex* X,
                                        int               ldx,
                                        int               stX,
                                        hipsolverComplex* work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverCCgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t               API,
                                        bool                    INPLACE,
                                        hipsolverHandle_t       handle,
                                        int                     m,
                                        int                     n,
                                        int                     nrhs,
                                        hipsolverDoubleComplex* A,
                                        int                     lda,
                                        int                     stA,
                                        hipsolverDoubleComplex* B,
                                        int                     ldb,
                                        int                     stB,
                                        hipsolverDoubleComplex* X,
                                        int                     ldx,
                                        int                     stX,
                                        hipsolverDoubleComplex* work,
                                        size_t                  lwork,
                                        int*                    niters,
                                        int*                    info,
                                        int                     bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverZZgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GEQRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_geqrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{

        return hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, lwork);

}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{

        return hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, lwork);

}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{

        return hipsolverCgeqrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lwork);

}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{

        return hipsolverZgeqrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lwork);

}

inline hipsolverStatus_t hipsolver_geqrf(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            tau,
                                         int               stT,
                                         float*            work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{

        return hipsolverSgeqrf(handle, m, n, A, lda, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_geqrf(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           tau,
                                         int               stT,
                                         double*           work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{

        return hipsolverDgeqrf(handle, m, n, A, lda, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_geqrf(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         hipsolverComplex* tau,
                                         int               stT,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{

        return hipsolverCgeqrf(handle,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_geqrf(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* tau,
                                         int                     stT,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{

        return hipsolverZgeqrf(handle,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** GESV ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   float*            A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   float*            B,
                                                   int               ldb,
                                                   float*            X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);


    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   double*           A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   double*           B,
                                                   int               ldb,
                                                   double*           X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);

    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   hipsolverComplex* A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   hipsolverComplex* B,
                                                   int               ldb,
                                                   hipsolverComplex* X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCCgesv_bufferSize(handle,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          (hipFloatComplex*)X,
                                          ldx,
                                          lwork);


    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t               API,
                                                   hipsolverHandle_t       handle,
                                                   int                     n,
                                                   int                     nrhs,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   int*                    ipiv,
                                                   hipsolverDoubleComplex* B,
                                                   int                     ldb,
                                                   hipsolverDoubleComplex* X,
                                                   int                     ldx,
                                                   size_t*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZZgesv_bufferSize(handle,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          (hipDoubleComplex*)X,
                                          ldx,
                                          lwork);


    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        float*            A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        float*            B,
                                        int               ldb,
                                        int               stB,
                                        float*            X,
                                        int               ldx,
                                        int               stX,
                                        float*            work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverSSgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        double*           A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        double*           B,
                                        int               ldb,
                                        int               stB,
                                        double*           X,
                                        int               ldx,
                                        int               stX,
                                        double*           work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverDDgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        hipsolverComplex* A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        hipsolverComplex* B,
                                        int               ldb,
                                        int               stB,
                                        hipsolverComplex* X,
                                        int               ldx,
                                        int               stX,
                                        hipsolverComplex* work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverCCgesv(handle,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               ipiv,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t               API,
                                        bool                    INPLACE,
                                        hipsolverHandle_t       handle,
                                        int                     n,
                                        int                     nrhs,
                                        hipsolverDoubleComplex* A,
                                        int                     lda,
                                        int                     stA,
                                        int*                    ipiv,
                                        int                     stP,
                                        hipsolverDoubleComplex* B,
                                        int                     ldb,
                                        int                     stB,
                                        hipsolverDoubleComplex* X,
                                        int                     ldx,
                                        int                     stX,
                                        hipsolverDoubleComplex* work,
                                        size_t                  lwork,
                                        int*                    niters,
                                        int*                    info,
                                        int                     bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverZZgesv(handle,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               ipiv,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GESVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    float*            A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);

    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    double*           A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);

    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    hipsolverComplex* A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);

    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    signed char             jobu,
                                                    signed char             jobv,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);

    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
                                         bool              NRWK,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            S,
                                         int               stS,
                                         float*            U,
                                         int               ldu,
                                         int               stU,
                                         float*            V,
                                         int               ldv,
                                         int               stV,
                                         float*            work,
                                         int               lwork,
                                         float*            rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NRWK))
    {
    case C_NORMAL:
        return hipsolverSgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
                                         bool              NRWK,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           S,
                                         int               stS,
                                         double*           U,
                                         int               ldu,
                                         int               stU,
                                         double*           V,
                                         int               ldv,
                                         int               stV,
                                         double*           work,
                                         int               lwork,
                                         double*           rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NRWK))
    {
    case C_NORMAL:
        return hipsolverDgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
                                         bool              NRWK,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         float*            S,
                                         int               stS,
                                         hipsolverComplex* U,
                                         int               ldu,
                                         int               stU,
                                         hipsolverComplex* V,
                                         int               ldv,
                                         int               stV,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         float*            rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NRWK))
    {
    case C_NORMAL:
        return hipsolverCgesvd(handle,
                               jobu,
                               jobv,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               S,
                               (hipFloatComplex*)U,
                               ldu,
                               (hipFloatComplex*)V,
                               ldv,
                               (hipFloatComplex*)work,
                               lwork,
                               rwork,
                               info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t               API,
                                         bool                    NRWK,
                                         hipsolverHandle_t       handle,
                                         signed char             jobu,
                                         signed char             jobv,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         double*                 S,
                                         int                     stS,
                                         hipsolverDoubleComplex* U,
                                         int                     ldu,
                                         int                     stU,
                                         hipsolverDoubleComplex* V,
                                         int                     ldv,
                                         int                     stV,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         double*                 rwork,
                                         int                     stRW,
                                         int*                    info,
                                         int                     bc)
{
    switch(api2marshal(API, NRWK))
    {
    case C_NORMAL:
        return hipsolverZgesvd(handle,
                               jobu,
                               jobv,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               S,
                               (hipDoubleComplex*)U,
                               ldu,
                               (hipDoubleComplex*)V,
                               ldv,
                               (hipDoubleComplex*)work,
                               lwork,
                               rwork,
                               info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GESVDJ ********************/
inline hipsolverStatus_t hipsolver_gesvdj_bufferSize(testAPI_t             API,
                                                     bool                  STRIDED,
                                                     hipsolverHandle_t     handle,
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
                                                     int*                  lwork,
                                                     hipsolverGesvdjInfo_t params,
                                                     int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverSgesvdj_bufferSize(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);


    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj_bufferSize(testAPI_t             API,
                                                     bool                  STRIDED,
                                                     hipsolverHandle_t     handle,
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
                                                     int*                  lwork,
                                                     hipsolverGesvdjInfo_t params,
                                                     int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverDgesvdj_bufferSize(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj_bufferSize(testAPI_t             API,
                                                     bool                  STRIDED,
                                                     hipsolverHandle_t     handle,
                                                     hipsolverEigMode_t    jobz,
                                                     int                   econ,
                                                     int                   m,
                                                     int                   n,
                                                     hipsolverComplex*     A,
                                                     int                   lda,
                                                     float*                S,
                                                     hipsolverComplex*     U,
                                                     int                   ldu,
                                                     hipsolverComplex*     V,
                                                     int                   ldv,
                                                     int*                  lwork,
                                                     hipsolverGesvdjInfo_t params,
                                                     int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverCgesvdj_bufferSize(handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           (hipFloatComplex*)A,
                                           lda,
                                           S,
                                           (hipFloatComplex*)U,
                                           ldu,
                                           (hipFloatComplex*)V,
                                           ldv,
                                           lwork,
                                           params);
    
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj_bufferSize(testAPI_t               API,
                                                     bool                    STRIDED,
                                                     hipsolverHandle_t       handle,
                                                     hipsolverEigMode_t      jobz,
                                                     int                     econ,
                                                     int                     m,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     double*                 S,
                                                     hipsolverDoubleComplex* U,
                                                     int                     ldu,
                                                     hipsolverDoubleComplex* V,
                                                     int                     ldv,
                                                     int*                    lwork,
                                                     hipsolverGesvdjInfo_t   params,
                                                     int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverZgesvdj_bufferSize(handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           (hipDoubleComplex*)A,
                                           lda,
                                           S,
                                           (hipDoubleComplex*)U,
                                           ldu,
                                           (hipDoubleComplex*)V,
                                           ldv,
                                           lwork,
                                           params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj(testAPI_t             API,
                                          bool                  STRIDED,
                                          hipsolverHandle_t     handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   econ,
                                          int                   m,
                                          int                   n,
                                          float*                A,
                                          int                   lda,
                                          int                   stA,
                                          float*                S,
                                          int                   stS,
                                          float*                U,
                                          int                   ldu,
                                          int                   stU,
                                          float*                V,
                                          int                   ldv,
                                          int                   stV,
                                          float*                work,
                                          int                   lwork,
                                          int*                  info,
                                          hipsolverGesvdjInfo_t params,
                                          int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverSgesvdj(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj(testAPI_t             API,
                                          bool                  STRIDED,
                                          hipsolverHandle_t     handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   econ,
                                          int                   m,
                                          int                   n,
                                          double*               A,
                                          int                   lda,
                                          int                   stA,
                                          double*               S,
                                          int                   stS,
                                          double*               U,
                                          int                   ldu,
                                          int                   stU,
                                          double*               V,
                                          int                   ldv,
                                          int                   stV,
                                          double*               work,
                                          int                   lwork,
                                          int*                  info,
                                          hipsolverGesvdjInfo_t params,
                                          int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverDgesvdj(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj(testAPI_t             API,
                                          bool                  STRIDED,
                                          hipsolverHandle_t     handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   econ,
                                          int                   m,
                                          int                   n,
                                          hipsolverComplex*     A,
                                          int                   lda,
                                          int                   stA,
                                          float*                S,
                                          int                   stS,
                                          hipsolverComplex*     U,
                                          int                   ldu,
                                          int                   stU,
                                          hipsolverComplex*     V,
                                          int                   ldv,
                                          int                   stV,
                                          hipsolverComplex*     work,
                                          int                   lwork,
                                          int*                  info,
                                          hipsolverGesvdjInfo_t params,
                                          int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverCgesvdj(handle,
                                jobz,
                                econ,
                                m,
                                n,
                                (hipFloatComplex*)A,
                                lda,
                                S,
                                (hipFloatComplex*)U,
                                ldu,
                                (hipFloatComplex*)V,
                                ldv,
                                (hipFloatComplex*)work,
                                lwork,
                                info,
                                params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj(testAPI_t               API,
                                          bool                    STRIDED,
                                          hipsolverHandle_t       handle,
                                          hipsolverEigMode_t      jobz,
                                          int                     econ,
                                          int                     m,
                                          int                     n,
                                          hipsolverDoubleComplex* A,
                                          int                     lda,
                                          int                     stA,
                                          double*                 S,
                                          int                     stS,
                                          hipsolverDoubleComplex* U,
                                          int                     ldu,
                                          int                     stU,
                                          hipsolverDoubleComplex* V,
                                          int                     ldv,
                                          int                     stV,
                                          hipsolverDoubleComplex* work,
                                          int                     lwork,
                                          int*                    info,
                                          hipsolverGesvdjInfo_t   params,
                                          int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverZgesvdj(handle,
                                jobz,
                                econ,
                                m,
                                n,
                                (hipDoubleComplex*)A,
                                lda,
                                S,
                                (hipDoubleComplex*)U,
                                ldu,
                                (hipDoubleComplex*)V,
                                ldv,
                                (hipDoubleComplex*)work,
                                lwork,
                                info,
                                params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GESVDA ********************/
inline hipsolverStatus_t hipsolver_gesvda_bufferSize(testAPI_t          API,
                                                     bool               STRIDED,
                                                     hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     int                rank,
                                                     int                m,
                                                     int                n,
                                                     float*             A,
                                                     int                lda,
                                                     long long int      stA,
                                                     float*             S,
                                                     long long int      stS,
                                                     float*             U,
                                                     int                ldu,
                                                     long long int      stU,
                                                     float*             V,
                                                     int                ldv,
                                                     long long int      stV,
                                                     int*               lwork,
                                                     int                bc)
{
    switch(api2marshal(API, STRIDED))
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda_bufferSize(testAPI_t          API,
                                                     bool               STRIDED,
                                                     hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     int                rank,
                                                     int                m,
                                                     int                n,
                                                     double*            A,
                                                     int                lda,
                                                     long long int      stA,
                                                     double*            S,
                                                     long long int      stS,
                                                     double*            U,
                                                     int                ldu,
                                                     long long int      stU,
                                                     double*            V,
                                                     int                ldv,
                                                     long long int      stV,
                                                     int*               lwork,
                                                     int                bc)
{
    switch(api2marshal(API, STRIDED))
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda_bufferSize(testAPI_t          API,
                                                     bool               STRIDED,
                                                     hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     int                rank,
                                                     int                m,
                                                     int                n,
                                                     hipsolverComplex*  A,
                                                     int                lda,
                                                     long long int      stA,
                                                     float*             S,
                                                     long long int      stS,
                                                     hipsolverComplex*  U,
                                                     int                ldu,
                                                     long long int      stU,
                                                     hipsolverComplex*  V,
                                                     int                ldv,
                                                     long long int      stV,
                                                     int*               lwork,
                                                     int                bc)
{
    switch(api2marshal(API, STRIDED))
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda_bufferSize(testAPI_t               API,
                                                     bool                    STRIDED,
                                                     hipsolverHandle_t       handle,
                                                     hipsolverEigMode_t      jobz,
                                                     int                     rank,
                                                     int                     m,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     long long int           stA,
                                                     double*                 S,
                                                     long long int           stS,
                                                     hipsolverDoubleComplex* U,
                                                     int                     ldu,
                                                     long long int           stU,
                                                     hipsolverDoubleComplex* V,
                                                     int                     ldv,
                                                     long long int           stV,
                                                     int*                    lwork,
                                                     int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda(testAPI_t          API,
                                          bool               STRIDED,
                                          hipsolverHandle_t  handle,
                                          hipsolverEigMode_t jobz,
                                          int                rank,
                                          int                m,
                                          int                n,
                                          float*             A,
                                          int                lda,
                                          int                stA,
                                          float*             S,
                                          int                stS,
                                          float*             U,
                                          int                ldu,
                                          int                stU,
                                          float*             V,
                                          int                ldv,
                                          int                stV,
                                          float*             work,
                                          int                lwork,
                                          int*               info,
                                          double*            hRnrmF,
                                          int                bc)
{
    switch(api2marshal(API, STRIDED))
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda(testAPI_t          API,
                                          bool               STRIDED,
                                          hipsolverHandle_t  handle,
                                          hipsolverEigMode_t jobz,
                                          int                rank,
                                          int                m,
                                          int                n,
                                          double*            A,
                                          int                lda,
                                          int                stA,
                                          double*            S,
                                          int                stS,
                                          double*            U,
                                          int                ldu,
                                          int                stU,
                                          double*            V,
                                          int                ldv,
                                          int                stV,
                                          double*            work,
                                          int                lwork,
                                          int*               info,
                                          double*            hRnrmF,
                                          int                bc)
{
    switch(api2marshal(API, STRIDED))
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda(testAPI_t          API,
                                          bool               STRIDED,
                                          hipsolverHandle_t  handle,
                                          hipsolverEigMode_t jobz,
                                          int                rank,
                                          int                m,
                                          int                n,
                                          hipsolverComplex*  A,
                                          int                lda,
                                          int                stA,
                                          float*             S,
                                          int                stS,
                                          hipsolverComplex*  U,
                                          int                ldu,
                                          int                stU,
                                          hipsolverComplex*  V,
                                          int                ldv,
                                          int                stV,
                                          hipsolverComplex*  work,
                                          int                lwork,
                                          int*               info,
                                          double*            hRnrmF,
                                          int                bc)
{
    switch(api2marshal(API, STRIDED))
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda(testAPI_t               API,
                                          bool                    STRIDED,
                                          hipsolverHandle_t       handle,
                                          hipsolverEigMode_t      jobz,
                                          int                     rank,
                                          int                     m,
                                          int                     n,
                                          hipsolverDoubleComplex* A,
                                          int                     lda,
                                          int                     stA,
                                          double*                 S,
                                          int                     stS,
                                          hipsolverDoubleComplex* U,
                                          int                     ldu,
                                          int                     stU,
                                          hipsolverDoubleComplex* V,
                                          int                     ldv,
                                          int                     stV,
                                          hipsolverDoubleComplex* work,
                                          int                     lwork,
                                          int*                    info,
                                          double*                 hRnrmF,
                                          int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GETRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork);


    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork);


    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lwork);


    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lwork);


    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t         API,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            work,
                                         int               lwork,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverSgetrf(handle, m, n, A, lda, work, lwork, ipiv, info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t         API,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           work,
                                         int               lwork,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverDgetrf(handle, m, n, A, lda, work, lwork, ipiv, info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t         API,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverCgetrf(
            handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, ipiv, info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t               API,
                                         bool                    NPVT,
                                         hipsolverHandle_t       handle,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    ipiv,
                                         int                     stP,
                                         int*                    info,
                                         int                     bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverZgetrf(
            handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, ipiv, info);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GETRS ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    float*               A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    float*               B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    //case API_FORTRAN:
    //    return hipsolverSgetrs_bufferSizeFortran(
    //        handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    double*              A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    double*              B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
/*     case API_FORTRAN:
        return hipsolverDgetrs_bufferSizeFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    hipsolverComplex*    A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    hipsolverComplex*    B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrs_bufferSize(handle,
                                          trans,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          lwork);
/*     case API_FORTRAN:
        return hipsolverCgetrs_bufferSizeFortran(handle,
                                                 trans,
                                                 n,
                                                 nrhs,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 lwork); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverOperation_t    trans,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    ipiv,
                                                    hipsolverDoubleComplex* B,
                                                    int                     ldb,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrs_bufferSize(handle,
                                          trans,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          lwork);
/*     case API_FORTRAN:
        return hipsolverZgetrs_bufferSizeFortran(handle,
                                                 trans,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 lwork); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         float*               A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         float*               B,
                                         int                  ldb,
                                         int                  stB,
                                         float*               work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
/*     case API_FORTRAN:
        return hipsolverSgetrsFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         double*              A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         double*              B,
                                         int                  ldb,
                                         int                  stB,
                                         double*              work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
/*     case API_FORTRAN:
        return hipsolverDgetrsFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         hipsolverComplex*    A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         hipsolverComplex*    B,
                                         int                  ldb,
                                         int                  stB,
                                         hipsolverComplex*    work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrs(handle,
                               trans,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               ipiv,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
/*     case API_FORTRAN:
        return hipsolverCgetrsFortran(handle,
                                      trans,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverOperation_t    trans,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         int*                    ipiv,
                                         int                     stP,
                                         hipsolverDoubleComplex* B,
                                         int                     ldb,
                                         int                     stB,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrs(handle,
                               trans,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               ipiv,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
 /*    case API_FORTRAN:
        return hipsolverZgetrsFortran(handle,
                                      trans,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** POTRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
/*     case API_FORTRAN:
        return hipsolverSpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork); */

    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
/*     case API_FORTRAN:
        return hipsolverDpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork); */

    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrf_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
/*     case API_FORTRAN:
        return hipsolverCpotrf_bufferSizeFortran(handle, uplo, n, (hipFloatComplex*)A, lda, lwork); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrf_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
/*     case API_FORTRAN:
        return hipsolverZpotrf_bufferSizeFortran(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork); */

    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, info);
/*     case API_FORTRAN:
        return hipsolverSpotrfFortran(handle, uplo, n, A, lda, work, lwork, info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, info);
/*     case API_FORTRAN:
        return hipsolverDpotrfFortran(handle, uplo, n, A, lda, work, lwork, info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
/*     case API_FORTRAN:
        return hipsolverCpotrfFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
/*     case API_FORTRAN:
        return hipsolverZpotrfFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info); */
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

// batched
inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, bc);
/*     case API_FORTRAN:
        return hipsolverSpotrfBatched_bufferSizeFortran(handle, uplo, n, A, lda, lwork, bc); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, bc);
/*     case API_FORTRAN:
        return hipsolverDpotrfBatched_bufferSizeFortran(handle, uplo, n, A, lda, lwork, bc); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrfBatched_bufferSize(
            handle, uplo, n, (hipFloatComplex**)A, lda, lwork, bc);
/*     case API_FORTRAN:
        return hipsolverCpotrfBatched_bufferSizeFortran(
            handle, uplo, n, (hipFloatComplex**)A, lda, lwork, bc); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A[],
                                                    int                     lda,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrfBatched_bufferSize(
            handle, uplo, n, (hipDoubleComplex**)A, lda, lwork, bc);
/*     case API_FORTRAN:
        return hipsolverZpotrfBatched_bufferSizeFortran(
            handle, uplo, n, (hipDoubleComplex**)A, lda, lwork, bc); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A[],
                                         int                 lda,
                                         int                 stA,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, bc);
/*     case API_FORTRAN:
        return hipsolverSpotrfBatchedFortran(handle, uplo, n, A, lda, work, lwork, info, bc); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A[],
                                         int                 lda,
                                         int                 stA,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, bc);
/*     case API_FORTRAN:
        return hipsolverDpotrfBatchedFortran(handle, uplo, n, A, lda, work, lwork, info, bc); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A[],
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrfBatched(
            handle, uplo, n, (hipFloatComplex**)A, lda, (hipFloatComplex*)work, lwork, info, bc);
/*     case API_FORTRAN:
        return hipsolverCpotrfBatchedFortran(
            handle, uplo, n, (hipFloatComplex**)A, lda, (hipFloatComplex*)work, lwork, info, bc); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A[],
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrfBatched(
            handle, uplo, n, (hipDoubleComplex**)A, lda, (hipDoubleComplex*)work, lwork, info, bc);
/*     case API_FORTRAN:
        return hipsolverZpotrfBatchedFortran(
            handle, uplo, n, (hipDoubleComplex**)A, lda, (hipDoubleComplex*)work, lwork, info, bc); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** POTRI ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potri_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    int*                lwork)
{

        return hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, lwork);
/*     else
        return hipsolverSpotri_bufferSizeFortran(handle, uplo, n, A, lda, lwork); */
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    int*                lwork)
{

        return hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, lwork);
/*     else
        return hipsolverDpotri_bufferSizeFortran(handle, uplo, n, A, lda, lwork); */
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    int*                lwork)
{

        return hipsolverCpotri_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
/*     else
        return hipsolverCpotri_bufferSizeFortran(handle, uplo, n, (hipFloatComplex*)A, lda, lwork); */
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{

        return hipsolverZpotri_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
/*     else
        return hipsolverZpotri_bufferSizeFortran(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork); */
}

inline hipsolverStatus_t hipsolver_potri(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{

        return hipsolverSpotri(handle, uplo, n, A, lda, work, lwork, info);
/*     else
        return hipsolverSpotriFortran(handle, uplo, n, A, lda, work, lwork, info); */
}

inline hipsolverStatus_t hipsolver_potri(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{

        return hipsolverDpotri(handle, uplo, n, A, lda, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_potri(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{

        return hipsolverCpotri(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);

}

inline hipsolverStatus_t hipsolver_potri(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{

        return hipsolverZpotri(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
/*     else
        return hipsolverZpotriFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info); */
}
/********************************************************/

/******************** POTRS ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    float*              A,
                                                    int                 lda,
                                                    float*              B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
/*     case API_FORTRAN:
        return hipsolverSpotrs_bufferSizeFortran(handle, uplo, n, nrhs, A, lda, B, ldb, lwork); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    double*             A,
                                                    int                 lda,
                                                    double*             B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
/*     case API_FORTRAN:
        return hipsolverDpotrs_bufferSizeFortran(handle, uplo, n, nrhs, A, lda, B, ldb, lwork); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    hipsolverComplex*   B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrs_bufferSize(
            handle, uplo, n, nrhs, (hipFloatComplex*)A, lda, (hipFloatComplex*)B, ldb, lwork);
/*     case API_FORTRAN:
        return hipsolverCpotrs_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipFloatComplex*)A, lda, (hipFloatComplex*)B, ldb, lwork); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    hipsolverDoubleComplex* B,
                                                    int                     ldb,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrs_bufferSize(
            handle, uplo, n, nrhs, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)B, ldb, lwork);
/*     case API_FORTRAN:
        return hipsolverZpotrs_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)B, ldb, lwork); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              B,
                                         int                 ldb,
                                         int                 stB,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
/*     case API_FORTRAN:
        return hipsolverSpotrsFortran(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             B,
                                         int                 ldb,
                                         int                 stB,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
/*     case API_FORTRAN:
        return hipsolverDpotrsFortran(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   B,
                                         int                 ldb,
                                         int                 stB,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrs(handle,
                               uplo,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
/*     case API_FORTRAN:
        return hipsolverCpotrsFortran(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* B,
                                         int                     ldb,
                                         int                     stB,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrs(handle,
                               uplo,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
/*     case API_FORTRAN:
        return hipsolverZpotrsFortran(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

// batched
inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    float*              A[],
                                                    int                 lda,
                                                    float*              B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
/*     case API_FORTRAN:
        return hipsolverSpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    double*             A[],
                                                    int                 lda,
                                                    double*             B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
/*     case API_FORTRAN:
        return hipsolverDpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipsolverComplex*   A[],
                                                    int                 lda,
                                                    hipsolverComplex*   B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrsBatched_bufferSize(
            handle, uplo, n, nrhs, (hipFloatComplex**)A, lda, (hipFloatComplex**)B, ldb, lwork, bc);
/*     case API_FORTRAN:
        return hipsolverCpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipFloatComplex**)A, lda, (hipFloatComplex**)B, ldb, lwork, bc); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A[],
                                                    int                     lda,
                                                    hipsolverDoubleComplex* B[],
                                                    int                     ldb,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrsBatched_bufferSize(handle,
                                                 uplo,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex**)A,
                                                 lda,
                                                 (hipDoubleComplex**)B,
                                                 ldb,
                                                 lwork,
                                                 bc);
/*     case API_FORTRAN:
        return hipsolverZpotrsBatched_bufferSizeFortran(handle,
                                                        uplo,
                                                        n,
                                                        nrhs,
                                                        (hipDoubleComplex**)A,
                                                        lda,
                                                        (hipDoubleComplex**)B,
                                                        ldb,
                                                        lwork,
                                                        bc); */
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         float*              A[],
                                         int                 lda,
                                         int                 stA,
                                         float*              B[],
                                         int                 ldb,
                                         int                 stB,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
/*     case API_FORTRAN:
        return hipsolverSpotrsBatchedFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         double*             A[],
                                         int                 lda,
                                         int                 stA,
                                         double*             B[],
                                         int                 ldb,
                                         int                 stB,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
/*     case API_FORTRAN:
        return hipsolverDpotrsBatchedFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         hipsolverComplex*   A[],
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   B[],
                                         int                 ldb,
                                         int                 stB,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrsBatched(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipFloatComplex**)A,
                                      lda,
                                      (hipFloatComplex**)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info,
                                      bc);
/*     case API_FORTRAN:
        return hipsolverCpotrsBatchedFortran(handle,
                                             uplo,
                                             n,
                                             nrhs,
                                             (hipFloatComplex**)A,
                                             lda,
                                             (hipFloatComplex**)B,
                                             ldb,
                                             (hipFloatComplex*)work,
                                             lwork,
                                             info,
                                             bc); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A[],
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* B[],
                                         int                     ldb,
                                         int                     stB,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrsBatched(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex**)A,
                                      lda,
                                      (hipDoubleComplex**)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info,
                                      bc);
/*     case API_FORTRAN:
        return hipsolverZpotrsBatchedFortran(handle,
                                             uplo,
                                             n,
                                             nrhs,
                                             (hipDoubleComplex**)A,
                                             lda,
                                             (hipDoubleComplex**)B,
                                             ldb,
                                             (hipDoubleComplex*)work,
                                             lwork,
                                             info,
                                             bc); */

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYEVD/HEEVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              W,
                                                          int*                lwork)
{

        return hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
/*     else
        return hipsolverSsyevd_bufferSizeFortran(handle, jobz, uplo, n, A, lda, W, lwork); */
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             W,
                                                          int*                lwork)
{

        return hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
/*     else
        return hipsolverDsyevd_bufferSizeFortran(handle, jobz, uplo, n, A, lda, W, lwork); */
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          float*              W,
                                                          int*                lwork)
{

        return hipsolverCheevd_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork);
/*     else
        return hipsolverCheevd_bufferSizeFortran(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork); */
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 W,
                                                          int*                    lwork)
{

        return hipsolverZheevd_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork);
/*     else
        return hipsolverZheevd_bufferSizeFortran(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork); */
}

inline hipsolverStatus_t hipsolver_syevd_heevd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              W,
                                               int                 stW,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
/*     else
        return hipsolverSsyevdFortran(handle, jobz, uplo, n, A, lda, W, work, lwork, info); */
}

inline hipsolverStatus_t hipsolver_syevd_heevd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             W,
                                               int                 stW,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
/*     else
        return hipsolverDsyevdFortran(handle, jobz, uplo, n, A, lda, W, work, lwork, info); */
}

inline hipsolverStatus_t hipsolver_syevd_heevd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               float*              W,
                                               int                 stW,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverCheevd(handle,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               W,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
/*     else
        return hipsolverCheevdFortran(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      W,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info); */
}

inline hipsolverStatus_t hipsolver_syevd_heevd(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               double*                 W,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{

        return hipsolverZheevd(handle,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               W,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** SYEVDX/HEEVDX ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_syevdx_heevdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            float*              A,
                                                            int                 lda,
                                                            float               vl,
                                                            float               vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            float*              W,
                                                            int*                lwork)
{
    switch(api2marshal(API, false))
    {
    case COMPAT_NORMAL:
        return hipsolverDnSsyevdx_bufferSize(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            double*             A,
                                                            int                 lda,
                                                            double              vl,
                                                            double              vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            double*             W,
                                                            int*                lwork)
{
    switch(api2marshal(API, false))
    {
    case COMPAT_NORMAL:
        return hipsolverDnDsyevdx_bufferSize(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            hipsolverComplex*   A,
                                                            int                 lda,
                                                            float               vl,
                                                            float               vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            float*              W,
                                                            int*                lwork)
{
    switch(api2marshal(API, false))
    {
    case COMPAT_NORMAL:
        return hipsolverDnCheevdx_bufferSize(
            handle, jobz, range, uplo, n, (hipFloatComplex*)A, lda, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx_bufferSize(testAPI_t               API,
                                                            hipsolverHandle_t       handle,
                                                            hipsolverEigMode_t      jobz,
                                                            hipsolverEigRange_t     range,
                                                            hipsolverFillMode_t     uplo,
                                                            int                     n,
                                                            hipsolverDoubleComplex* A,
                                                            int                     lda,
                                                            double                  vl,
                                                            double                  vu,
                                                            int                     il,
                                                            int                     iu,
                                                            int*                    nev,
                                                            double*                 W,
                                                            int*                    lwork)
{
    switch(api2marshal(API, false))
    {
    case COMPAT_NORMAL:
        return hipsolverDnZheevdx_bufferSize(
            handle, jobz, range, uplo, n, (hipDoubleComplex*)A, lda, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 float*              A,
                                                 int                 lda,
                                                 int                 stA,
                                                 float               vl,
                                                 float               vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 float*              W,
                                                 int                 stW,
                                                 float*              work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(api2marshal(API, false))
    {
    case COMPAT_NORMAL:
        return hipsolverDnSsyevdx(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 double*             A,
                                                 int                 lda,
                                                 int                 stA,
                                                 double              vl,
                                                 double              vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 double*             W,
                                                 int                 stW,
                                                 double*             work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(api2marshal(API, false))
    {
    case COMPAT_NORMAL:
        return hipsolverDnDsyevdx(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 hipsolverComplex*   A,
                                                 int                 lda,
                                                 int                 stA,
                                                 float               vl,
                                                 float               vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 float*              W,
                                                 int                 stW,
                                                 hipsolverComplex*   work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(api2marshal(API, false))
    {
    case COMPAT_NORMAL:
        return hipsolverDnCheevdx(handle,
                                  jobz,
                                  range,
                                  uplo,
                                  n,
                                  (hipFloatComplex*)A,
                                  lda,
                                  vl,
                                  vu,
                                  il,
                                  iu,
                                  nev,
                                  W,
                                  (hipFloatComplex*)work,
                                  lwork,
                                  info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx(testAPI_t               API,
                                                 hipsolverHandle_t       handle,
                                                 hipsolverEigMode_t      jobz,
                                                 hipsolverEigRange_t     range,
                                                 hipsolverFillMode_t     uplo,
                                                 int                     n,
                                                 hipsolverDoubleComplex* A,
                                                 int                     lda,
                                                 int                     stA,
                                                 double                  vl,
                                                 double                  vu,
                                                 int                     il,
                                                 int                     iu,
                                                 int*                    nev,
                                                 double*                 W,
                                                 int                     stW,
                                                 hipsolverDoubleComplex* work,
                                                 int                     lwork,
                                                 int*                    info,
                                                 int                     bc)
{
    switch(api2marshal(API, false))
    {
    case COMPAT_NORMAL:
        return hipsolverDnZheevdx(handle,
                                  jobz,
                                  range,
                                  uplo,
                                  n,
                                  (hipDoubleComplex*)A,
                                  lda,
                                  vl,
                                  vu,
                                  il,
                                  iu,
                                  nev,
                                  W,
                                  (hipDoubleComplex*)work,
                                  lwork,
                                  info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYEVJ/HEEVJ ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_syevj_heevj_bufferSize(testAPI_t            API,
                                                          bool                 STRIDED,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          float*               A,
                                                          int                  lda,
                                                          float*               W,
                                                          int*                 lwork,
                                                          hipsolverSyevjInfo_t params,
                                                          int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj_bufferSize(testAPI_t            API,
                                                          bool                 STRIDED,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          double*              A,
                                                          int                  lda,
                                                          double*              W,
                                                          int*                 lwork,
                                                          hipsolverSyevjInfo_t params,
                                                          int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj_bufferSize(testAPI_t            API,
                                                          bool                 STRIDED,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          float*               W,
                                                          int*                 lwork,
                                                          hipsolverSyevjInfo_t params,
                                                          int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverCheevj_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj_bufferSize(testAPI_t               API,
                                                          bool                    STRIDED,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 W,
                                                          int*                    lwork,
                                                          hipsolverSyevjInfo_t    params,
                                                          int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverZheevj_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj(testAPI_t            API,
                                               bool                 STRIDED,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               float*               A,
                                               int                  lda,
                                               int                  stA,
                                               float*               W,
                                               int                  stW,
                                               float*               work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj(testAPI_t            API,
                                               bool                 STRIDED,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               double*              A,
                                               int                  lda,
                                               int                  stA,
                                               double*              W,
                                               int                  stW,
                                               double*              work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj(testAPI_t            API,
                                               bool                 STRIDED,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               hipsolverComplex*    A,
                                               int                  lda,
                                               int                  stA,
                                               float*               W,
                                               int                  stW,
                                               hipsolverComplex*    work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverCheevj(handle,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               W,
                               (hipFloatComplex*)work,
                               lwork,
                               info,
                               params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj(testAPI_t               API,
                                               bool                    STRIDED,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               double*                 W,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               hipsolverSyevjInfo_t    params,
                                               int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverZheevj(handle,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               W,
                               (hipDoubleComplex*)work,
                               lwork,
                               info,
                               params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYGVD/HEGVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
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
{

        return hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);

}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
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
{

        return hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);

}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   B,
                                                          int                 ldb,
                                                          float*              W,
                                                          int*                lwork)
{

        return hipsolverChegvd_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          W,
                                          lwork);

}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigType_t      itype,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* B,
                                                          int                     ldb,
                                                          double*                 W,
                                                          int*                    lwork)
{

        return hipsolverZhegvd_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          W,
                                          lwork);

}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              B,
                                               int                 ldb,
                                               int                 stB,
                                               float*              W,
                                               int                 stW,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             B,
                                               int                 ldb,
                                               int                 stB,
                                               double*             W,
                                               int                 stW,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               hipsolverComplex*   B,
                                               int                 ldb,
                                               int                 stB,
                                               float*              W,
                                               int                 stW,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverChegvd(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               W,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigType_t      itype,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               hipsolverDoubleComplex* B,
                                               int                     ldb,
                                               int                     stB,
                                               double*                 W,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{

        return hipsolverZhegvd(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               W,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** SYGVDX/HEGVDX ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sygvdx_hegvdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigType_t  itype,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            float*              A,
                                                            int                 lda,
                                                            float*              B,
                                                            int                 ldb,
                                                            float               vl,
                                                            float               vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            float*              W,
                                                            int*                lwork)
{
    switch(API)
    {
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigType_t  itype,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            double*             A,
                                                            int                 lda,
                                                            double*             B,
                                                            int                 ldb,
                                                            double              vl,
                                                            double              vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            double*             W,
                                                            int*                lwork)
{
    switch(API)
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigType_t  itype,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            hipsolverComplex*   A,
                                                            int                 lda,
                                                            hipsolverComplex*   B,
                                                            int                 ldb,
                                                            float               vl,
                                                            float               vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            float*              W,
                                                            int*                lwork)
{
    switch(API)
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx_bufferSize(testAPI_t               API,
                                                            hipsolverHandle_t       handle,
                                                            hipsolverEigType_t      itype,
                                                            hipsolverEigMode_t      jobz,
                                                            hipsolverEigRange_t     range,
                                                            hipsolverFillMode_t     uplo,
                                                            int                     n,
                                                            hipsolverDoubleComplex* A,
                                                            int                     lda,
                                                            hipsolverDoubleComplex* B,
                                                            int                     ldb,
                                                            double                  vl,
                                                            double                  vu,
                                                            int                     il,
                                                            int                     iu,
                                                            int*                    nev,
                                                            double*                 W,
                                                            int*                    lwork)
{
    switch(API)
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigType_t  itype,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 float*              A,
                                                 int                 lda,
                                                 int                 stA,
                                                 float*              B,
                                                 int                 ldb,
                                                 int                 stB,
                                                 float               vl,
                                                 float               vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 float*              W,
                                                 int                 stW,
                                                 float*              work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigType_t  itype,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 double*             A,
                                                 int                 lda,
                                                 int                 stA,
                                                 double*             B,
                                                 int                 ldb,
                                                 int                 stB,
                                                 double              vl,
                                                 double              vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 double*             W,
                                                 int                 stW,
                                                 double*             work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigType_t  itype,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 hipsolverComplex*   A,
                                                 int                 lda,
                                                 int                 stA,
                                                 hipsolverComplex*   B,
                                                 int                 ldb,
                                                 int                 stB,
                                                 float               vl,
                                                 float               vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 float*              W,
                                                 int                 stW,
                                                 hipsolverComplex*   work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx(testAPI_t               API,
                                                 hipsolverHandle_t       handle,
                                                 hipsolverEigType_t      itype,
                                                 hipsolverEigMode_t      jobz,
                                                 hipsolverEigRange_t     range,
                                                 hipsolverFillMode_t     uplo,
                                                 int                     n,
                                                 hipsolverDoubleComplex* A,
                                                 int                     lda,
                                                 int                     stA,
                                                 hipsolverDoubleComplex* B,
                                                 int                     ldb,
                                                 int                     stB,
                                                 double                  vl,
                                                 double                  vu,
                                                 int                     il,
                                                 int                     iu,
                                                 int*                    nev,
                                                 double*                 W,
                                                 int                     stW,
                                                 hipsolverDoubleComplex* work,
                                                 int                     lwork,
                                                 int*                    info,
                                                 int                     bc)
{
    switch(API)
    {

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYGVJ/HEGVJ ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sygvj_hegvj_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
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
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsygvj_bufferSize(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
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
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsygvj_bufferSize(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverEigType_t   itype,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          hipsolverComplex*    B,
                                                          int                  ldb,
                                                          float*               W,
                                                          int*                 lwork,
                                                          hipsolverSyevjInfo_t params)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChegvj_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          W,
                                          lwork,
                                          params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigType_t      itype,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* B,
                                                          int                     ldb,
                                                          double*                 W,
                                                          int*                    lwork,
                                                          hipsolverSyevjInfo_t    params)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhegvj_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          W,
                                          lwork,
                                          params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj(testAPI_t            API,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigType_t   itype,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               float*               A,
                                               int                  lda,
                                               int                  stA,
                                               float*               B,
                                               int                  ldb,
                                               int                  stB,
                                               float*               W,
                                               int                  stW,
                                               float*               work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsygvj(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj(testAPI_t            API,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigType_t   itype,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               double*              A,
                                               int                  lda,
                                               int                  stA,
                                               double*              B,
                                               int                  ldb,
                                               int                  stB,
                                               double*              W,
                                               int                  stW,
                                               double*              work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsygvj(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj(testAPI_t            API,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigType_t   itype,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               hipsolverComplex*    A,
                                               int                  lda,
                                               int                  stA,
                                               hipsolverComplex*    B,
                                               int                  ldb,
                                               int                  stB,
                                               float*               W,
                                               int                  stW,
                                               hipsolverComplex*    work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChegvj(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               W,
                               (hipFloatComplex*)work,
                               lwork,
                               info,
                               params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigType_t      itype,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               hipsolverDoubleComplex* B,
                                               int                     ldb,
                                               int                     stB,
                                               double*                 W,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               hipsolverSyevjInfo_t    params,
                                               int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhegvj(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               W,
                               (hipDoubleComplex*)work,
                               lwork,
                               info,
                               params);

    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYTRD/HETRD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          float*              tau,
                                                          int*                lwork)
{

        return hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);

}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             D,
                                                          double*             E,
                                                          double*             tau,
                                                          int*                lwork)
{

        return hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);

}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{

        return hipsolverChetrd_bufferSize(
            handle, uplo, n, (hipFloatComplex*)A, lda, D, E, (hipFloatComplex*)tau, lwork);

}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 D,
                                                          double*                 E,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{

        return hipsolverZhetrd_bufferSize(
            handle, uplo, n, (hipDoubleComplex*)A, lda, D, E, (hipDoubleComplex*)tau, lwork);

}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              D,
                                               int                 stD,
                                               float*              E,
                                               int                 stE,
                                               float*              tau,
                                               int                 stP,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             D,
                                               int                 stD,
                                               double*             E,
                                               int                 stE,
                                               double*             tau,
                                               int                 stP,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               float*              D,
                                               int                 stD,
                                               float*              E,
                                               int                 stE,
                                               hipsolverComplex*   tau,
                                               int                 stP,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{

        return hipsolverChetrd(handle,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               D,
                               E,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);

}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               double*                 D,
                                               int                     stD,
                                               double*                 E,
                                               int                     stE,
                                               hipsolverDoubleComplex* tau,
                                               int                     stP,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{

        return hipsolverZhetrd(handle,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               D,
                               E,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);

}
/********************************************************/

/******************** SYTRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int n, float* A, int lda, int* lwork)
{

        return hipsolverSsytrf_bufferSize(handle, n, A, lda, lwork);

}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int n, double* A, int lda, int* lwork)
{

        return hipsolverDsytrf_bufferSize(handle, n, A, lda, lwork);

}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int n, hipsolverComplex* A, int lda, int* lwork)
{

        return hipsolverCsytrf_bufferSize(handle, n, (hipFloatComplex*)A, lda, lwork);

}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
{

        return hipsolverZsytrf_bufferSize(handle, n, (hipDoubleComplex*)A, lda, lwork);

}

inline hipsolverStatus_t hipsolver_sytrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{

        return hipsolverSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_sytrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{

        return hipsolverDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info);

}

inline hipsolverStatus_t hipsolver_sytrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{

        return hipsolverCsytrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, ipiv, (hipFloatComplex*)work, lwork, info);

}

inline hipsolverStatus_t hipsolver_sytrf(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         int*                    ipiv,
                                         int                     stP,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{

        return hipsolverZsytrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)work, lwork, info);

}
/********************************************************/
