#include "common.h"
#include <oneapi/mkl.hpp>
#include "sycl_solver.h"

extern "C" int64_t onemkl_Sgebrd_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<float>(device_queue->val, m, n, lda);
  return size;
  ONEMKL_CATCH("Sgebrd_scratchpad")
}

extern "C" int64_t onemkl_Dgebrd_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<double>(device_queue->val, m, n, lda);
  return size;
  ONEMKL_CATCH("Dgebrd_scratchpad")
}

extern "C" int64_t onemkl_Cgebrd_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
  return size;
  ONEMKL_CATCH("Cgebrd_scratchpad")
}
extern "C" int64_t onemkl_Zgebrd_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
  return size;
  ONEMKL_CATCH("Zgebrd_scratchpad")
}

extern "C" void onemkl_Sgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float* a, int64_t lda,
                   float* d, float* e, float* tauq, float* taup, float* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup,
                scratchpad, scratchpad_size);
  ONEMKL_CATCH("Sgebrd")
}
extern "C" void onemkl_Dgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double* a, int64_t lda,
                   double* d, double* e, double* tauq, double* taup, double* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup,
                scratchpad, scratchpad_size);
  ONEMKL_CATCH("Dgebrd")
}
extern "C" void onemkl_Cgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex* a, int64_t lda,
                   float * d, float* e, float _Complex* tauq, float _Complex* taup, float _Complex* scratchpad,
                   int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda,
                                         d, e, reinterpret_cast<std::complex<float>*>(tauq),
                                         reinterpret_cast<std::complex<float>*>(taup),
                                         reinterpret_cast<std::complex<float>*>(scratchpad),
                                         scratchpad_size);
  ONEMKL_CATCH("Cgebrd")
}
extern "C" void onemkl_Zgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex* a, int64_t lda,
                   double * d, double* e, double _Complex* tauq, double _Complex* taup, double _Complex* scratchpad,
                   int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda,
                                         d, e, reinterpret_cast<std::complex<double>*>(tauq),
                                         reinterpret_cast<std::complex<double>*>(taup),
                                         reinterpret_cast<std::complex<double>*>(scratchpad),
                                         scratchpad_size);
  ONEMKL_CATCH("Zgebrd")
}

oneapi::mkl::jobsvd convert(signed char j) {
  switch(j) {
    case 'N': return oneapi::mkl::jobsvd::N;
    case 'A': return oneapi::mkl::jobsvd::A;
    case 'S': return oneapi::mkl::jobsvd::S;
    case 'O': return oneapi::mkl::jobsvd::O;
    default : return oneapi::mkl::jobsvd::N; // need to test
  }
}

extern "C" int64_t onemkl_Sgesvd_ScPadSz(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                             int64_t lda, int64_t ldu, int64_t ldvt){
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::gesvd_scratchpad_size<float>(device_queue->val, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
  return size;
  ONEMKL_CATCH("Sgesvd_scratchpad")
}
extern "C" int64_t onemkl_Dgesvd_ScPadSz(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                             int64_t lda, int64_t ldu, int64_t ldvt){
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(device_queue->val, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
  return size;
  ONEMKL_CATCH("Dgesvd_scratchpad")
}
extern "C" int64_t onemkl_Cgesvd_ScPadSz(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                             int64_t lda, int64_t ldu, int64_t ldvt){
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(device_queue->val, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
  return size;
  ONEMKL_CATCH("Cgesvd_scratchpad")
}
extern "C" int64_t onemkl_Zgesvd_ScPadSz(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n,
                             int64_t lda, int64_t ldu, int64_t ldvt){
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(device_queue->val, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
  return size;
  ONEMKL_CATCH("Zgesvd_scratchpad")
}
extern "C" void onemkl_Sgesvd(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n, float* A, int64_t lda,
                    float* S, float* U, int64_t ldu, float* V, int64_t ldv, float* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, A, lda, S, U, ldu, V, ldv, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Sgesvd")
}
extern "C" void onemkl_Dgesvd(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n, double* A, int64_t lda,
                    double* S, double* U, int64_t ldu, double* V, int64_t ldv, double* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, A, lda, S, U, ldu, V, ldv, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Dgesvd")
}
extern "C" void onemkl_Cgesvd(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n, float _Complex* A, int64_t lda,
                    float* S, float _Complex* U, int64_t ldu, float _Complex* V, int64_t ldv, float _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, reinterpret_cast<std::complex<float>*>(A), lda,
                                         S, reinterpret_cast<std::complex<float>*>(U), ldu, reinterpret_cast<std::complex<float>*>(V), ldv,
                                         reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
  ONEMKL_CATCH("Cgesvd")
}
extern "C" void onemkl_Zgesvd(syclQueue_t device_queue, signed char jobu, signed char jobvt, int64_t m, int64_t n, double _Complex* A, int64_t lda,
                    double* S, double _Complex* U, int64_t ldu, double _Complex* V, int64_t ldv, double _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, reinterpret_cast<std::complex<double>*>(A), lda,
                                         S, reinterpret_cast<std::complex<double>*>(U), ldu, reinterpret_cast<std::complex<double>*>(V), ldv,
                                         reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
  ONEMKL_CATCH("Zgesvd")
}