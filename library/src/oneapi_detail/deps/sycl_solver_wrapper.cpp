#include "common.h"
#include <oneapi/mkl.hpp>
#include "sycl_solver.h"

// local functions
oneapi::mkl::jobsvd convert(signed char j) {
  switch(j) {
    case 'N': return oneapi::mkl::jobsvd::N;
    case 'A': return oneapi::mkl::jobsvd::A;
    case 'S': return oneapi::mkl::jobsvd::S;
    case 'O': return oneapi::mkl::jobsvd::O;
    default : return oneapi::mkl::jobsvd::N; // need to test
  }
}

oneapi::mkl::job convert(onemklJob j) {
  switch(j) {
    case ONEMKL_JOB_NOVEC: return oneapi::mkl::job::novec;
    case ONEMKL_JOB_VEC: return oneapi::mkl::job::vec;
  }
}

oneapi::mkl::uplo convert(onemklUplo ul) {
  switch(ul) {
    case ONEMKL_UPLO_LOWER: return oneapi::mkl::uplo::lower;
    case ONEMKL_UPLO_UPPER: return oneapi::mkl::uplo::upper;
  }
}

oneapi::mkl::generate convert(onemklGen g) {
  switch(g) {
    case ONEMKL_GEN_Q: return oneapi::mkl::generate::q;
    case ONEMKL_GEN_P: return oneapi::mkl::generate::p;
  }
}

oneapi::mkl::side convert(onemklSide s) {
  switch(s){
    case ONEMKL_SIDE_LEFT: return oneapi::mkl::side::left;
    case ONEMKL_SIDE_RIGHT: return oneapi::mkl::side::right;
  }
}

oneapi::mkl::transpose convert(onemklTranspose val) {
    switch (val) {
    case ONEMKL_TRANSPOSE_NONTRANS:
        return oneapi::mkl::transpose::nontrans;
    case ONEMKL_TRANSPOSE_TRANS:
        return oneapi::mkl::transpose::trans;
    case ONEMLK_TRANSPOSE_CONJTRANS:
        return oneapi::mkl::transpose::conjtrans;
    }
}

extern "C" int64_t onemkl_Sorgbr_ScPadSz(syclQueue_t device_queue,onemklGen gen, int64_t m, int64_t n, int64_t k,
                             int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::orgbr_scratchpad_size<float>(device_queue->val, convert(gen), m, n, k, lda);
  return size;
  ONEMKL_CATCH("Sorgbr_ScPadSz")
}

extern "C" int64_t onemkl_Dorgbr_ScPadSz(syclQueue_t device_queue,onemklGen gen, int64_t m, int64_t n, int64_t k,
                             int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::orgbr_scratchpad_size<double>(device_queue->val, convert(gen), m, n, k, lda);
  return size;
  ONEMKL_CATCH("Dorgbr_ScPadSz")
}

extern "C" void onemkl_Sorgbr(syclQueue_t device_queue, onemklGen gen, int64_t m, int64_t n, int64_t k, float* A, int64_t lda,
                   float* tua, float* scratchpad, int64_t scratchpad_size) {
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::orgbr(device_queue->val, convert(gen), m, n, k, A, lda, tua, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Sorgbr")
}

extern "C" void onemkl_Dorgbr(syclQueue_t device_queue, onemklGen gen, int64_t m, int64_t n, int64_t k, double* A, int64_t lda,
                   double* tua, double* scratchpad, int64_t scratchpad_size) {
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::orgbr(device_queue->val, convert(gen), m, n, k, A, lda, tua, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Dorgbr")
}

extern "C" int64_t onemkl_Cungbr_ScPadSz(syclQueue_t device_queue,onemklGen gen, int64_t m, int64_t n, int64_t k,
                             int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<float>>(device_queue->val, convert(gen), m, n, k, lda);
  return size;
  ONEMKL_CATCH("Cungbr_ScPadSz")
}

extern "C" int64_t onemkl_Zungbr_ScPadSz(syclQueue_t device_queue,onemklGen gen, int64_t m, int64_t n, int64_t k,
                             int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<double>>(device_queue->val, convert(gen), m, n, k, lda);
  return size;
  ONEMKL_CATCH("Zungbr_ScPadSz")
}

extern "C" void onemkl_Cungbr(syclQueue_t device_queue, onemklGen gen, int64_t m, int64_t n, int64_t k, float _Complex* A, int64_t lda,
                   float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::ungbr(device_queue->val, convert(gen), m, n, k, reinterpret_cast<std::complex<float>*>(A), lda,
                                         reinterpret_cast<std::complex<float>*>(tua), reinterpret_cast<std::complex<float>*>(scratchpad),
                                         scratchpad_size);
  ONEMKL_CATCH("Cungbr")
}
extern "C" void onemkl_Zungbr(syclQueue_t device_queue, onemklGen gen, int64_t m, int64_t n, int64_t k, double _Complex* A, int64_t lda,
                   double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::ungbr(device_queue->val, convert(gen), m, n, k, reinterpret_cast<std::complex<double>*>(A), lda,
                                         reinterpret_cast<std::complex<double>*>(tua), reinterpret_cast<std::complex<double>*>(scratchpad),
                                         scratchpad_size);
  ONEMKL_CATCH("Zungbr")
}

extern "C" int64_t onemkl_Sorgqr_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::orgqr_scratchpad_size<float>(device_queue->val, m, n, k, lda);
  return size;
  ONEMKL_CATCH("Sorgqr_ScPadSz")
}

extern "C" int64_t onemkl_Dorgqr_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::orgqr_scratchpad_size<double>(device_queue->val, m, n, k, lda);
  return size;
  ONEMKL_CATCH("Dorgqr_ScPadSz")
}

extern "C" void onemkl_Sorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float* A, int64_t lda,
                   float* tua, float* scratchpad, int64_t scratchpad_size) {
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::orgqr(device_queue->val, m, n, k, A, lda, tua, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Sorgqr")
}

extern "C" void onemkl_Dorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double* A, int64_t lda,
                   double* tua, double* scratchpad, int64_t scratchpad_size) {
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::orgqr(device_queue->val, m, n, k, A, lda, tua, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Dorgqr")
}

extern "C" int64_t onemkl_Cungqr_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<float>>(device_queue->val, m, n, k, lda);
  return size;
  ONEMKL_CATCH("Cungqr_ScPadSz")
}

extern "C" int64_t onemkl_Zungqr_ScPadSz(syclQueue_t device_queue,int64_t m, int64_t n, int64_t k, int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<double>>(device_queue->val, m, n, k, lda);
  return size;
  ONEMKL_CATCH("Zungqr_ScPadSz")
}

extern "C" void onemkl_Cungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex* A, int64_t lda,
                   float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::ungqr(device_queue->val, m, n, k, reinterpret_cast<std::complex<float>*>(A), lda,
                                         reinterpret_cast<std::complex<float>*>(tua), reinterpret_cast<std::complex<float>*>(scratchpad),
                                         scratchpad_size);
  ONEMKL_CATCH("Cungqr")
}
extern "C" void onemkl_Zungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex* A, int64_t lda,
                   double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::ungqr(device_queue->val, m, n, k, reinterpret_cast<std::complex<double>*>(A), lda,
                                         reinterpret_cast<std::complex<double>*>(tua), reinterpret_cast<std::complex<double>*>(scratchpad),
                                         scratchpad_size);
  ONEMKL_CATCH("Zungqr")
}

extern "C" int64_t onemkl_Sorgtr_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::orgtr_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
  return size;
  ONEMKL_CATCH("Sorgtr_ScPadSz")
}

extern "C" int64_t onemkl_Dorgtr_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::orgtr_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
  return size;
  ONEMKL_CATCH("Dorgtr_ScPadSz")
}

extern "C" void onemkl_Sorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float* A, int64_t lda,
                   float* tua, float* scratchpad, int64_t scratchpad_size) {
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::orgtr(device_queue->val, convert(uplo), n, A, lda, tua, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Sorgtr")
}

extern "C" void onemkl_Dorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double* A, int64_t lda,
                   double* tua, double* scratchpad, int64_t scratchpad_size) {
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::orgtr(device_queue->val, convert(uplo), n, A, lda, tua, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Dorgtr")
}

extern "C" int64_t onemkl_Cungtr_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
  return size;
  ONEMKL_CATCH("Cungtr_ScPadSz")
}

extern "C" int64_t onemkl_Zungtr_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
  return size;
  ONEMKL_CATCH("Zungtr_ScPadSz")
}

extern "C" void onemkl_Cungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda,
                   float _Complex* tua, float _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::ungtr(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda,
                                         reinterpret_cast<std::complex<float>*>(tua), reinterpret_cast<std::complex<float>*>(scratchpad),
                                         scratchpad_size);
  ONEMKL_CATCH("Cungtr")
}
extern "C" void onemkl_Zungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda,
                   double _Complex* tua, double _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::ungtr(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda,
                                         reinterpret_cast<std::complex<double>*>(tua), reinterpret_cast<std::complex<double>*>(scratchpad),
                                         scratchpad_size);
  ONEMKL_CATCH("Zungtr")
}

extern "C" int64_t onemkl_Sormqr_ScPadSz(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                   int64_t lda, int64_t ldc) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::ormqr_scratchpad_size<float>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
  return size;
  ONEMKL_CATCH("Sormqr_ScPadSz")
}

extern "C" int64_t onemkl_Sormqr_DcPadSz(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                   int64_t lda, int64_t ldc) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::ormqr_scratchpad_size<double>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
  return size;
  ONEMKL_CATCH("Dormqr_ScPadSz")
}

extern "C" int64_t onemkl_Cunmqr_ScPadSz(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                   int64_t lda, int64_t ldc) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<float>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
  return size;
  ONEMKL_CATCH("Cunmqr_ScPadSz")
}

extern "C" int64_t onemkl_Zunmqr_ScPadSz(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                   int64_t lda, int64_t ldc) {
  ONEMKL_TRY()
  auto size = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<double>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
  return size;
  ONEMKL_CATCH("Zunmqr_ScPadSz")
}



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

extern "C" int64_t onemkl_Ssyevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
ONEMKL_TRY()
auto size = oneapi::mkl::lapack::syevd_scratchpad_size<float>(device_queue->val, convert(job), convert(uplo), n, lda);
return size;
ONEMKL_CATCH("Ssyevd_scratchpad")
}

extern "C" int64_t onemkl_Dsyevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
ONEMKL_TRY()
auto size = oneapi::mkl::lapack::syevd_scratchpad_size<double>(device_queue->val, convert(job), convert(uplo), n, lda);
return size;
ONEMKL_CATCH("Dsyevd_scratchpad")
}

extern "C" void onemkl_Ssyevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* w,
                   float* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::syevd(device_queue->val, convert(job), convert(uplo), n, A, lda, w, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Ssyevd")
}

extern "C" void onemkl_Dsyevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* w,
                   double* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::syevd(device_queue->val, convert(job), convert(uplo), n, A, lda, w, scratchpad, scratchpad_size);
  ONEMKL_CATCH("Dsyevd")
}

extern "C" int64_t onemkl_Cheevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
ONEMKL_TRY()
auto size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<float>>(device_queue->val, convert(job), convert(uplo), n, lda);
return size;
ONEMKL_CATCH("Csyevd_scratchpad")
}

extern "C" int64_t onemkl_Zheevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
ONEMKL_TRY()
auto size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<double>>(device_queue->val, convert(job), convert(uplo), n, lda);
return size;
ONEMKL_CATCH("Zsyevd_scratchpad")
}

extern "C" void onemkl_Cheevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float* w,
                   float _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::heevd(device_queue->val, convert(job), convert(uplo), n,
                  reinterpret_cast<std::complex<float>*>(A), lda, w, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
  ONEMKL_CATCH("Cheevd")
}

extern "C" void onemkl_Zheevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double* w,
                  double _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::heevd(device_queue->val, convert(job), convert(uplo), n,
                  reinterpret_cast<std::complex<double>*>(A), lda, w, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
  ONEMKL_CATCH("Zheevd")
}