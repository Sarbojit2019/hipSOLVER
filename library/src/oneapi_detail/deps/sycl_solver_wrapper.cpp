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

//getrs
  extern "C" int64_t onemkl_Sgetrs_ScPadSz(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::getrs_scratchpad_size<float>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH("Sgetrs_scratchpad")
  }
  extern "C" int64_t onemkl_Dgetrs_ScPadSz(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::getrs_scratchpad_size<double>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH("Dgetrs_scratchpad")
  }
  extern "C" int64_t onemkl_Cgetrs_ScPadSz(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<float>>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH("Cgetrs_scratchpad")
  }
  extern "C" int64_t onemkl_Zgetrs_ScPadSz(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<double>>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH("Zgetrs_scratchpad")
  }
  extern "C" void onemkl_Sgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float* A, int64_t lda, std::int64_t *ipiv,
              float* B, int64_t ldb, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, A, lda, ipiv, B, ldb, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sgetrs")
  }
  extern "C" void onemkl_Dgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double* A, int64_t lda, std::int64_t *ipiv,
              double* B, int64_t ldb, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, A, lda, ipiv, B, ldb, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dgetrs")
  }
  extern "C" void onemkl_Cgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex* A, int64_t lda, std::int64_t *ipiv,
              float _Complex* B, int64_t ldb, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<float>*>(A), lda, ipiv,
                  reinterpret_cast<std::complex<float>*>(B), ldb, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Cgetrs")
  }
  extern "C" void onemkl_Zgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex* A, int64_t lda, std::int64_t *ipiv,
              double _Complex* B, int64_t ldb, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<double>*>(A), lda, ipiv,
                  reinterpret_cast<std::complex<double>*>(B), ldb, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zgetrs")
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
  event.wait();
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

extern "C" int64_t onemkl_Dormqr_ScPadSz(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
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

  extern "C" void onemkl_Sormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
              float* A, int64_t lda, float* tua, float* C, int64_t ldc, float* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::ormqr(device_queue->val, convert(side), convert(trans), m, n, k, A, lda, tua, C, ldc, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sormqr")
  }
  extern "C" void onemkl_Dormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    double* A, int64_t lda, double* tua, double* C, int64_t ldc, double* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::ormqr(device_queue->val, convert(side), convert(trans), m, n, k, A, lda, tua, C, ldc, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dormqr")
  }
  extern "C" void onemkl_Cunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    float _Complex* A, int64_t lda, float _Complex* tua, float _Complex* C, int64_t ldc,
                    float _Complex* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::unmqr(device_queue->val, convert(side), convert(trans), m, n, k,
                            reinterpret_cast<std::complex<float>*>(A), lda, reinterpret_cast<std::complex<float>*>(tua),
                            reinterpret_cast<std::complex<float>*>(C), ldc, reinterpret_cast<std::complex<float>*>(scratchpad),
                            scratchpad_size);
    ONEMKL_CATCH("Cunmqr")
  }
  extern "C" void onemkl_Zunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k,
                    double _Complex* A, int64_t lda, double _Complex* tua, double _Complex* C, int64_t ldc,
                    double _Complex* scratchpad, int64_t scratchpad_size) {
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::unmqr(device_queue->val, convert(side), convert(trans), m, n, k,
                            reinterpret_cast<std::complex<double>*>(A), lda, reinterpret_cast<std::complex<double>*>(tua),
                            reinterpret_cast<std::complex<double>*>(C), ldc, reinterpret_cast<std::complex<double>*>(scratchpad),
                            scratchpad_size);
    ONEMKL_CATCH("Zunmqr")
  }

// ormtr/unmtr
  extern "C" int64_t onemkl_Sormtr_ScPadSz(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::ormtr_scratchpad_size<float>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return size;
    ONEMKL_CATCH("Sormtr_ScPadSz")
  }
  extern "C" int64_t onemkl_Dormtr_ScPadSz(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::ormtr_scratchpad_size<double>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return size;
    ONEMKL_CATCH("Dormtr_ScPadSz")
  }
  extern "C" void onemkl_Sormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float* A, int64_t lda,
              float* tua, float* C, int64_t ldc, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::ormtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, A, lda, tua, C, ldc,
                                             scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sormtr")
  }
  extern "C" void onemkl_Dormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double* A, int64_t lda,
              double* tua, double* C, int64_t ldc, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::ormtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, A, lda, tua, C, ldc,
                                             scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dormtr")
  }
  extern "C" int64_t onemkl_Cunmtr_ScPadSz(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<float>>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return size;
    ONEMKL_CATCH("Cunmtr_ScPadSz")
  }
  extern "C" int64_t onemkl_Zunmtr_ScPadSz(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n,
                    int64_t lda, int64_t ldc) {
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<double>>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
    return size;
    ONEMKL_CATCH("Zunmtr_ScPadSz")
  }
  extern "C" void onemkl_Cunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float _Complex* A, int64_t lda,
              float _Complex* tua, float _Complex* C, int64_t ldc, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::unmtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n,
                                            reinterpret_cast<std::complex<float>*>(A), lda, reinterpret_cast<std::complex<float>*>(tua),
                                            reinterpret_cast<std::complex<float>*>(C), ldc, reinterpret_cast<std::complex<float>*>(scratchpad),
                                            scratchpad_size);
    ONEMKL_CATCH("Cunmtr")
  }
  extern "C" void onemkl_Zunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double _Complex* A, int64_t lda,
              double _Complex* tua, double _Complex* C, int64_t ldc, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::unmtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n,
                                            reinterpret_cast<std::complex<double>*>(A), lda, reinterpret_cast<std::complex<double>*>(tua),
                                            reinterpret_cast<std::complex<double>*>(C), ldc, reinterpret_cast<std::complex<double>*>(scratchpad),
                                            scratchpad_size);
    ONEMKL_CATCH("Zunmtr")
  }

  //getrf
  extern "C" int64_t onemkl_Sgetrf_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::getrf_scratchpad_size<float>(device_queue->val, m, n, lda);
    return size;
    ONEMKL_CATCH("Sgetrf_scratchpad")
  }
  extern "C" int64_t onemkl_Dgetrf_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(device_queue->val, m, n, lda);
    return size;
    ONEMKL_CATCH("Dgetrf_scratchpad")
  }
  extern "C" int64_t onemkl_Cgetrf_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
    return size;
    ONEMKL_CATCH("Cgetrf_scratchpad")
  }
  extern "C" int64_t onemkl_Zgetrf_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
    return size;
    ONEMKL_CATCH("Zgetrf_scratchpad")
  }
  extern "C" void onemkl_Sgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float* A, int64_t lda, int64_t *ipiv, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, A, lda, ipiv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Sgetrf")
  }
  extern "C" void onemkl_Dgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double* A, int64_t lda, int64_t *ipiv, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, A, lda, ipiv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dgetrf")
  }
  extern "C" void onemkl_Cgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex* A, int64_t lda, int64_t *ipiv, float _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(A), lda, ipiv,
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Cgetrf")
  }
  extern "C" void onemkl_Zgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex* A, int64_t lda, int64_t *ipiv, double _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(A), lda, ipiv,
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Zgetrf")
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
  event.wait();
  ONEMKL_CATCH("Sgebrd")
}
extern "C" void onemkl_Dgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double* a, int64_t lda,
                   double* d, double* e, double* tauq, double* taup, double* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup,
                scratchpad, scratchpad_size);
  event.wait();
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
  event.wait();
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
  event.wait();
  ONEMKL_CATCH("Zgebrd")
}

  //geqrf
  extern "C" int64_t onemkl_Sgeqrf_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::geqrf_scratchpad_size<float>(device_queue->val, m, n, lda);
    return size;
    ONEMKL_CATCH("Sgeqrf_scratchpad")
  }
  extern "C" int64_t onemkl_Dgeqrf_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::geqrf_scratchpad_size<double>(device_queue->val, m, n, lda);
    return size;
    ONEMKL_CATCH("Dgeqrf_scratchpad")
  }
  extern "C" int64_t onemkl_Cgeqrf_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
    return size;
    ONEMKL_CATCH("Cgeqrf_scratchpad")
  }
  extern "C" int64_t onemkl_Zgeqrf_ScPadSz(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
    return size;
    ONEMKL_CATCH("Zgeqrf_scratchpad")
  }
  extern "C" void onemkl_Sgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float* A, int64_t lda, float* tua, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, A, lda, tua, scratchpad, scratchpad_size);

    ONEMKL_CATCH("Sgeqrf")
  }
  extern "C" void onemkl_Dgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double* A, int64_t lda, double* tua, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, A, lda, tua, scratchpad, scratchpad_size);

    ONEMKL_CATCH("Dgeqrf")
  }
extern "C"   void onemkl_Cgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex* A, int64_t lda, float _Complex* tua, float _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(A), lda,
                                             reinterpret_cast<std::complex<float>*>(tua),
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Cgeqrf")
  }
extern "C" void onemkl_Zgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex* A, int64_t lda, double _Complex* tua, double _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(A), lda,
                                             reinterpret_cast<std::complex<double>*>(tua),
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Zgeqrf")
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
  event.wait();
  ONEMKL_CATCH("Ssyevd")
}

extern "C" void onemkl_Dsyevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* w,
                   double* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::syevd(device_queue->val, convert(job), convert(uplo), n, A, lda, w, scratchpad, scratchpad_size);
  event.wait();
  ONEMKL_CATCH("Dsyevd")
}

extern "C" int64_t onemkl_Cheevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
ONEMKL_TRY()
auto size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<float>>(device_queue->val, convert(job), convert(uplo), n, lda);
return size;
ONEMKL_CATCH("Cheevd_scratchpad")
}

extern "C" int64_t onemkl_Zheevd_ScPadSz(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda) {
ONEMKL_TRY()
auto size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<double>>(device_queue->val, convert(job), convert(uplo), n, lda);
return size;
ONEMKL_CATCH("Zheevd_scratchpad")
}

extern "C" void onemkl_Cheevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float* w,
                   float _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::heevd(device_queue->val, convert(job), convert(uplo), n,
                  reinterpret_cast<std::complex<float>*>(A), lda, w, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
  event.wait();
  ONEMKL_CATCH("Cheevd")
}

extern "C" void onemkl_Zheevd(syclQueue_t device_queue, onemklJob job, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double* w,
                  double _Complex* scratchpad, int64_t scratchpad_size){
  ONEMKL_TRY()
  auto event = oneapi::mkl::lapack::heevd(device_queue->val, convert(job), convert(uplo), n,
                  reinterpret_cast<std::complex<double>*>(A), lda, w, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
  event.wait();
  ONEMKL_CATCH("Zheevd")
}

//potrf
  extern "C" int64_t onemkl_Spotrf_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potrf_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Spotrf_ScPadSz")
  }
  extern "C" int64_t onemkl_Dpotrf_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potrf_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Dpotrf_ScPadSz")
  }
  extern "C" int64_t onemkl_Cpotrf_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Cpotrf_ScPadSz")
  }
  extern "C" int64_t onemkl_Zpotrf_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Zpotrf_ScPadSz")
  }
  extern "C" void onemkl_Spotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, A, lda, scratchpad, scratchpad_size);

    ONEMKL_CATCH("Spotrf")
  }
  extern "C" void onemkl_Dpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, A, lda, scratchpad, scratchpad_size);

    ONEMKL_CATCH("Dpotrf")
  }
  extern "C" void onemkl_Cpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda,
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Cpotrf")
  }
  extern "C" void onemkl_Zpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda,
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Zpotrf")
  }

//potri
  extern "C" int64_t onemkl_Spotri_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potri_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Spotri_ScPadSz")
  }
  extern "C" int64_t onemkl_Dpotri_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potri_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Dpotri_ScPadSz")
  }
  extern "C" int64_t onemkl_Cpotri_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Cpotri_ScPadSz")
  }
  extern "C" int64_t onemkl_Zpotri_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Zpotri_ScPadSz")
  }
  extern "C" void onemkl_Spotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, A, lda, scratchpad, scratchpad_size);

    ONEMKL_CATCH("Spotri")
  }
  extern "C" void onemkl_Dpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, A, lda, scratchpad, scratchpad_size);

    ONEMKL_CATCH("Dpotri")
  }
  extern "C" void onemkl_Cpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda,
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Cpotri")
  }
  extern "C" void onemkl_Zpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda,
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Zpotri")
  }

  //potrs
  // Note: Wait needs to be added else gtests will fail as this is async API call
  extern "C" int64_t onemkl_Spotrs_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potrs_scratchpad_size<float>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH("Spotrs_scratchpad")
  }
  extern "C" int64_t onemkl_Dpotrs_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potrs_scratchpad_size<double>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH("Dpotrs_scratchpad")
  }
  extern "C" int64_t onemkl_Cpotrs_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH("Cpotrs_scratchpad")
  }
  extern "C" int64_t onemkl_Zpotrs_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
    return size;
    ONEMKL_CATCH("Zpotrs_scratchpad")
  }
  extern "C" void onemkl_Spotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float* A, int64_t lda,
              float* B, int64_t ldb, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, A, lda, B, ldb, scratchpad, scratchpad_size);
    status.wait();
    ONEMKL_CATCH("Spotrs")
  }
  extern "C" void onemkl_Dpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double* A, int64_t lda,
              double* B, int64_t ldb, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, A, lda, B, ldb, scratchpad, scratchpad_size);
    status.wait();
    ONEMKL_CATCH("Dpotrs")
  }
  extern "C" void onemkl_Cpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex* A, int64_t lda,
              float _Complex* B, int64_t ldb, float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<float>*>(A), lda,
                  reinterpret_cast<std::complex<float>*>(B), ldb, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    status.wait();
    ONEMKL_CATCH("Cpotrs")
  }
  extern "C" void onemkl_Zpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex* A, int64_t lda,
              double _Complex* B, int64_t ldb, double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<double>*>(A), lda,
                  reinterpret_cast<std::complex<double>*>(B), ldb, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    status.wait();
    ONEMKL_CATCH("Zpotrs")
  }

  //sytrd/hetrd
  extern "C" int64_t onemkl_Ssytrd_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::sytrd_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Ssytrd_ScPadSz")
  }
  extern "C" int64_t onemkl_Dsytrd_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::sytrd_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Dsytrd_ScPadSz")
  }
  extern "C" int64_t onemkl_Chetrd_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Chetrd_ScPadSz")
  }
  extern "C" int64_t onemkl_Zhetrd_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Dhetrd_ScPadSz")
  }
  extern "C" void onemkl_Ssytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* d, float* e, float* tau,
             float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::sytrd(device_queue->val, convert(uplo), n, A, lda, d, e, tau, scratchpad, scratchpad_size);

    ONEMKL_CATCH("Ssytrf")
  }
  extern "C" void onemkl_Dsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* d, double* e, double* tau,
             double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::sytrd(device_queue->val, convert(uplo), n, A, lda, d, e, tau, scratchpad, scratchpad_size);

    ONEMKL_CATCH("Dsytrf")
  }
  extern "C" void onemkl_Chetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float* d, float* e, float _Complex* tau,
             float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::hetrd(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda, d, e,
                                             reinterpret_cast<std::complex<float>*>(tau),
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Chetrd")
  }
  extern "C" void onemkl_Zhetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double* d, double* e, double _Complex* tau,
             double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::hetrd(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda, d, e,
                                             reinterpret_cast<std::complex<double>*>(tau),
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);

    ONEMKL_CATCH("Zhetrd")
  }

  //sytrf
  extern "C" int64_t onemkl_Ssytrf_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::sytrf_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Ssytrf_ScPadSz")
  }
  extern "C" int64_t onemkl_Dsytrf_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::sytrf_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Dsytrf_ScPadSz")
  }
  extern "C" int64_t onemkl_Csytrf_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Csytrf_ScPadSz")
  }
  extern "C" int64_t onemkl_Zsytrf_ScPadSz(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
    return size;
    ONEMKL_CATCH("Zsytrf_ScPadSz")
  }
  extern "C" void onemkl_Ssytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float* A, int64_t lda, int64_t* ipiv, float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, A, lda, ipiv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Ssytrf")
  }
  extern "C" void onemkl_Dsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double* A, int64_t lda, int64_t* ipiv, double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, A, lda, ipiv, scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dsytrf")
  }
  extern "C" void onemkl_Csytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, int64_t* ipiv, float _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda, ipiv,
                                            reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Csytrf")
  }
  extern "C" void onemkl_Zsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, int64_t* ipiv, double _Complex* scratchpad,
             int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda, ipiv,
                                            reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zsytrf")
  }

  // sygvd/hegvd
  extern "C" int64_t onemkl_Ssygvd_ScPadSz(syclQueue_t device_queue, int64_t itype, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(device_queue->val, itype, convert(job), convert(uplo), n, lda, ldb);
    return size;
    ONEMKL_CATCH("Ssygvd_ScPadSz")
  }
  extern "C" int64_t onemkl_Dsygvd_ScPadSz(syclQueue_t device_queue, int64_t itype, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(device_queue->val, itype, convert(job), convert(uplo), n, lda, ldb);
    return size;
    ONEMKL_CATCH("Dsygvd_ScPadSz")
  }
  extern "C" int64_t onemkl_Chegvd_ScPadSz(syclQueue_t device_queue, int64_t itype, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>(device_queue->val, itype, convert(job), convert(uplo), n, lda, ldb);
    return size;
    ONEMKL_CATCH("Chegvd_ScPadSz")
  }
  extern "C" int64_t onemkl_Zhegvd_ScPadSz(syclQueue_t device_queue, int64_t itype, onemklJob job, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb){
    ONEMKL_TRY()
    auto size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>(device_queue->val, itype, convert(job), convert(uplo), n, lda, ldb);
    return size;
    ONEMKL_CATCH("Zhegvd_ScPadSz")
  }
  extern "C" void onemkl_Ssygvd(syclQueue_t device_queue, int64_t itype, onemklJob job, onemklUplo uplo, int64_t n, float* A, int64_t lda, float* B, int64_t ldb, float* W,
              float* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::sygvd(device_queue->val, itype, convert(job), convert(uplo), n, A, lda, B, ldb, W,
                                            scratchpad, scratchpad_size);
    ONEMKL_CATCH("Ssygvd")
  }
  extern "C" void onemkl_Dsygvd(syclQueue_t device_queue, int64_t itype, onemklJob job, onemklUplo uplo, int64_t n, double* A, int64_t lda, double* B, int64_t ldb, double* W,
              double* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::sygvd(device_queue->val, itype, convert(job), convert(uplo), n, A, lda, B, ldb, W,
                                            scratchpad, scratchpad_size);
    ONEMKL_CATCH("Dsygvd")
  }
  extern "C" void onemkl_Chegvd(syclQueue_t device_queue, int64_t itype, onemklJob job, onemklUplo uplo, int64_t n, float _Complex* A, int64_t lda, float _Complex* B, int64_t ldb, float* W,
              float _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::hegvd(device_queue->val, itype, convert(job), convert(uplo), n, reinterpret_cast<std::complex<float>*>(A), lda,
                                             reinterpret_cast<std::complex<float>*>(B), ldb, W,
                                             reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Chegvd")
  }
  extern "C" void onemkl_Zhegvd(syclQueue_t device_queue, int64_t itype, onemklJob job, onemklUplo uplo, int64_t n, double _Complex* A, int64_t lda, double _Complex* B, int64_t ldb, double* W,
              double _Complex* scratchpad, int64_t scratchpad_size){
    ONEMKL_TRY()
    auto event = oneapi::mkl::lapack::hegvd(device_queue->val, itype, convert(job), convert(uplo), n, reinterpret_cast<std::complex<double>*>(A), lda,
                                             reinterpret_cast<std::complex<double>*>(B), ldb, W,
                                             reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
    ONEMKL_CATCH("Zhegvd")
  }