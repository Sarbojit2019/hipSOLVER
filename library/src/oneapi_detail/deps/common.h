#pragma once

#include <CL/sycl.hpp>

#define ONEMKL_TRY() try {
#define ONEMKL_CATCH(msg) \
            } catch(sycl::exception const& e) {\
              std::cerr << msg<<" SYCL exception: " << e.what() << std::endl;\
              throw;}\
              catch(std::exception const& e){\
              std::cerr << msg<<" exception: " << e.what() << std::endl;\
              throw;}

struct syclPlatform_st
{
    sycl::platform val;
};

struct syclDevice_st
{
    sycl::device val;
};

struct syclContext_st
{
    sycl::context val;
};

struct syclQueue_st
{
    sycl::queue val;
};

struct syclEvent_st
{
    sycl::event val;
};

