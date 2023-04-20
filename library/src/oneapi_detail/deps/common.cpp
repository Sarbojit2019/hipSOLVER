#include <iostream>
#include <ext/oneapi/backend/level_zero.hpp>
#include <include/ze_api.h>
#include <oneapi/mkl.hpp>
#include "common.h"

#include "sycl_solver.h"

struct syclHandle
{
    syclPlatform_t platform;
    syclDevice_t   device;
    syclContext_t  context;
    syclQueue_t    queue;
    hipStream_t    hip_stream; 

    syclHandle(void)
        : platform()
        , device()
        , context()
        , queue()
        , hip_stream()
    {
    }

    ~syclHandle()
    {
        // Fix_Me: CHIP owns LZ resources hecen need to find better way to release sycl resources here
        //syclQueueDestroy(queue);
        //syclContextDestroy(context);
        //syclDeviceDestroy(device);
        //syclPlatformDestroy(platform);
    }
};
// local functions
static int sycl_platform_create(syclPlatform_t *obj,
                                  ze_driver_handle_t driver) {
  auto sycl_platform = sycl::ext::oneapi::level_zero::make_platform((pi_native_handle) driver);
  *obj = new syclPlatform_st({sycl_platform});
  return 0;
}

static int sycl_platform_destroy(syclPlatform_t obj) {
  delete obj;
  return 0;
}

static int sycl_device_create(syclDevice_t *obj, syclPlatform_t platform,
                                ze_device_handle_t device) {
  auto sycl_device =
      sycl::ext::oneapi::level_zero::make_device(platform->val, (pi_native_handle) device);
  *obj = new syclDevice_st({sycl_device});
  return 0;
}

static int sycl_device_destroy(syclDevice_t obj) {
  delete obj;
  return 0;
}

static int sycl_context_create(syclContext_t *obj, syclDevice_t *devices,
                                 size_t ndevices, ze_context_handle_t context,
                                 int keep_ownership) {
  std::vector<sycl::device> sycl_devices(ndevices);
  for (size_t i = 0; i < ndevices; i++)
      sycl_devices[i] = devices[i]->val;

  auto sycl_context =
      sycl::ext::oneapi::level_zero::make_context(sycl_devices, (pi_native_handle) context, keep_ownership);
  *obj = new syclContext_st({sycl_context});
  return 0;
}

static int sycl_context_destroy(syclContext_t obj) {
  delete obj;
  return 0;
}

static int sycl_queue_create(syclQueue_t *obj, syclContext_t context, syclDevice_t device,
                               ze_command_queue_handle_t queue,
                               int keep_ownership) {
  // XXX: ownership argument only used on master
  auto sycl_queue = sycl::ext::oneapi::level_zero::make_queue(context->val, device->val, (pi_native_handle) queue, keep_ownership);
  *obj = new syclQueue_st({sycl_queue});
  return 0;
}

static int sycl_queue_destroy(syclQueue_t obj) {
  delete obj;
  return 0;
}

static int sycl_event_create(syclEvent_t *obj, syclContext_t context,
                               ze_event_handle_t event, int keep_ownership) {
  auto sycl_event = sycl::ext::oneapi::level_zero::make_event(context->val, (pi_native_handle) event, keep_ownership);
  *obj = new syclEvent_st({sycl_event});
  return 0;
}

static int sycl_event_destroy(syclEvent_t obj) {
  delete obj;
  return 0;
}

// Helper functions
hipsolverStatus_t sycl_create_handle(syclHandle_t* handle){
  if(handle != nullptr)
  {
    *handle = new syclHandle();
  }
  return (handle != nullptr) ? HIPSOLVER_STATUS_SUCCESS : HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
}

hipsolverStatus_t sycl_destroy_handle(syclHandle_t handle) {
  if(handle != nullptr)
  {
    delete handle;
  }
  return (handle != nullptr) ? HIPSOLVER_STATUS_SUCCESS : HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
}

hipsolverStatus_t sycl_set_hipstream(syclHandle_t handle,
                                  unsigned long const* nativeHandles,
                                  int                  nHandles,
                                   hipStream_t          stream,
                                   const char*          hipBackendName){
  assert(nHandles == 4);
  if(handle != nullptr)
  {
    handle->hip_stream = stream;
    std::string hipBackend(hipBackendName);
    if (hipBackend == "opencl") {
      // handle openCl case here
      cl_platform_id hPlatformId = (cl_platform_id)nativeHandles[0];
      cl_device_id hDeviceId = (cl_device_id)nativeHandles[1];
      cl_context hContext = (cl_context)nativeHandles[2];
      cl_command_queue hQueue = (cl_command_queue)nativeHandles[3];

      // platform
      auto sycl_platform = sycl::opencl::make_platform((pi_native_handle)hPlatformId);
      handle->platform = new syclPlatform_st({sycl_platform});
      // device
      auto sycl_device = sycl::opencl::make_device((pi_native_handle)hDeviceId);
      handle->device = new syclDevice_st({sycl_device});

      // context
      auto sycl_context = sycl::opencl::make_context((pi_native_handle)hContext);
      handle->context = new syclContext_st({sycl_context});

      // queue
      auto sycl_queue = sycl::opencl::make_queue(sycl_context, (pi_native_handle)hQueue);
      handle->queue = new syclQueue_st({sycl_queue});
    } else {
      // Obtain the handles to the LZ constructs.
      auto hDriver  = (ze_driver_handle_t)nativeHandles[0];
      auto hDevice  = (ze_device_handle_t)nativeHandles[1];
      auto hContext = (ze_context_handle_t)nativeHandles[2];
      auto hQueue   = (ze_command_queue_handle_t)nativeHandles[3];

      // Build SYCL platform/device/queue from the LZ handles.
      sycl_platform_create(&handle->platform, hDriver);
      sycl_device_create(&handle->device, handle->platform, hDevice);

      // FIX ME: only 1 device is returned from CHIP-SPV's lzHandles
      sycl_context_create(
        &handle->context, &handle->device, 1 /*ndevices*/, hContext, 1 /*keep_ownership*/);
      sycl_queue_create(&handle->queue, handle->context, handle->device, hQueue, 1 /* keep ownership */);
    }
  }
  return (handle != nullptr) ? HIPSOLVER_STATUS_SUCCESS : HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
}

hipsolverStatus_t sycl_get_hipstream(syclHandle_t handle, hipStream_t* pStream){
  if (handle == nullptr || pStream == nullptr) {
    return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
  }
  *pStream = handle->hip_stream;
  return HIPSOLVER_STATUS_SUCCESS;
}

syclQueue_t sycl_get_queue(syclHandle_t handle){
  if (handle == nullptr)
    return nullptr;

  return handle->queue;
}