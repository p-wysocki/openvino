#pragma once

#include "taskSystem/shared/hostDeviceCompilation.h"

#ifdef DEVICE_COMPILATION
#include "common/semaphore.hcl"
typedef half GemvTaskElement;
#else
#include <CL/cl_half.h>
typedef cl_half GemvTaskElement;
#endif