#pragma once

#include "taskSystem/shared/hostDeviceCompilation.h"

#ifdef DEVICE_COMPILATION
typedef half GemvTaskElement;
#else
#include <CL/cl_half.h>
typedef cl_half GemvTaskElement;
#endif