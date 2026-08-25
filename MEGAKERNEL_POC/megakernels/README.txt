Folder contains POC impl of megakernel implemenations for openvino.

Currrently the idea is that each megakernel will be delivered as a stand alone library,
containing kernel plus runtime implementation.

The openvino engine has to link against that library.

Each implementation is backed by header-only, interface lib megakernel runtime, which defines 
interface that openvino engien will link against. 

All megakernels implementations has to export functions:
-> extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();
-> extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime);
which has to be defined in file megakernelImpl.h

which creates runtime for given megakernel.

The created library has to be shared lib with the same name as the main folder.

Each megakernel implementationt has to define:
class ConstantParamsImpl -> which inherits from mk::IConstantParams
class RuntimeParamsImpl -> which inherits from mk::IRuntimeParams
class PlatformParamsImpl -> which inherits from mk::IPlatformParams
