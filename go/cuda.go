package torch

// #include "gotorch.h"
import "C"

func CudaDeviceCount() uintptr {
	return C.cuda_getDeviceCount()
}
