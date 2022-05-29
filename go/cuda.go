package torch

// #include "gotorch.h"
import "C"

func CudaDeviceCount() C.ulong {
	return C.cuda_getDeviceCount()
}
