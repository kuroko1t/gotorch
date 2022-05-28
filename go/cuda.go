package torch

// #include "gotorch.h"
import "C"

func device_count() uintptr {
	return C.cuda_getDeviceCount()
}
