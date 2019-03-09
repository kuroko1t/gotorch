package main
// #cgo CFLAGS: -I${SRCDIR}/../libtorch/include/ -I${SRCDIR}/../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../libtorch/lib -L${SRCDIR}/../build
// #cgo LDFLAGS: -lgotorch -lcaffe2 -lc10 -ltorch -lpthread
//#include "gotorch.h"
import "C"
import "unsafe"

func main() {
	var a C.int = 10
	var b C.int = 10
	model := C.modelInit();
	linear := C.torch_nn_Linear(a, b)
	fc1 := C.Register_module(C.CString("fc1"), unsafe.Pointer(&linear), model)
	fc2 := C.Register_module(C.CString("fc2"), unsafe.Pointer(&linear), model)
	fc3 := C.Register_module(C.CString("fc3"), unsafe.Pointer(&linear), model)
	data := C.data_loder(C.CString("./data"), 64)
	//C.forward(model, tenosr);

}
