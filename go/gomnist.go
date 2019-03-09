package main
// #cgo CFLAGS: -I/root/libtorch/include/ -I/root/libtorch/include/torch/csrc/api/include/
// #cgo LDFLAGS: -L/root/libtorch/lib -L/root/gotorch/build
// #cgo LDFLAGS: -lgotorch -lcaffe2 -lc10 -ltorch -lpthread
//#include </root/gotorch/cpp/gotorch.h>
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
}
