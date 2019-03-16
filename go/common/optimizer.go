package torch

// #cgo CFLAGS: -I${SRCDIR}/../../libtorch/include/ -I${SRCDIR}/../../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../../libtorch/lib -L${SRCDIR}/../../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"

type SGD struct {
	param C.SGD
}

func Opimizer(tensors GoTensors, lr float32) SGD {
	sgd := SGD{}
	sgd.param = C.optimizer_sgd(&tensors.tensors[0], C.float(lr), C.int(len(tensors.tensors)))
	return sgd
}

func (sgd SGD) Zero_grad() {
	C.optimizer_zero_grad(sgd.param)
}

func (sgd SGD) Step() {
	C.optimizer_step(sgd.param)
}
