package torch

// #cgo CFLAGS: -I${SRCDIR}/../libtorch/include/ -I${SRCDIR}/../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../libtorch/lib -L${SRCDIR}/../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"

type GoModel struct {
	model C.TModel
}

func (model GoModel) Register_module(name string, f GoLinear) GoLinear {
	ret_linear := GoLinear{}
	ret_linear.linear = C.Register_module(C.CString(name), f.linear, model.model)
	return ret_linear
}

func ModelInit() GoModel {
	torhmodel := C.modelInit()
	gmodel := GoModel{model: torhmodel}
	return gmodel
}

func (model GoModel) Parameters() GoTensors {
	var size C.int
	C.params_size(model.model, &size)
	tensor_slice := make([]C.Tensor, size, size)
	tensors := GoTensors{}
	C.params(model.model, size, &(tensor_slice[0]))
	tensors.tensors = tensor_slice
	return tensors
}

func (model GoModel) Is_training() bool {
	if int(C.istraining(model.model)) != 0 {
		return true
	} else {
		return false
	}
}
