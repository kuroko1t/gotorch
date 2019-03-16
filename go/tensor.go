package torch

// #cgo CFLAGS: -I${SRCDIR}/../libtorch/include/ -I${SRCDIR}/../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../libtorch/lib -L${SRCDIR}/../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"

type GoTensor struct {
	tensor C.Tensor
}

type GoTensors struct {
	tensors []C.Tensor
}

func (tensor GoTensor) Size(dim int) int {
	return int(C.tensor_size(tensor.tensor, C.int(dim)))
}

func (tensor GoTensor) Reshape(shapes []int) GoTensor {
	cshapes := make([]C.int, len(shapes))
	for i, shape := range shapes {
		cshapes[i] = C.int(shape)
	}
	ret_tensor := GoTensor{}
	ret_tensor.tensor = C.tensor_reshape(tensor.tensor, &cshapes[0], C.int(len(shapes)))
	return ret_tensor
}

func (tensor GoTensor) Backward() {
	C.backward(tensor.tensor)
}

func (tensor GoTensor) Item() float32 {
	return float32(C.tensor_item(tensor.tensor))
}
