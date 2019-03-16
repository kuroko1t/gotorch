package torch

// #cgo CFLAGS: -I${SRCDIR}/../libtorch/include/ -I${SRCDIR}/../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../libtorch/lib -L${SRCDIR}/../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"

type GoLinear struct {
	linear C.Linear
}


func (linear GoLinear) Forward(tensor GoTensor) GoTensor {
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.forward(linear.linear, tensor.tensor)
	return ret_gtensor
}

func Torch_nn_Linear(a, b int) GoLinear {
	golinear := GoLinear{}
	golinear.linear = C.torch_nn_Linear(C.int(a), C.int(b))
	return golinear
}

func Log_Softmax(tensor GoTensor, dim int) GoTensor {
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.log_softmax(tensor.tensor, C.int(dim))
	return ret_gtensor
}

func Nll_loss(tensor, target GoTensor) GoTensor {
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.tensor_nll_loss(tensor.tensor, target.tensor)
	return ret_gtensor
}

func Relu(tensor GoTensor) GoTensor {
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.relu(tensor.tensor)
	return ret_gtensor
}

func Dropout(tensor GoTensor, lr float32, is_training bool) GoTensor {
	ret_gtensor := GoTensor{}
	is_training_int := 0
	if is_training {
		is_training_int = 1
	}
	ret_gtensor.tensor = C.dropout(tensor.tensor, C.float(lr), C.int(is_training_int))
	return ret_gtensor
}
