/*
MIT License

Copyright (c) 2019 kurosawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
package torch

// #cgo CFLAGS: -I${SRCDIR}/../libtorch/include/ -I${SRCDIR}/../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../libtorch/lib -L${SRCDIR}/../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"

func (linear GoLinear) Forward(tensor GoTensor) GoTensor {
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.forward(linear.linear, tensor.tensor)
	return ret_gtensor
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
