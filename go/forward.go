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

// #include "gotorch.h"
import "C"

func (impl Impl) Forward(tensor Tensor) Tensor {
	ret_gtensor := Tensor{}
	//tensor_device_check(tensor)
	if impl.linear != nil {
		ret_gtensor.tensor = C.forward_linear(impl.linear, tensor.tensor)
	} else if impl.conv2d != nil {
		ret_gtensor.tensor = C.forward_conv2d(impl.conv2d, tensor.tensor)
	} else if impl.featureDropout != nil {
		ret_gtensor.tensor = C.forward_featureDropout(impl.featureDropout, tensor.tensor)
	}
	return ret_gtensor
}

func Log_Softmax(tensor Tensor, dim int) Tensor {
	ret_gtensor := Tensor{}
	ret_gtensor.tensor = C.log_softmax(tensor.tensor, C.int(dim))
	return ret_gtensor
}

func Nll_loss(tensor, target Tensor) Tensor {
	ret_gtensor := Tensor{}
	ret_gtensor.tensor = C.tensor_nll_loss(tensor.tensor, target.tensor)
	return ret_gtensor
}

func Relu(tensor Tensor) Tensor {
	ret_gtensor := Tensor{}
	ret_gtensor.tensor = C.relu(tensor.tensor)
	return ret_gtensor
}

func Dropout(tensor Tensor, lr float32, is_training bool) Tensor {
	ret_gtensor := Tensor{}
	is_training_int := 0
	if is_training {
		is_training_int = 1
	}
	ret_gtensor.tensor = C.dropout(tensor.tensor, C.float(lr), C.int(is_training_int))
	return ret_gtensor
}

func Max_pool2d(tensor Tensor, kernel_size int) Tensor {
	ret_gtensor := Tensor{}
	ret_gtensor.tensor = C.max_pool2d(tensor.tensor, C.int(kernel_size))
	return ret_gtensor
}
