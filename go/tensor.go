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
//import "log"

type GoTensor struct {
	tensor C.Tensor
	device GoDevice
}

type ATensor struct {
	atensor C.ATensor
	device GoDevice
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
	ret_tensor.device = tensor.device
	return ret_tensor
}

func (tensor GoTensor) Backward() {
	C.backward(tensor.tensor)
}

func (tensor GoTensor) Item() float32 {
	return float32(C.tensor_item(tensor.tensor))
}

func (tensor GoTensor) View(shapes []int) GoTensor {
	cshapes := make([]C.int, len(shapes))
	for i, shape := range shapes {
		cshapes[i] = C.int(shape)
	}
	ret_tensor := GoTensor{}
	ret_tensor.tensor = C.tensor_view(tensor.tensor, &cshapes[0], C.int(len(shapes)))
	return ret_tensor
}

func (tensor *GoTensor) To(device GoDevice) GoTensor {
	ret_tensor := GoTensor{}
	if device.cuda != nil {
		ret_tensor.tensor = C.tensor_to_cuda(tensor.tensor, device.cuda)
	} else if device.cpu != nil {
		ret_tensor.tensor = C.tensor_to_cpu(tensor.tensor, device.cpu)
	}
	ret_tensor.device = device
	return ret_tensor
}

func (tensor GoTensor) Is_cuda() bool {
	if C.tensor_is_cuda(tensor.tensor) != 0 {
		return true
	} else {
		return false
	}
}

func Randn(shapes []int) GoTensor {
    cshapes := make([]C.int, len(shapes))
	for i, shape := range shapes {
		cshapes[i] = C.int(shape)
	}
    ret_tensor := GoTensor{}
    ret_tensor.tensor = C.Randn(&cshapes[0], C.int(len(shapes)))
    return ret_tensor
}

//func tensor_device_check(tensor GoTensor) {
// 	if C.tensor_is_cuda(tensor.tensor) != 0 {
// 		if tensor.device.cuda == nil {
// 			log.Fatal("Tensor is gpu, but model is cpu")
// 		}
// 	}
//}
