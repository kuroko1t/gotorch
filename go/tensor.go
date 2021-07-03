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
import (
    "reflect"
    "unsafe"
    "fmt"
)

type Tensor struct {
	tensor C.Tensor
	device GoDevice
}

type ATensor struct {
	atensor C.ATensor
	device GoDevice
}

type Tensors struct {
	tensors []C.Tensor
}

func (atensor ATensor) Value() []float32 {
	tensor_size := (int)(C.AtensorSize(atensor.atensor))
	tensor_value := make([]float32, tensor_size)
	h := (*reflect.SliceHeader)((unsafe.Pointer)(&tensor_value))
	h.Data = (uintptr)((unsafe.Pointer)(C.AtensorToVec(atensor.atensor)))
	h.Len = tensor_size
	h.Cap = tensor_size
	for _, tv := range tensor_value {
		fmt.Println(tv)
	}
	return tensor_value
	//&tensor_value[0] = C.AtensorToVec(atensor.atensor)
	//float32vec AtensorToVec(ATensor atensor)
    //for i, _ := range tensor_value {
    //    tensor_value[i] = float32(C.AtensorToVec(atensor.atensor)[i])
    //}
}


func (tensor Tensor) Size(dim int) int {
	return int(C.tensor_size(tensor.tensor, C.int(dim)))
}

func (tensor Tensor) Reshape(shapes []int) Tensor {
	cshapes := make([]C.int, len(shapes))
	for i, shape := range shapes {
		cshapes[i] = C.int(shape)
	}
	ret_tensor := Tensor{}
	ret_tensor.tensor = C.tensor_reshape(tensor.tensor, &cshapes[0], C.int(len(shapes)))
	ret_tensor.device = tensor.device
	return ret_tensor
}

func (tensor Tensor) Backward() {
	C.backward(tensor.tensor)
}

func (tensor Tensor) Item() float32 {
	return float32(C.tensor_item(tensor.tensor))
}

func (tensor Tensor) View(shapes []int) Tensor {
	cshapes := make([]C.int, len(shapes))
	for i, shape := range shapes {
		cshapes[i] = C.int(shape)
	}
	ret_tensor := Tensor{}
	ret_tensor.tensor = C.tensor_view(tensor.tensor, &cshapes[0], C.int(len(shapes)))
	return ret_tensor
}

func (tensor *Tensor) To(device GoDevice) Tensor {
	ret_tensor := Tensor{}
	if device.cuda != nil {
		ret_tensor.tensor = C.tensor_to_cuda(tensor.tensor, device.cuda)
	} else if device.cpu != nil {
		ret_tensor.tensor = C.tensor_to_cpu(tensor.tensor, device.cpu)
	}
	ret_tensor.device = device
	return ret_tensor
}

func (tensor Tensor) Is_cuda() bool {
	if C.tensor_is_cuda(tensor.tensor) != 0 {
		return true
	} else {
		return false
	}
}

func Randn(shapes []int) Tensor {
    cshapes := make([]C.int, len(shapes))
	for i, shape := range shapes {
		cshapes[i] = C.int(shape)
	}
    ret_tensor := Tensor{}
    ret_tensor.tensor = C.Randn(&cshapes[0], C.int(len(shapes)))
    return ret_tensor
}

//func tensor_device_check(tensor Tensor) {
// 	if C.tensor_is_cuda(tensor.tensor) != 0 {
// 		if tensor.device.cuda == nil {
// 			log.Fatal("Tensor is gpu, but model is cpu")
// 		}
// 	}
//}
