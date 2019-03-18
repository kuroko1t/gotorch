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

type GoLinear struct {
	linear C.Linear
}

type Conv2d struct {
	conv C.Conv2dImpl
}

type GoModel struct {
	model C.TModel
}

func (model GoModel) Register_module(name string, f GoLinear) GoLinear {
	ret_linear := GoLinear{}
	ret_linear.linear = C.Register_module_linear(C.CString(name), f.linear, model.model)
	return ret_linear
}

func ModelInit() GoModel {
	torhmodel := C.modelInit()
	gmodel := GoModel{model: torhmodel}
	return gmodel
}

func Torch_nn_Linear(a, b int) GoLinear {
	golinear := GoLinear{}
	golinear.linear = C.torch_nn_Linear(C.int(a), C.int(b))
	return golinear
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

func (model GoModel) Save(path string) {
	C.save(model.model, C.CString(path))
}
