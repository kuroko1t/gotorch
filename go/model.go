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
import "log"

type Impl struct {
	conv2d         C.Conv2dImpl
	linear         C.LinearImpl
	featureDropout C.FeatureDropoutImpl
}

type GoModel struct {
	model C.TModel
}

type GoDevice struct {
	cuda C.CUDA
	cpu C.CPU
}

func (model GoModel) Register_module(name string, f Impl) Impl {
	ret_impl := Impl{}
	if f.linear != nil {
		ret_impl.linear = C.register_module_linear(C.CString(name), f.linear, model.model)
	} else if f.conv2d != nil {
		ret_impl.conv2d = C.register_module_conv2d(C.CString(name), f.conv2d, model.model)
	} else if f.featureDropout != nil {
		ret_impl.featureDropout =
			C.register_module_featureDropout(C.CString(name), f.featureDropout, model.model)
	}
	return ret_impl
}

func ModelInit() GoModel {
	torhmodel := C.modelInit()
	gmodel := GoModel{model: torhmodel}
	return gmodel
}

func Linear(a, b int) Impl {
	impl := Impl{}
	impl.linear = C.linear(C.int(a), C.int(b))
	return impl
}

func Conv2d(in_channels, out_channels, kernel_size int) Impl {
	impl := Impl{}
	impl.conv2d = C.conv2d(C.int(in_channels), C.int(out_channels), C.int(kernel_size))
	return impl
}

func FeatureDropout() Impl {
	impl := Impl{}
	impl.featureDropout = C.FeatureDropout()
	return impl
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

func Device(device string) GoDevice {
	godevice := GoDevice{}
	if device == "cuda" {
		godevice.cuda = C.cuda_device()
	} else if device == "cpu" {
		godevice.cpu = C.cpu_device()
	} else {
		log.Fatal("Please input cpu or cuda")
	}
	return godevice
}

func (model *GoModel) To(device GoDevice) {
	if device.cuda != nil {
		C.model_to_cuda(model.model, device.cuda)
	} else if device.cpu != nil {
		C.model_to_cpu(model.model, device.cpu)
	}
}

func Cuda_is_available() bool {
	if C.cuda_is_available() == 0 {
		return false
	} else {
		return true
	}
}
