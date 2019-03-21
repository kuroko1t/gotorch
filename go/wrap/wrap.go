// +build gpu
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
package w

// #cgo CFLAGS: -I${SRCDIR}/../../libtorch/include/ -I${SRCDIR}/../../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../../libtorch/lib -L${SRCDIR}/../../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++ -lnvrtc
// #include "gotorch.h"
import "C"

type cint _Ctype_int
type cfloat _Ctype_float
type cTModel _Ctype_TModel
type cTensor _Ctype_Tensor
type cSGD _Ctype_SGD
type cchar _Ctype_char
type cLinearImpl _Ctype_LinearImpl
type cConv2dImpl  _Ctype_Conv2dImpl
type cFeatureDropoutImpl  _Ctype_FeatureDropoutImpl

func ModelInint() cTModel {
	return cTModel(C.modelInit())
}

func Params_size(model cTModel, size *cint) {
	C.params_size(C.TModel(model), size)
}

func Params(model cTModel, size cint, tensor *cTensor) {
	C.params(C.TModel(model), C.int(size), tensor)
}

func IsTraining(model cTModel) cint {
	return C.istraining(model)
}

func Optimizer_SGD(tensor *cTensor, lr cfloat, size cint) cSGD {
	return C.optimizer_sgd(tensor, lr, size)
}

func Optimizer_zero_grad(optimizer cSGD) {
	C.optimizer_zero_grad(optimizer)
}

func Optimizer_step(optimizer cSGD) {
	C.optimizer_step(optimizer)
}

func Register_module_linear(name *cchar, linear cLinearImpl, model cTModel) cLinearImpl {
	return C.register_module_linear(name, linear, model)
}

func Register_module_conv2d(name  *cchar, conv2d cConv2dImpl, model cTModel) cConv2dImpl {
	return C.register_module_conv2d(name, conv2d, model)
}

func Linear(a, b cint) cLinearImpl {
	return C.linear(a, b)
}

func Conv2d(in_channels, out_channels, kernel_size cint) cConv2dImpl {
	return C.conv2d(in_channels, out_channels, kernel_size)
}

func FeatureDropout() cFeatureDropoutImpl {
	return C.FeatureDropout()
}

func Register_module_featureDropout(name *cchar,
	featuredrop cFeatureDropoutImpl, model cTModel) cFeatureDropoutImpl {
	return C.register_module_featureDropout(name, featuredrop, model)
}

func Forward_linear(mod cLinearImpl, tensor cTensor) cTensor {
	return C.forward_linear(mod, tensor)
}

func forward_conv2d(conv2d cConv2dImpl, tensor cTensor) cTensor {
	return C.forward_conv2d(conv2d, tensor)
}

func Forward_featureDropout(featuredrop cFeatureDropoutImpl, tensor cTensor) cTensor {
	return C.forward_featureDropout(featuredrop, tensor)
}

func Data_loader_size(path  *cchar, batch_size cint) cint {
	return C.data_loader_size(path, batch_size)
}

func Data_loader(path  *cchar, batch_size cint, data_vec, target_vec *cTensor) {
	C.data_loader(path, batch_size, data_vec, target_vec)
}

func Tensor_size(tensor cTensor, dim cint) cint {
	return C.tensor_size(tensor, dim)
}

func Tensor_reshape(tensor cTensor, shape *cint, size cint) cTensor {
	return C.tensor_reshape(tensor, shape, size)
}

func Tensor_view(tensor cTensor, shape *cint, size cint) cTensor {
	return C.tensor_view(tensor, shape, size)
}

func Backward(tensor cTensor) {
	C.backward(tensor)
}

func Tensor_item(tensor cTensor) cfloat {
	return C.tensor_item(tensor)
}

func Log_softmax(tensor cTensor, dim cint) cTensor {
	return C.log_softmax(tensor, dim)
}

func Tensor_nll_loss(tensor, target cTensor) cTensor {
	return C.tensor_nll_loss(tensor, target)
}

func Relu(tensor cTensor) cTensor {
	return C.relu(tensor)
}

func Dropout(tensor cTensor, droprate cfloat, is_training cint) cTensor {
	return C.dropout(tensor, droprate, is_training)
}

func Max_pool2d(tensor cTensor, kernel_size cint) cTensor {
	return C.max_pool2d(tensor, kernel_size)
}

func Save(model cTModel, path *cchar) {
	C.save(model, path)
}

func Cuda_is_available() cint {
	return C.cuda_is_available()
}
