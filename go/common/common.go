package torch

// #cgo CFLAGS: -I${SRCDIR}/../../libtorch/include/ -I${SRCDIR}/../../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../../libtorch/lib -L${SRCDIR}/../../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"

//import "unsafe"

type MnistData struct {
	cmdata C.MnistDataSet
}

//func (mdata MnistData) Loader_to_Tensor() GoTensor {
// 	gtensor := GoTensor{}
// 	gtensor.tensor = C.loader_to_tensor(mdata.cmdata)
// 	return gtensor
//}

type GoTensor struct {
	tensor C.Tensor
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

type GoLinear struct {
	linear C.Linear
}

func (linear GoLinear) Forward(tensor GoTensor) GoTensor {
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.forward(linear.linear, tensor.tensor)
	return ret_gtensor
}

type GoModel struct {
	model C.TModel
}

func (gmodel GoModel) Register_module(name string, f GoLinear) GoLinear {
	ret_linear := GoLinear{}
	ret_linear.linear = C.Register_module(C.CString(name), f.linear, gmodel.model)
	return ret_linear
}

func ModelInit() GoModel {
	torhmodel := C.modelInit()
	gmodel := GoModel{model:torhmodel}
	return gmodel
}
func MnistDataloader(path string, batch_size int) GoTensor {
	//data := MnistData{}
	ret_gtensor := GoTensor{}
	//data.cmdata = C.data_loader(C.CString(path), C.int(batch_size))
	ret_gtensor.tensor = C.data_loader(C.CString(path), C.int(batch_size))
	//return data;
	return ret_gtensor;
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


//type Tensor struct {
// 	GoTensor C.Tensor
// 	GoTensor_ptr C.Tensor_ptr
//}
