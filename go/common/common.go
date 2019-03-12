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

type GoLinear struct {
	linear C.Linear
}

type GoModel struct {
	model C.TModel
}

type GoModule struct {
	module C.Module
}

func (mod GoModel) Forward(tensor GoTensor) GoTensor {
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.forward(mod.model, tensor.tensor)
	return ret_gtensor
}

func (gmodel GoModel) Register_module(name string, f GoLinear) GoModel {
	ret_gomodel := GoModel{}
	ret_gomodel.model = C.Register_module(C.CString(name), f.linear, gmodel.model)
	return ret_gomodel
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

//type Tensor struct {
// 	GoTensor C.Tensor
// 	GoTensor_ptr C.Tensor_ptr
//} 
