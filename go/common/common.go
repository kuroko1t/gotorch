package torch

// #cgo CFLAGS: -I${SRCDIR}/../../libtorch/include/ -I${SRCDIR}/../../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../../libtorch/lib -L${SRCDIR}/../../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"
//import "reflect"
//import "unsafe"
import "fmt"

type ExampleData struct {
	dataset C.ExampleDataSet
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

func (tensor GoTensor) Item() float32 {
	return float32(C.tensor_item(tensor.tensor))
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
func MnistDataloader(path string, batch_size int) ExampleData {
	var size C.int
	loader := C.data_loader(C.CString(path), C.int(batch_size), &size)
	fmt.Println(size)
	datasets := ExampleData{}
	datasets.dataset = loader
	//datasets := make([]ExampleDatap, size)
	//var go_array []C.ExampleDataSet
	//slice := (*reflect.SliceHeader)(unsafe.Pointer(&go_array))
	//slice.Cap = int(size)
	//slice.Len = int(size)
	//slice.Data = uintptr(unsafe.Pointer(loader))
	//for i := range datasets {
	// 	datasets[i].dataset = go_array[i]
	//}
	return datasets;
}

func (data ExampleData) Data() GoTensor {
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.loader_to_tensor(data.dataset)
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

func Nll_Loss(tensor, target GoTensor) GoTensor{
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.tensor_nll_loss(tensor.tensor, target.tensor)
	return ret_gtensor
}



//type Tensor struct {
// 	GoTensor C.Tensor
// 	GoTensor_ptr C.Tensor_ptr
//}
