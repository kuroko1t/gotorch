package torch

// #cgo CFLAGS: -I${SRCDIR}/../../libtorch/include/ -I${SRCDIR}/../../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../../libtorch/lib -L${SRCDIR}/../../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"
//import "reflect"
//import "unsafe"
import "fmt"
import "log"
import "os"

type ExampleData struct {
	dataset C.ExampleDataSet
	current int
	data []C.Tensor
	target []C.Tensor
}

func (it *ExampleData) Data() GoTensor {
	//fmt.Println("it.current:",it.current)
	ret_tensor := GoTensor{}
	ret_tensor.tensor = it.data[it.current]
	return ret_tensor
}

func (it *ExampleData) Target() GoTensor {
	ret_tensor := GoTensor{}
	ret_tensor.tensor = it.target[it.current]
	return ret_tensor
}

func (it *ExampleData) Next() bool {
    it.current += 1
    if it.current >= len(it.data) {
		it.current = -1
        return false
    }
    return true
}

type GoTensor struct {
	tensor C.Tensor
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
	return ret_tensor
}

func (tensor GoTensor) Backward() {
	C.backward(tensor.tensor)
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

type SGD struct {
	param C.SGD
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

func (model GoModel) Parameters() GoTensors {
	var size C.int
	C.params_size(model.model, &size)
	tensor_slice := make([]C.Tensor, size, size)
	tensors := GoTensors{}
	C.params(model.model, size, &(tensor_slice[0]))
	tensors.tensors = tensor_slice
	return tensors
}

func Opimizer(tensors GoTensors, lr float32) SGD {
	sgd := SGD{}
	sgd.param = C.optimizer(&tensors.tensors[0], C.float(lr), C.int(len(tensors.tensors)))
	return sgd
}

func (sgd SGD) Zero_grad() {
	C.optimizer_zero_grad(sgd.param)
}

func (sgd SGD) Step() {
	C.optimizer_step(sgd.param)
}



func MnistDataloader(path string, batch_size int) *ExampleData {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		log.Fatal(err)
	}
	var size C.int
	exdata := ExampleData{current:-1}
	size = C.data_loader_size(C.CString(path), C.int(batch_size))
	fmt.Println(size)
	data_slice := make([]C.Tensor, size, size)
	target_slice := make([]C.Tensor, size, size)
	fmt.Println(len(data_slice))
	C.data_loader(C.CString(path), C.int(batch_size),	&(data_slice[0]), &(target_slice[0]))
	exdata.data = data_slice
	exdata.target = target_slice
	return &exdata;
}

//func (data ExampleData) Data() GoTensor {
// 	ret_gtensor := GoTensor{}
// 	ret_gtensor.tensor = data.data[0]
// 	//ret_gtensor.tensor = C.loader_to_tensor(data.dataset)
// 	return ret_gtensor
//}

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

func Nll_loss(tensor, target GoTensor) GoTensor{
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.tensor_nll_loss(tensor.tensor, target.tensor)
	return ret_gtensor
}

func Relu(tensor GoTensor) GoTensor{
	ret_gtensor := GoTensor{}
	ret_gtensor.tensor = C.relu(tensor.tensor)
	return ret_gtensor
}



//type Tensor struct {
// 	GoTensor C.Tensor
// 	GoTensor_ptr C.Tensor_ptr
//}
