
package main

import "./common"

func main() {
	//var a C.int = 10
	//var b C.int = 10
	//model := C.modelInit();
	model := torch.ModelInit()
	linear := torch.Torch_nn_Linear(784, 64)
	//fc1 := model.Register_module("fc1", linear)
	fc1 := model.Register_module("fc1", linear)
	//fc1 := C.Register_module(C.CString("fc1"), unsafe.Pointer(&linear), model)
	//fc2 := C.Register_module(C.CString("fc2"), unsafe.Pointer(&linear), model)
	//fc3 := C.Register_module(C.CString("fc3"), unsafe.Pointer(&linear), model)
	data := torch.MnistDataloader("./data", 10)
	x_re := data.Reshape([]int{data.Size(0), 784})
	x_re2 := fc1.Forward(x_re)
	x_re3 := torch.Log_Softmax(x_re2, 1)
	//data.Loader_to_Tensor()
	//fc1.Forward(data)
	//data := C.data_loader(C.CString("./data"), 64)
	//tensor := C.loader_to_torch(data)
	//C.forward(model, tenosr);

}
