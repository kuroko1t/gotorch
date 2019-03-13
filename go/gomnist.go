
package main

import "./common"

func main() {
	//var a C.int = 10
	//var b C.int = 10
	model := torch.ModelInit()
	linear := torch.Torch_nn_Linear(784, 10)
	fc1 := model.Register_module("fc1", linear)
	//fc2 := C.Register_module(C.CString("fc2"), unsafe.Pointer(&linear), model)
	//fc3 := C.Register_module(C.CString("fc3"), unsafe.Pointer(&linear), model)
	dataset := torch.MnistDataloader("./data", 10)
	batch := dataset.Data()
	x_re := batch.Reshape([]int{batch.Size(0), 784})
	x_re2 := fc1.Forward(x_re)
	torch.Log_Softmax(x_re2, 1)

}
