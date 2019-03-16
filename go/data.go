package torch

// #cgo CFLAGS: -I${SRCDIR}/../libtorch/include/ -I${SRCDIR}/../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../libtorch/lib -L${SRCDIR}/../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"
import "log"
import "os"

type ExampleData struct {
	dataset C.ExampleDataSet
	current int
	data    []C.Tensor
	target  []C.Tensor
}

func (it *ExampleData) Data() GoTensor {
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

func MnistDataloader(path string, batch_size int) *ExampleData {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		log.Fatal(err)
	}
	var size C.int
	exdata := ExampleData{current: -1}
	size = C.data_loader_size(C.CString(path), C.int(batch_size))
	data_slice := make([]C.Tensor, size, size)
	target_slice := make([]C.Tensor, size, size)
	C.data_loader(C.CString(path), C.int(batch_size), &(data_slice[0]), &(target_slice[0]))
	exdata.data = data_slice
	exdata.target = target_slice
	return &exdata
}
