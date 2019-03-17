package torch

// #cgo CFLAGS: -I${SRCDIR}/../libtorch/include/ -I${SRCDIR}/../libtorch/include/torch/csrc/api/include/ -I${SRCDIR}/../cpp
// #cgo LDFLAGS: -L${SRCDIR}/../libtorch/lib -L${SRCDIR}/../build -lgotorch -lpthread -lcaffe2 -lc10 -ltorch -lstdc++
// #include "gotorch.h"
import "C"
import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"sync"
)

import "github.com/kuroko1t/gotorch/go/common"

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
		fmt.Println("start mnist downloading...")
		download_mnist(path)
		fmt.Println("end mnist download")
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

func download_mnist(path string) {
	err := os.Mkdir(path, 0764)
	common.Log_Fatal(err)
	var wg sync.WaitGroup
	wg.Add(4)
	go download(path+"/train-images-idx3-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", &wg)
	go download(path+"/train-labels-idx1-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", &wg)
	go download(path+"/t10k-images-idx3-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", &wg)
	go download(path+"/t10k-labels-idx1-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", &wg)
	wg.Wait()
	wg.Add(4)
	go unzip(path+"/train-images-idx3-ubyte.gz", &wg)
	go unzip(path+"/train-labels-idx1-ubyte.gz", &wg)
	go unzip(path+"/t10k-images-idx3-ubyte.gz", &wg)
	go unzip(path+"/t10k-labels-idx1-ubyte.gz", &wg)
	wg.Wait()
}

func download(path, url string, wg *sync.WaitGroup) {
	resp, err := http.Get(url)
	common.Log_Fatal(err)

	defer resp.Body.Close()

	out, err := os.Create(path)
	common.Log_Fatal(err)
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	common.Log_Fatal(err)
	defer wg.Done()
}

func unzip(path string, wg *sync.WaitGroup) {
	err := exec.Command("gzip", "-d", path).Run()
	common.Log_Fatal(err)
	defer wg.Done()
}
