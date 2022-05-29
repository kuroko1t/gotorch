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

import "github.com/PrestoBot/gotorch/go/common"

type ExampleData struct {
	dataset C.ExampleDataSet
	current int
	data    []C.Tensor
	target  []C.Tensor
}

func (it *ExampleData) Data() Tensor {
	ret_tensor := Tensor{}
	ret_tensor.tensor = it.data[it.current]
	return ret_tensor
}

func (it *ExampleData) Target() Tensor {
	ret_tensor := Tensor{}
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
