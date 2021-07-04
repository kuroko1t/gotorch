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

type SGD struct {
	param C.SGD
}

func Opimizer(tensors Tensors, lr float32) SGD {
	sgd := SGD{}
	sgd.param = C.optimizer_sgd(&tensors.tensors[0], C.float(lr), C.int(len(tensors.tensors)))
	return sgd
}

func (sgd SGD) Zero_grad() {
	C.optimizer_zero_grad(sgd.param)
}

func (sgd SGD) Step() {
	C.optimizer_step(sgd.param)
}
