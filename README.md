# gotorch

gobinding for pytorch. please contibute this project

## Install

```
go get github.com/kuroko1t/gotorch
```

## Download Libtorch

* CPU
```
./download.sh
```

* GPU
```
./download.sh -g
```

## Setting

```
$ cd gotorch
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libtorch/lib
```

## Load Torchscript

```golang
package main

import "github.com/kuroko1t/gotorch/go"
import "fmt"

func main() {
	image := torch.ImageRead("neko.jpg")
	tensor := image.ToTensor()
	module := torch.Load("traced_from_pytorch_model.pt")
	output := module.Forward(tensor)
	fmt.Println(output.Argmax())
}
```

## mnist sample

```
$ go run example/gomnist.go
```

If you don't have a GPU.

```
$ go run -tags cpu example/gomnist.go
```

code sample

```golang

package main

import "github.com/kuroko1t/gotorch/go"
import "fmt"

func main() {
	model := torch.ModelInit()
	fc1 := model.Register_module("fc1", torch.Linear(784, 64))
	fc2 := model.Register_module("fc2", torch.Linear(64, 32))
	fc3 := model.Register_module("fc3", torch.Linear(32, 10))

	dataset := torch.MnistDataloader("./data", 64)
	optimizer := torch.Opimizer(model.Parameters(), 0.01)
	for epoch := 0; epoch < 10; epoch++ {
		batch_index := 0
		for dataset.Next() {
			// Reset gradients.
			optimizer.Zero_grad()

			batch := dataset.Data()

			x := torch.Relu(fc1.Forward(batch.Reshape([]int{batch.Size(0), 784})))
			x = torch.Dropout(x, 0.5, model.Is_training())
			x = torch.Relu(fc2.Forward(x))

			prediction := torch.Log_Softmax(fc3.Forward(x), 1)
			loss := torch.Nll_loss(prediction, dataset.Target())

			// Compute gradients of the loss w.r.t. the parameters of our model.
			loss.Backward()

			// Update the parameters based on the calculated gradients.
			optimizer.Step()
			batch_index += 1
			if batch_index%100 == 0 {
				fmt.Println("epoch:", epoch, "batch:", batch_index, "loss:", loss.Item())
				model.Save("net.pt")
			}
		}
	}
}


```

## Support

* Pytorch 1.6.0

## License
MIT
