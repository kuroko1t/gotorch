
package main

import "./common"
import "fmt"

func main() {
	model := torch.ModelInit()
	fc1 := model.Register_module("fc1", torch.Torch_nn_Linear(784, 64))
	fc2 := model.Register_module("fc2", torch.Torch_nn_Linear(64, 32))
	fc3 := model.Register_module("fc3", torch.Torch_nn_Linear(32, 10))

	dataset := torch.MnistDataloader("./data", 64)
	optimizer := torch.Opimizer(model.Parameters(), 0.01)
	for epoch := 0; epoch < 10; epoch++ {
		batch_index := 0
		for dataset.Next() {
			optimizer.Zero_grad()
			batch := dataset.Data()
			x_re  := batch.Reshape([]int{batch.Size(0), 784})
			x_re2 := torch.Relu(fc1.Forward(x_re))
			x_re3 := torch.Relu(fc2.Forward(x_re2))
			prediction := torch.Log_Softmax(fc3.Forward(x_re3), 1)
			loss := torch.Nll_loss(prediction, dataset.Target())
			loss.Backward()
			optimizer.Step()
			batch_index += 1
			if (batch_index %100==0) {
				fmt.Println(loss.Item())
			}
		}
	}
}
