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

	device := torch.GoDevice{}
	if torch.Cuda_is_available() {
		fmt.Println("gpu is available")
		device = torch.Device("cuda")
	} else {
		device = torch.Device("cpu")
	}
	model.To(device)
	for epoch := 0; epoch < 10; epoch++ {
		batch_index := 0
		for dataset.Next() {
			// Reset gradients.
			optimizer.Zero_grad()

			batch := dataset.Data()
			target := dataset.Target()

			// data to device
			data := batch.To(device)
			target_gpu := target.To(device)

			x := torch.Relu(fc1.Forward(data.Reshape([]int{data.Size(0), 784})))
			x = torch.Dropout(x, 0.5, model.Is_training())
			x = torch.Relu(fc2.Forward(x))

			prediction := torch.Log_Softmax(fc3.Forward(x), 1)
			loss := torch.Nll_loss(prediction, target_gpu)

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
