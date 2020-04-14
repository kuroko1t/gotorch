package main

import "github.com/kuroko1t/gotorch/go"
import "fmt"
import "testing"

func TestMnistLinar(t *testing.T) {
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
	for epoch := 0; epoch < 2; epoch++ {
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

func TestMnistCNN(t *testing.T) {
	model := torch.ModelInit()
	conv1 := model.Register_module("conv1", torch.Conv2d(1, 10, 5))
	conv2 := model.Register_module("conv2", torch.Conv2d(10, 20, 5))
	conv2_drop := model.Register_module("conv2_drop", torch.FeatureDropout())
	fc1 := model.Register_module("fc1", torch.Linear(320, 50))
	fc2 := model.Register_module("fc2", torch.Linear(50, 10))

	if torch.Cuda_is_available() {
		fmt.Println("gpu is available")
	}

	dataset := torch.MnistDataloader("./data", 64)
	optimizer := torch.Opimizer(model.Parameters(), 0.01)
	for epoch := 0; epoch < 2; epoch++ {
		batch_index := 0
		for dataset.Next() {
			optimizer.Zero_grad()

			batch := dataset.Data()

			x := torch.Relu(torch.Max_pool2d(conv1.Forward(batch), 2))
			x = torch.Relu(torch.Max_pool2d(
				conv2_drop.Forward(conv2.Forward(x)), 2))
			x = x.View([]int{-1, 320})
			x = torch.Relu(fc1.Forward(x))
			x = torch.Dropout(x, 0.5, model.Is_training())
			x = fc2.Forward(x)
			prediction := torch.Log_Softmax(x, 1)
			loss := torch.Nll_loss(prediction, dataset.Target())

			loss.Backward()
			optimizer.Step()

			batch_index += 1
			if batch_index%100 == 0 {
				fmt.Println("epoch:", epoch, "batch:", batch_index, "loss:", loss.Item())
				model.Save("net.pt")
			}
		}
	}
}
