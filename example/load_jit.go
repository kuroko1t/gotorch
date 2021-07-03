package main

import "github.com/kuroko1t/gotorch/go"
import "fmt"

func main() {
	image := torch.ImageRead("neko.jpg")
	image.Normalize([]float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225})
	tensor := image.ToTensor()
	//shape := []int{1, 3, 224, 224}
	//tensor := torch.Randn(shape)
	module := torch.Load("traced_from_pytorch_model.pt")
	output := module.Forward(tensor)
	fmt.Println(output.Argmax())
}
