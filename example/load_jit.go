package main

import "github.com/kuroko1t/gotorch/go"
import "fmt"

func main() {
	torch.ImageRead("n02123394_Persian_cat.jpg")
	shape := []int{1, 3, 224, 224}
	tensor := torch.Randn(shape)
	module := torch.Load("traced_from_pytorch_model.pt")
	output := module.Forward(tensor)
	fmt.Println(output)
}
