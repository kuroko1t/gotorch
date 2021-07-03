package main

import "github.com/kuroko1t/gotorch/go"
import "fmt"

func main() {
	shape := []int{1, 3, 224, 224}
	tensor := torch.Randn(shape)
	module := torch.Load("resnet_script.pt")
	output := module.Forward(tensor)
	fmt.Println(output.Value())
}
