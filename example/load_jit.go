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
