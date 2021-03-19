package main

import "github.com/kuroko1t/gotorch/go"
import "fmt"

func main() {
    model := torch.Load("resnet_scipt.pt")
    dataset := torch.MnistDataloader("./data", 64)

}

