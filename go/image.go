package torch

import (
	//"fmt"
	"image/jpeg"
	//"encoding/base64"
	//"image/png"
	"os"
	//"io"
	//"strings"
	"log"
)

type Image struct {
	r []float32
	g []float32
	b []float32
}

func (image *Image) Normalize(mean float[], std float[]) {
	if len(mean) != 3 or len(std) != 3 {
		log.Fatal("mean and std lenght must be 3")
	}
}

func ImageRead(path string) Image {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	img, err := jpeg.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	bounds := img.Bounds()
	var r []float32
	var g []float32
	var b []float32
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r_, g_, b_, _ := img.At(x, y).RGBA()
			r = append(r, float32(r_/257))
			g = append(g, float32(g_/257))
			b = append(b, float32(b_/257))
		}
	}
	return Image{r, g, b}
}
