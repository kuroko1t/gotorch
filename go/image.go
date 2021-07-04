package torch

// #include "gotorch.h"
import "C"
import (
	"image/jpeg"
	"math"
	"os"
	"log"
)

type Image struct {
	w int
	h int
	r []float32
	g []float32
	b []float32
}

func Mean(value []float32) (float32) {
	var sum float32
	for _, v := range value {
		sum += v
	}
	mean := sum / float32(len(value))
	return mean
}

func Std(value []float32, mean float32) (float32) {
	var sd float32
	for _, v := range value {
		sd += float32(math.Pow(float64(v - mean), 2))
	}
	sd = float32(math.Sqrt(float64(sd / float32(len(value)))))
	return sd
}

func max(value []float32) float32 {
	var max_val float32
	for _, v := range value {
		if max_val < v {
			max_val = v
		}
	}
	return max_val
}

func min(value []float32) float32 {
	var min_val float32
	for _, v := range value {
		if min_val > v {
			min_val = v
		}
	}
	return min_val
}


func norm(v float32, mean_param, std_param float32) float32 {
	res := (v - mean_param) / std_param
	return res
}

func (image *Image) Normalize(mean_param []float32, std_param []float32) {
	if len(mean_param) != 3 || len(std_param) != 3 {
		log.Fatal("mean and std lenght must be 3")
	}
	for i, r := range image.r {
		image.r[i] = norm(r, mean_param[0], std_param[0])
		image.g[i] = norm(image.g[i], mean_param[1], std_param[1])
		image.b[i] = norm(image.b[i], mean_param[2], std_param[2])
	}
}

func (image *Image) ToTensor() ATensor {
	r_c := make([]C.float, len(image.r))
	g_c := make([]C.float, len(image.g))
	b_c := make([]C.float, len(image.b))
	for i, _ := range image.r {
		r_c[i] = C.float(image.r[i])
		g_c[i] = C.float(image.g[i])
		b_c[i] = C.float(image.b[i])
	}
	hole_value := append(r_c,  g_c...)
	hole_value = append(hole_value,  b_c...)
	shapes := []int{1, 3, image.h, image.w}
	shapes_c := make([]C.int, len(image.b))
	for i, s := range shapes {
		shapes_c[i] = C.int(s)
	}

	atensor := ATensor{}
	atensor.atensor = C.from_blob(&hole_value[0], &shapes_c[0], C.int(len(shapes)))
	return atensor
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
			r = append(r, float32(r_)/257.0/ 255.0)
			g = append(g, float32(g_)/257.0/ 255.0)
			b = append(b, float32(b_)/257.0/ 255.0)
		}
	}
	h := bounds.Max.Y - bounds.Min.Y
	w := bounds.Max.X - bounds.Min.X
	return Image{h, w, r, g, b}
}
