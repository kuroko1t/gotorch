/*
MIT License

Copyright (c) 2019 kurosawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifdef __cplusplus
extern "C" {
  #endif

  /* Operation Impl */
  typedef void* LinearImpl;
  typedef void* Conv2dImpl;
  typedef void* FeatureDropoutImpl;

  typedef void* ATensor;
  typedef void* Module;
  typedef void* MnistDataSet;
  typedef void* TModel;
  typedef void* Tensor;
  typedef void* SGD;
  typedef void* ExampleDataSet;

  TModel modelInit();
  void params_size(TModel model, int *size);
  void params(TModel model, int size, Tensor *tensor);
  int istraining(TModel model);

  SGD optimizer_sgd(Tensor *tensor, float lr, int size);
  void optimizer_zero_grad(SGD optimizer);
  void optimizer_step(SGD optimizer);

  LinearImpl linear(int a, int b);
  LinearImpl register_module_linear(const char *name, LinearImpl linear, TModel mod);

  Conv2dImpl conv2d(int in_channels, int out_channels, int kernel_size);
  Conv2dImpl register_module_conv2d(const char *name, Conv2dImpl conv2d, TModel mod);

  FeatureDropoutImpl FeatureDropout();
  FeatureDropoutImpl register_module_featureDropout(const char *name,
                                                    FeatureDropoutImpl featuredrop, TModel mod);

  Tensor forward_linear(LinearImpl mod, Tensor tensor);
  Tensor forward_conv2d(Conv2dImpl conv2d, Tensor tensor);
  Tensor forward_featureDropout(FeatureDropoutImpl featuredrop, Tensor tensor);

  int data_loader_size(const char *path, int batch_size);
  void data_loader(const char *path, int batch_size,
                 Tensor *data_vec, Tensor *target_vec);
  //Tensor loader_to_tensor(ExampleDataSet dataset);

  int tensor_size(Tensor tensor, int dim);
  Tensor tensor_reshape(Tensor tensor, int* shape, int size);
  Tensor tensor_view(Tensor tensor, int* shape, int size);

  void backward(Tensor tensor);

  float tensor_item(Tensor tensor);

  Tensor log_softmax(Tensor tensor, int dim);
  Tensor tensor_nll_loss(Tensor tensor, Tensor target);
  Tensor relu(Tensor tensor);
  Tensor dropout(Tensor tensor, float droprate, int is_training);
  Tensor max_pool2d(Tensor tensor, int kernel_size);

  void save(TModel model, const char *path);
#ifdef __cplusplus
}
#endif
