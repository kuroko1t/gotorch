#ifdef __cplusplus
extern "C" {
  #endif

  typedef void* Linear;
  typedef void* ATensor;
  typedef void* Module;
  typedef void* MnistDataSet;
  typedef void* TModel;
  typedef void* Tensor;
  typedef void* ExampleDataSet;
  TModel modelInit();
  Linear torch_nn_Linear(int a, int b);
  Linear Register_module(const char *name, Linear liner, TModel model);
  Tensor forward(Linear mod, Tensor tensor);
  Tensor data_loader(const char *path, int batch_size,
                     int *size, Tensor *target);
  Tensor loader_to_tensor(ExampleDataSet dataset);

  int tensor_size(Tensor tensor, int dim);
  Tensor tensor_reshape(Tensor tensor, int* shape, int size);
  Tensor log_softmax(Tensor tensor, int dim);
  float tensor_item(Tensor tensor);
  Tensor tensor_nll_loss(Tensor tensor, Tensor target);
#ifdef __cplusplus
}
#endif
