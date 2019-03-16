#ifdef __cplusplus
extern "C" {
  #endif

  typedef void* Linear;
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

  Linear torch_nn_Linear(int a, int b);
  Linear Register_module(const char *name, Linear liner, TModel model);
  Tensor forward(Linear mod, Tensor tensor);
  int data_loader_size(const char *path, int batch_size);
  void data_loader(const char *path, int batch_size,
                 Tensor *data_vec, Tensor *target_vec);
  //Tensor loader_to_tensor(ExampleDataSet dataset);

  int tensor_size(Tensor tensor, int dim);
  Tensor tensor_reshape(Tensor tensor, int* shape, int size);
  void backward(Tensor tensor);

  Tensor log_softmax(Tensor tensor, int dim);
  float tensor_item(Tensor tensor);
  Tensor tensor_nll_loss(Tensor tensor, Tensor target);
  Tensor relu(Tensor tensor);
  Tensor dropout(Tensor tensor, float droprate, int is_training);
#ifdef __cplusplus
}
#endif
