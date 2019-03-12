#ifdef __cplusplus
extern "C" {
  #endif
  typedef void* Linear;
  //typedef void* Tensor_ptr;
  typedef void* Tensor;
  typedef void* ATensor;
  typedef void* Module;
  typedef void* MnistDataSet;
  typedef void* TModel;
  TModel modelInit();
  Linear torch_nn_Linear(int a, int b);
  TModel Register_module(const char *name, Linear liner, TModel model);
  Tensor forward(TModel mod, Tensor tensor);
  Tensor data_loader(const char *path, int batch_size);
  //Tensor loader_to_tensor(MnistDataSet dataset);
  #ifdef __cplusplus
}
#endif
