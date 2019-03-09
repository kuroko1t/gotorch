#ifdef __cplusplus
extern "C" {
  #endif
  typedef void Golinear;
  typedef void* Tensor_ptr;
  typedef void Tensor;
  typedef void* Module;
  void* modelInit();
  Golinear torch_nn_Linear(int a, int b);
  Module Register_module(const char *name, Golinear *module_holder, Module module);
  Tensor forward(Module *mod, Tensor_ptr tensor);
  void data_loder(const char *path, int batch_size);
  #ifdef __cplusplus
}
#endif
