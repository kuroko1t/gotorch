#ifdef __cplusplus
extern "C" {
  #endif
  typedef void Golinear;
  //typedef void* ModuleHolder;
  typedef void* Module;
  void* modelInit();
  Golinear torch_nn_Linear(int a, int b);
  Module Register_module(const char *name, Golinear *module_holder, Module module);
  #ifdef __cplusplus

}
#endif
