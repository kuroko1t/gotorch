//#include <torch/nn/modules/linear.h>
//#include <torch/csrc/api/include/torch/nn/module.h>
//#include <torch/nn/module.h>
#include <torch/torch.h>
#include <gotorch.h>

struct TestModel : public torch::nn::Module {
  using torch::nn::Module::register_module;
};

Golinear torch_nn_Linear(int a, int b) {
  return (void)torch::nn::Linear(a, b);
}

void* modelInit() {
  TestModel *mod= new TestModel();
  return (void*)mod;
}

Module Register_module(const char *name, Golinear *liner, Module mod) {
  std::string str(name);
  TestModel *mod_test = (TestModel*)(mod);
  return (void*)(mod_test->register_module(str, std::shared_ptr<TestModel>(mod_test)).get());
}


forward(Module *mod) {
  TestModel *mods = (TestModel*)mod;
}
