//#include <torch/nn/modules/linear.h>
//#include <torch/csrc/api/include/torch/nn/module.h>
//#include <torch/nn/module.h>
#include <torch/torch.h>
#include <gotorch.h>

struct TorchModel : public torch::nn::Module {
  using torch::nn::Module::register_module;
};

Linear torch_nn_Linear(int a, int b) {
  torch::nn::LinearImpl *linear= new torch::nn::LinearImpl(a, b);
  return linear;
}

TModel modelInit() {
  TorchModel *mod= new TorchModel();
  return (void*)mod;
}


TModel Register_module(const char *name, Linear liner, TModel mod) {
  std::string str(name);
  TorchModel *mod_test = (TorchModel*)(mod);
  return (void*)((mod_test->register_module(str, std::shared_ptr<TorchModel>(mod_test))).get());
}

Tensor forward(TModel mod, Tensor tensor) {
  torch::nn::LinearImpl* pmod = (torch::nn::LinearImpl*)mod;
  torch::Tensor* ptensor = (torch::Tensor*)tensor;
  torch::Tensor *atensor = new torch::Tensor();
  torch::Tensor* go_atensor = (torch::Tensor*)atensor;
  *go_atensor = pmod->forward(*ptensor);
  return (void*)go_atensor;
}

using mnistDataset = torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            torch::data::datasets::MNIST,
            torch::data::transforms::Stack<torch::data::Example<>>>,
        torch::data::samplers::RandomSampler>;

//MnistDataSet data_loader(const char *path, int batch_size) {
//  std::string spath(path);
//  mnistDataset *dataset = torch::data::make_data_loader(
//      torch::data::datasets::MNIST(spath)
//      .map(torch::data::transforms::Stack<>()),
//      batch_size).get();
//  return (void*)(dataset);
//}

Tensor data_loader(const char *path, int batch_size) {
  std::string spath(path);
  auto dataset = torch::data::make_data_loader(
      torch::data::datasets::MNIST(spath)
      .map(torch::data::transforms::Stack<>()),
      batch_size);
  torch::Tensor *tensor = new torch::Tensor();
  for (auto &x : *dataset) {
      *tensor = (x.data);
      break;
  }
  return tensor;
  //tensor = &((*dataset).begin()).data;
  //return (void*)(dataset);
}


//Tensor loader_to_tensor(MnistDataSet dataset) {
//   mnistDataset* mdataset = (mnistDataset*)dataset;
//   torch::Tensor *tensor = new torch::Tensor();
//   for (auto &x : *mdataset) {
//       *tensor = (x.data);
//       break;
//   }
//   //*tensor = (*(*mdataset).begin()).data;
//   return (void*)(tensor);
//}
