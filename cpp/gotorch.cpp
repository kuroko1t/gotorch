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

Linear Register_module(const char *name, Linear linear, TModel mod) {
  std::string str(name);
  TorchModel *mod_test = (TorchModel*)(mod);
  torch::nn::LinearImpl *plinear = (torch::nn::LinearImpl*)linear;
  std::shared_ptr<torch::nn::LinearImpl> p1(plinear);
  return (void*)((mod_test->register_module(str, p1)).get());
}

Tensor forward(Linear mod, Tensor tensor) {
  torch::nn::LinearImpl* pmod = (torch::nn::LinearImpl*)mod;
  std::shared_ptr<torch::nn::LinearImpl> p1(pmod);
  torch::Tensor* ptensor = (torch::Tensor*)tensor;
  torch::Tensor *atensor = new torch::Tensor();
  torch::Tensor* go_atensor = (torch::Tensor*)atensor;
  *go_atensor = p1->forward(*ptensor);
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

//using example_data = torch::data::Example<at::Tensor, at::Tensor>;

Tensor data_loader(const char *path, int batch_size,
                 int *size, Tensor *target) {
  std::string spath(path);
  auto dataset = torch::data::make_data_loader(
      torch::data::datasets::MNIST(spath)
      .map(torch::data::transforms::Stack<>()),
      batch_size);
  std::vector<torch::Tensor> data_vec;
  std::vector<torch::Tensor> target_vec;
  for (auto& x : *dataset) {
    data_vec.push_back(x.data);
    target_vec.push_back(x.target);
  }
  *target = &target_vec[0];
  *size = data_vec.size();
  return (void*)&data_vec[0];
}

//Tensor loader_to_tensor(ExampleDataSet dataset) {
//   example_data* mdataset = (example_data*)dataset;
//   torch::Tensor *tensor = new torch::Tensor();
//   *tensor = mdataset->data;
//   return (void*)(tensor);
//}

int tensor_size(Tensor tensor, int dim) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  return atensor->size(dim);
}

Tensor tensor_reshape(Tensor tensor, int* shape, int size) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  std::vector<long> x;
  for (int i = 0; i < size; i++) {
    x.push_back((long)shape[i]);
  }
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = atensor->reshape(x);
  return (void*)ret_tensor;
}

Tensor log_softmax(Tensor tensor, int dim) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = torch::log_softmax(*atensor, dim);
  return (void*)ret_tensor;
}

Tensor tensor_nll_loss(Tensor tensor, Tensor target) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  torch::Tensor *atarget = (torch::Tensor*)target;
  torch::Tensor *ret_tensor = new torch::Tensor();
  *ret_tensor = torch::nll_loss(*atensor, *atarget);
  return (void*)ret_tensor;
}

float tensor_item(Tensor tensor) {
  torch::Tensor *atensor = (torch::Tensor*)tensor;
  return atensor->item<float>();
}
