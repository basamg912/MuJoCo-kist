#include "policy.h"
#include <iostream>

Policy::Policy(const std::string& model_path)
// ? { } 생성자 안에서 하면 대입에 가까움. ':' 에 쓰면 메모리 할당할때 초기화하면서 같이 값 넣어버림
// ? const 변수들도 ':' 에서 초기화 , 참조멤버 int& ref (.h) -> : ref(ext_var) (.cpp) 도 마찬가지 
    : _env(ORT_LOGGING_LEVEL_WARNING, "policy") 
{
    Ort::SessionOptions session_options;
    // ! 1) 하나 연산에 몇개 쓰레드 사용할지 결정, onnx 연산 그래프 최적화를 얼마나할지 결정
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // ! 2) 세션 객체 생성 + 메모리 할당 방식 결정
    _session = Ort::Session(_env, model_path.c_str(), session_options);
    // ? Arena : os 한테 메모리를 매번 요청하지말고 처음부터 큰 구역만큼 빌려온다. 그 구역을 Arena 라고 표현
    _memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // ! 3) Allocator 는 메모리를 관리하는 주체, 0번째 인덱스 레이어 = 입력단 이름
    Ort::AllocatorWithDefaultOptions allocator;
    // auto input_name_ptr = _session.GetInputNameAllocated(0, allocator);
    _input_name = "obs";
    auto input_shape = _session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // ? 실제 feature 는 마지막 차원, batch_size 제외
    _input_dim = input_shape.back();
    // ! 4) 
    // auto output_name_ptr = _session.GetOutputNameAllocated(0, allocator);
    _output_name="actions";
    auto output_shape = _session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    _output_dim = output_shape.back();
    std::cout << "[INFO]\tFILE PATH\t:\t" << model_path << '\n';
    std::cout << "[INFO]\tINPUT DIM\t:\t" << _input_dim << '\n';
    std::cout << "[INFO]\tOutput DIM\t:\t" << _output_dim << '\n';
}

Policy::~Policy() {}

// ? VectorXd 는 (d) -> double type, static_cast 를 사용해서 정밀도 손실 명시적 허용
Eigen::VectorXd Policy::inference(const Eigen::VectorXd& obs){
    std::vector<float> input_data(obs.size());
    for (int i=0; i<obs.size(); i++){
        input_data[i] = static_cast<float>(obs[i]);
    }
    std::vector<int64_t> input_shape = {1, _input_dim};
    // ! 5) 텐서 생성, 메모리 정보, 입력 데이터, 입력 크기, 입력 차원, 차원 갯수
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        _memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()
    );
    
    const char* input_name[] = {_input_name.c_str()};
    const char* output_name[] = {_output_name.c_str()};

    // ? input, output 이 1개씩 
    // ? network 구성안하고 알아서 onnx 에서 연산그래프 최적화 후 연산 진행 해주는데, input_tensor, output_tensor 전후처리는 직접해야한다.
    auto output_tensors = _session.Run(
        Ort::RunOptions{nullptr},
        input_name, &input_tensor, 1,
        output_name, 1);
    
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    Eigen::VectorXd action(_output_dim);
    for (int i=0; i<_output_dim; i++){
        action[i] = output_data[i];
    }
    return action;
}