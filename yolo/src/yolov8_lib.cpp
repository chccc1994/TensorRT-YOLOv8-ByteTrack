#include "yolov8_lib.h"

YoloDetecter::YoloDetecter(const std::string trtFile): trtFile_(trtFile)
// YoloDetecter::YoloDetecter(const std::string trtFile)
{
    gLogger = Logger(ILogger::Severity::kERROR);
    cudaSetDevice(kGpuId);

    // load engine
    deserialize_engine();

    CUDA_CHECK(cudaStreamCreate(&stream));

    // bytes of input and output
    kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
    vTensorSize.resize(2, 0);
    vTensorSize[0] = 3 * kInputH * kInputW * sizeof(float);
    vTensorSize[1] = kOutputSize * sizeof(float);

    // prepare input data and output data ---------------------------
    inputData = new float[3 * kInputH * kInputW];
    outputData = new float[kOutputSize];

    // prepare input and output space on device
    vBufferD.resize(2, nullptr);
    for (int i = 0; i < 2; i++)
    {
        CUDA_CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }
}
YoloDetecter::~YoloDetecter() {
    cudaStreamDestroy(stream);
    for (void* ptr : vBufferD) {
        CUDA_CHECK(cudaFree(ptr));
    }
    // 正确释放 TensorRT 对象
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
    delete[] inputData;
    delete[] outputData;
}

void YoloDetecter::deserialize_engine()
{
    std::ifstream file(trtFile_, std::ios::binary);
    if (!file.good()){
        std::cerr << "read " << trtFile_ << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(serialized_engine, size);
    context = engine->createExecutionContext();
    delete[] serialized_engine;
}


void YoloDetecter::inference()
{
    CUDA_CHECK(cudaMemcpyAsync(vBufferD[0], (void *)inputData, vTensorSize[0], cudaMemcpyHostToDevice, stream));
    context->enqueue(1, vBufferD.data(), stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync((void *)outputData, vBufferD[1], vTensorSize[1], cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::vector<DetectResult> YoloDetecter::inference(cv::Mat& img)
{
    preprocess(img, inputData, kInputH, kInputW);  // put image data on inputData

    inference();

    std::vector<Detection> res;
    nms(res, outputData, kConfThresh, kNmsThresh);

    std::vector<DetectResult> final_res;
    for (size_t j = 0; j < res.size(); j++)
    {
        cv::Rect r = get_rect(img, res[j].bbox);
        DetectResult single_res {r, res[j].conf, (int)res[j].class_id};
        final_res.push_back(single_res);
    }

    return final_res;
}
