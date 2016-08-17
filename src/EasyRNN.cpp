#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <functional>

//sigmoid函数
static float sigmoid(const float x)
{
	return 1.0f / (1.0f + std::exp(-x));
}
//sigmoid的导数
static float df_sigmoid(const float y)
{
	return y*(1.0f - y);
}
//cross entropy函数
static float cross_entropy_error(const std::vector<float>& groundTruth,const std::vector<float>& testResult)
{
	assert(groundTruth.size() == testResult.size());
	assert(groundTruth.size() > 0);
	float loss = 0.0f;
	for (size_t i = 0; i < groundTruth.size();i++)
	{
		loss += groundTruth[i] * std::log(testResult[i]) + (1.0f - groundTruth[i])*std::log(1.0f - testResult[i]);
	}
	return loss;
}
//cross entropy残差
static std::vector<float> df_cross_entropy(const std::vector<float>& groundTruth, const std::vector<float>& testResult)
{
	assert(groundTruth.size() == testResult.size());
	assert(groundTruth.size() > 0);
	std::vector<float> diff(groundTruth.size(), 0.0f);
	for (size_t i = 0; i < groundTruth.size(); i++)
	{
		diff[i] -= groundTruth[i] / testResult[i];
	}
	return diff;
}
//数字转换成二进制
static std::string convertNumberToBinary(int x)
{
	std::string result;
	result.resize(sizeof(x) * 8);
	assert(result.size() > 0);
	for (size_t i = result.size()-1; i >= 0; i--)
	{
		if (x%2==0)
		{
			result[i] = '0';
		}
		else
		{
			result[i] = '1';
		}
		x /= 2;
	}
	return result;
}
//生成随机数，size：数量，maxVal：最大值
static std::vector<int> generateRandomValue(const int size, const int maxVal)
{
	std::vector<int> result(size);
	std::random_device rd;
	for (size_t i = 0; i < result.size();i++)
	{
		result[i] = rd() % maxVal;
	}
	return result;
}
//生成操作结果 （相加，或者...)
static std::vector<int> getOpResult(const std::vector<int>& x1, const std::vector<int>& x2, std::function<int(int, int)> OpCb)
{
	assert(x1.size() == x2.size());
	std::vector<int> result(x1.size());
	for (size_t i = 0; i < result.size();i++)
	{
		result[i] = OpCb(x1[i],x2[i]);
	}
	return result;
}
//获取预测精度
static float getAccuracy(const std::vector<int>& groundTruth, const std::vector<int>& testResult)
{
	assert(groundTruth.size() == testResult.size());
	assert(groundTruth.size() > 0);
	size_t correctCount = 0;
	for (size_t i = 0; i < groundTruth.size(); i++)
	{
		if (testResult[i] == groundTruth[i])
		{
			correctCount++;
		}
	}
	const float accuracy = (float)correctCount / (float)groundTruth.size();
	return accuracy;
}
//高斯分布
static void normal_distribution_init(float* data, const size_t size, const float mean_value, const float standard_deviation)
{
	std::random_device rd;
	std::mt19937 engine(rd());
	std::normal_distribution<float> dist(mean_value, standard_deviation);
	for (size_t i = 0; i < size; i++)
	{
		data[i] = dist(engine);
	}
}
//常量分布
static void const_distribution_init(float* data, const size_t size, const float const_value)
{
	for (size_t i = 0; i < size; i++)
	{
		data[i] = const_value;
	}
}
static std::pair<std::vector<float>, std::vector<float>> fetchSamples(const std::vector<int>& x1, 
	const std::vector<int>& x2, 
	const std::vector<int>& y,
	const size_t offset,
	const size_t count)
{
	std::pair<std::vector<float>, std::vector<float>> result;
	//todo
	return result;
}
int main(int, char*[])
{
	//step1 : generate samples
	std::cout << "begin generate samples..." << std::endl;
	const int sampleCount = 1024;
	const std::vector<int> train_x1 = generateRandomValue(sampleCount, 2024);
	const std::vector<int> train_x2 = generateRandomValue(sampleCount, 2024);
	const std::vector<int> train_y = getOpResult(train_x1, train_x2, [](int x1, int x2){return x1 + x2; });
	std::cout << "finish generate samples.\n" << std::endl;

	//step2 : construct network
	std::cout << "begin build RNN network..." << std::endl;
	//网络结构：32 x {2->16->1}
	//10 epoch
	const int maxTrainSampleCount = sampleCount * 10;
	const int bactch = 16;
	float learnRate = 0.1f;
	const int depth = 32;
	const int input_dim = 2;
	const int hidden_dim = 16;
	const int output_dim = 1;	
	//数据
	std::vector<float> input_data(bactch*depth*input_dim, 0.0f);
	//+1是因为第一个节点前面没有数据，需要填充0
	std::vector<float> hidden_data((depth + 1)*hidden_dim, 0.0f);
	std::vector<float> output_data(depth*output_dim, 0.0f);
	//权值
	std::vector<float> ih_weight(depth*input_dim*hidden_dim);
	std::vector<float> hh_weight(depth*hidden_dim*hidden_dim);
	std::vector<float> ho_weight(depth*hidden_dim*output_dim);
	normal_distribution_init(&ih_weight[0], ih_weight.size(), 0.0f, 0.1f);
	normal_distribution_init(&hh_weight[0], hh_weight.size(), 0.0f, 0.1f);
	normal_distribution_init(&ho_weight[0], ho_weight.size(), 0.0f, 0.1f);
	//偏置
	std::vector<float> h_bias(depth*hidden_dim,0.0f);
	std::vector<float> o_bias(depth*output_dim, 0.0f);
	//Loss函数 : cross entropy
	//残差	
	std::vector<float> h_diff(hidden_data.size(), 0.0f);
	std::vector<float> o_diff(output_data.size(), 0.0f);
	//权值修正值
	std::vector<float> ih_delta(ih_weight.size(), 0.0f);
	std::vector<float> hh_delta(hh_weight.size(), 0.0f);
	std::vector<float> ho_delta(ho_weight.size(), 0.0f);
	//偏置修正值
	std::cout << "finish build RNN network.\n" << std::endl;

	//step3 : train network
	std::cout << "begin train..." << std::endl;
	int trainedSampleCount = 0;
	while (trainedSampleCount < maxTrainSampleCount)
	{
		//fetch data
		const std::pair<std::vector<float>, std::vector<float>> train_batch_samples = fetchSamples(train_x1, train_x2, train_y, trainedSampleCount, bactch);
		const std::vector<float> train_batch_input = train_batch_samples.first;
		const std::vector<float> train_batch_output = train_batch_samples.second;
		assert(train_batch_input.size() == train_batch_output.size() * 2);
		assert(train_batch_output.size() % depth == 0);
		//forward
		{
			//wrap to input

			//input -> hidden
			for (int i = 0; i < depth;i++)
			{

			}
		}
		//backward
		//output loss and train accuracy
		trainedSampleCount += bactch;
	}
	std::cout << "finish train.\n" << std::endl;

	//todo : save model & load model
	//step4 : test network
	std::cout << "begin test..." << std::endl;
	const std::vector<int> test_x1 = generateRandomValue(sampleCount, 2024);
	const std::vector<int> test_x2 = generateRandomValue(sampleCount, 2024);
	const std::vector<int> test_gt_y = getOpResult(test_x1, test_x2, [](int x1, int x2){return x1 + x2; });
	//预测
	std::vector<int> test_result_y(test_gt_y.size(),0);
	const float test_accuracy = getAccuracy(test_gt_y, test_result_y);
	std::cout << "final test accuracy is " << test_accuracy << std::endl;
	std::cout << "finish test.\n" << std::endl;

	return 0;
}