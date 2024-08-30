#include "Tensor.hpp"

// Constructor
Tensor::Tensor(const std::vector<Eigen::Index>& dims) : dimensions_(dims), total_size_(1) {
    for (const auto& dim : dimensions_) {
        total_size_ *= dim;
    }
    data_.resize(total_size_);
}

// private function to compute the 1D index from ND index
Eigen::Index Tensor::computeIndex(const std::vector<Eigen::Index>& indices) const {
    Eigen::Index index = 0;
    Eigen::Index multiplier = 1;
    for (int i = dimensions_.size() - 1; i >= 0; --i) {
        index += indices[i] * multiplier;
        multiplier *= dimensions_[i];
    }
    return index;
}

// Access element by multi-dimensional indices
std::shared_ptr<Value>& Tensor::operator()(const std::vector<Eigen::Index>& indices) {
    return data_(computeIndex(indices));
}
const std::shared_ptr<Value>& Tensor::operator()(const std::vector<Eigen::Index>& indices) const {
    return data_(computeIndex(indices));
}

// Fill tensor with a specific value
void Tensor::fill(const std::shared_ptr<Value>& value) {
    data_.setConstant(value);
}

// Get the underlying Eigen::Matrix
const Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1>& Tensor::data() const {
    return data_;
}

// Get the rank
int Tensor::rank() const { 
    return dimensions_.size(); 
}

// Get dimensions of the tensor
std::vector<Eigen::Index> Tensor::dim() const {
    return dimensions_;
}

// Total size
Eigen::Index Tensor::size() const {
    return data_.size();
}

// Display tensor for debugging
void Tensor::display() const {
    std::cout << "Tensor: ";
    for (int i = 0; i < data_.size(); ++i) {
        std::cout << data_.data()[i]->getData() << " ";  // Assuming Value has a get() method
    }
    std::cout << std::endl;
}

// Reshape the tensor
void Tensor::reshape(const std::vector<Eigen::Index>& new_dims) {
    // Compute the total size of the new dimensions
    Eigen::Index new_total_size = 1;
    for (const auto& dim : new_dims) {
        new_total_size *= dim;
    }

    // Check if the total size matches the current size
    if (new_total_size != total_size_) {
        throw std::invalid_argument("Total size of new dimensions must match the current size.");
    }

    // If valid, update dimensions
    dimensions_ = new_dims;
}

// Flatten the tensor
void Tensor::flatten() {
    dimensions_ = { total_size_ };
}

// Static method to create a tensor filled with ones
Tensor Tensor::ones(const std::vector<Eigen::Index>& dims) {
    Tensor tensor(dims);
    for (int i = 0; i < tensor.size(); ++i) {
        tensor.data_(i) = Value::create(1.0f);
    }
    return tensor;
}

// Static method to create a tensor filled with zeros
Tensor Tensor::zeros(const std::vector<Eigen::Index>& dims) {
    Tensor tensor(dims);
    for (int i = 0; i < tensor.size(); ++i) {
        tensor.data_(i) = Value::create(0.0f);
    }
    return tensor;
}

// Static method to create a tensor filled with random values
Tensor Tensor::random(const std::vector<Eigen::Index>& dims, float min, float max) {
    Tensor tensor(dims);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (int i = 0; i < tensor.size(); ++i) {
        tensor.data_(i) = Value::create(dis(gen));
    }
    return tensor;
}


// Static method to create a tensor filled with values from a randn distribution
Tensor Tensor::randn(const std::vector<Eigen::Index>& dims, float mean, float stddev) {
    Tensor tensor(dims);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);

    for (int i = 0; i < tensor.size(); ++i) {
        tensor.data_(i) = Value::create(dis(gen));
    }
    return tensor;
}