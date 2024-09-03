#include "Tensor.hpp"

// Constructor
Tensor::Tensor(const std::vector<int>& dims) {
    total_size_ = 1;
    // dimensions_ = dims;
    dimensions_ = dims;

    for (auto& d : dimensions_) {
        total_size_ *= d;
    }

    data_ = std::vector<std::shared_ptr<Value>>(total_size_);
}

Tensor::Tensor(const std::vector<int>& dims, std::vector<std::shared_ptr<Value>> data)
           : Tensor(dims) {

    // Init a subtensor sharing same data as main tensor
    // but indexes will be different
    data_ = std::move(data);
}

// private function to compute the 1D index from ND index
int Tensor::computeIndex(const std::vector<int>& indices) const {
    int index = 0;
    int multiplier = 1;
    for (int i = dimensions_.size() - 1; i >= 0; --i) {
        index += indices[i] * multiplier;
        multiplier *= dimensions_[i];
    }
    return index;
}

// Access element by multi-dimensional indices
std::shared_ptr<Value>& Tensor::operator()(const std::vector<int>& indices) {
    return mat()[computeIndex(indices)];
}
const std::shared_ptr<Value>& Tensor::operator()(const std::vector<int>& indices) const {
    return mat()[computeIndex(indices)];
}

Tensor Tensor::operator[](int index) {
    // Validate the index
    if (index < 0 || index >= dimensions_[0]) {
        throw std::out_of_range("Index out of bounds for the first dimension");
    }

    // Calculate the new dimensions after slicing
    std::vector<int> newDims(dimensions_.begin() + 1, dimensions_.end());

    // Calculate the stride for the first dimension
    int stride = 1;
    for (int i = 1; i < dimensions_.size(); ++i) {
        stride *= dimensions_[i];
    }

    // Calculate the offset in the data array
    int offset = index * stride;

    // Create a new data vector for the subtensor
    std::vector<std::shared_ptr<Value>> newData(stride);

    // Copy the relevant data
    for (int i = 0; i < stride; ++i) {
        newData[i] = data_[offset + i];
    }

    // Return the new subtensor
    return Tensor(newDims, newData);
}

// Fill tensor with a specific value
void Tensor::fill(float value) {
    for (int i = 0 ; i < total_size_ ; ++i) {
        data_[i] = Value::create(value);
    }
}

// Get the underlying Eigen::Matrix
const std::vector<std::shared_ptr<Value>>& Tensor::mat() const {
    return data_;
}
std::vector<std::shared_ptr<Value>>& Tensor::mat() {
    return data_;
}

// Get the rank
int Tensor::rank() const {
    return dimensions_.size();
}

// Get dimensions of the tensor
const std::vector<int>& Tensor::dim() const {
    return dimensions_;
}

// Total size
int Tensor::size() const {
    return total_size_;
}

// Display tensor for debugging
void Tensor::display() const {
    std::cout << "Tensor: ";
    for (int i = 0; i < mat().size(); ++i) {
        std::cout << mat().data()[i]->getData() << " ";
    }
    std::cout << std::endl;
}

// Reshape the tensor
void Tensor::reshape(const std::vector<int>& new_dims) {
    // Compute the total size of the new dimensions
    int new_total_size = 1;
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

Tensor Tensor::slice(int start, int end, int axis) {
    // Validate the axis
    if (axis < 0 || axis >= dimensions_.size()) {
        throw std::invalid_argument("Invalid axis for slicing.");
    }

    // Validate the start and end indices
    if (start < 0 || end > dimensions_[axis] || start >= end) {
        throw std::invalid_argument("Invalid start or end indices for slicing.");
    }

    // Calculate the new dimensions after slicing
    std::vector<int> newDims = dimensions_; // copy
    newDims[axis] = end - start;

    // Prepare new data storage for the sliced tensor
    int newSize = 1;
    for (int dim : newDims) {
        newSize *= dim;
    }
    std::vector<std::shared_ptr<Value>> newData(newSize);

    // Calculate the strides for the original tensor
    std::vector<int> strides(dimensions_.size(), 1);
    for (int i = dimensions_.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dimensions_[i + 1];
    }

    // Calculate the offset to the start index along the slicing axis
    int baseOffset = start * strides[axis];

    // Iterate over the new tensor and fill the data
    for (int i = 0; i < newSize; ++i) {
        // Compute the corresponding index in the original tensor
        int newIndex = i;
        int originalIndex = baseOffset;
        for (int j = newDims.size() - 1; j >= 0; --j) {
            int idx = newIndex % newDims[j];
            originalIndex += idx * strides[j];
            newIndex /= newDims[j];
        }

        // Copy the data from the original tensor to the new tensor
        newData[i] = data_[originalIndex];
    }

    // Return the new sliced tensor
    return Tensor(newDims, newData);
}

void Tensor::setValueLike(Tensor& tensor) {
    if (dim() != tensor.dim()) {
        throw std::runtime_error("Tensor shape not equal");
    }
    int n = tensor.total_size_;
    for (int i = 0; i < n ; ++i) {
        float newData = tensor.data_[i]->getData();
        data_[i]->setValue(newData);
    }
}

// Static method to create a tensor filled with ones
Tensor Tensor::ones(const std::vector<int>& dims) {
    Tensor tensor(dims);
    for (int i = 0; i < tensor.size(); ++i) {
        tensor.mat()[i] = Value::create(1.0f);
    }
    return tensor;
}

// Static method to create a tensor filled with zeros
Tensor Tensor::zeros(const std::vector<int>& dims) {
    Tensor tensor(dims);
    for (int i = 0; i < tensor.size(); ++i) {
        tensor.mat()[i] = Value::create(0.0f);
    }
    return tensor;
}

// Static method to create a tensor filled with random values
Tensor Tensor::random(const std::vector<int>& dims, float min, float max) {
    Tensor tensor(dims);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (int i = 0; i < tensor.size(); ++i) {
        tensor.mat()[i] = Value::create(dis(gen));
    }
    return tensor;
}


// Static method to create a tensor filled with values from a randn distribution
Tensor Tensor::randn(const std::vector<int>& dims, float mean, float stddev) {
    Tensor tensor(dims);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);

    for (int i = 0; i < tensor.size(); ++i) {
        tensor.mat()[i] = Value::create(dis(gen));
    }
    return tensor;
}