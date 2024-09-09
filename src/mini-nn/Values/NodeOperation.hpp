#pragma once
#include <memory>
#include <vector>
#include <functional>

// Forward declaration of the Value class
class Value;

namespace NodeOperation {

    class Operation {
    public:
        virtual void forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) = 0;
        virtual void backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) = 0;
        virtual ~Operation() = default;
    };

    class Add : public Operation {
    public:
        void forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        void backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        virtual ~Add() = default;
    };

    class Avg : public Operation {
    public:
        void forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        void backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        virtual ~Avg() = default;
    };

    class Mul : public Operation {
    public:
        void forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        void backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        virtual ~Mul() = default;
    };

    class Sub : public Operation {
    public:
        void forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        void backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        virtual ~Sub() = default;
    };

    class Div : public Operation {
    public:
        void forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        void backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        virtual ~Div() = default;
    };

    class Function1 : public Operation {
    private:
        std::function<float(float)> forward_;
        std::function<float(float)> backward_;

    public:
        Function1(std::function<float(float)> forward, std::function<float(float)> backward);
        void forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        void backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) override;
        virtual ~Function1() = default;
    };

} // namespace NodeOperation
