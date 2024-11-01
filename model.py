import torch

class LinearRegression:
    def __init__(self):
        self.w0 = torch.randn(1)
        self.w1 = torch.randn(1)
    
    def __call__(self, input):
        prediction = input * self.w1 + self.w0
        return prediction
    
    def __repr__(self):
        return "Linear Regression"

def gradient_descent(input, output, prediction, learning_rate, w0, w1):
    # Gradient
    grad_w0 = 2*torch.mean(prediction - output)
    grad_w1 = 2*torch.mean(input*(prediction - output))
    
    # Update
    w0 -= learning_rate * grad_w0
    w1 -= learning_rate * grad_w1
    
    return w0, w1