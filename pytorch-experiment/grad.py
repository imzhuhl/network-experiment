import torch

def test_grad():
    # x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float, requires_grad=True)
    # w = torch.tensor([1, 2, 3, 4], dtype=torch.float, requires_grad=True)
    x = torch.tensor([2, 3], dtype=torch.float, requires_grad=True)
    w = torch.tensor([5], dtype=torch.float, requires_grad=True)
    y = w * x
    print(y)
    y = y.mean()
    print(y)
    y.backward()
    print(x.grad)
    print(w.grad)


if __name__ == '__main__':
    test_grad()
