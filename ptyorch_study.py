# filename: basic_pytorch_flow.py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 1. 构造一个简单的数据集 y = 2x + 1 + 噪声
class LinearDataset(Dataset):
    def __init__(self, n_samples=100):
        super().__init__()
        self.x = torch.linspace(-1, 1, n_samples).unsqueeze(1)  # shape [100, 1]
        self.y = 2 * self.x + 1 + 0.1 * torch.randn_like(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 2. 定义一个简单的线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # y = wx + b

    def forward(self, x):
        return self.linear(x)

def main():
    # 数据准备
    dataset = LinearDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 模型、损失函数、优化器
    model = LinearModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 训练
    for epoch in range(5):  # 训练5轮
        for x_batch, y_batch in dataloader:
            # 前向传播
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 测试：打印参数
    w, b = model.linear.weight.item(), model.linear.bias.item()
    print(f"Learned parameters: w={w:.2f}, b={b:.2f}")

if __name__ == "__main__":
    main()
