import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# Kiểm tra xem GPU có sẵn không
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Nếu sử dụng GPU, hiển thị thông tin
if device.type == 'cuda':
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"Bộ nhớ GPU đã cấp phát: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Bộ nhớ GPU đã cache: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    print(f"GPU có sẵn: {torch.cuda.is_available()}")
    print(f"Số lượng GPU: {torch.cuda.device_count()}")

# Tạo dữ liệu giả lập đơn giản
np.random.seed(42)
X = np.random.randn(1000, 20).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

# Chuyển đổi thành tensor PyTorch
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y).view(-1, 1)

# Tạo dataset và dataloader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Định nghĩa một mô hình đơn giản
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Khởi tạo mô hình
model = SimpleModel(input_dim=20)

# Di chuyển mô hình đến GPU nếu có sẵn
model.to(device)
print(f"Mô hình đang ở thiết bị: {next(model.parameters()).device}")

# Định nghĩa hàm mất mát và tối ưu hóa
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hàm huấn luyện
def train(num_epochs):
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Di chuyển dữ liệu đến GPU nếu có sẵn
            data, target = data.to(device), target.to(device)
            
            # Đưa gradient về 0
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Tính loss
            loss = criterion(outputs, target)
            
            # Backward pass và optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # In thông tin sau mỗi epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')
    
    end_time = time.time()
    print(f'Thời gian huấn luyện: {end_time - start_time:.2f} giây')

# Huấn luyện mô hình
print("Bắt đầu huấn luyện...")
train(num_epochs=10)

# Đánh giá mô hình
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = (outputs >= 0.5).float()
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    print(f'Độ chính xác: {100 * correct / total:.2f}%')

# Thêm một hàm kiểm tra tốc độ GPU vs CPU
def compare_speed():
    # Tạo tensor lớn
    size = 5000
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    # Kiểm tra trên CPU
    start = time.time()
    c_cpu = torch.matmul(a, b)
    cpu_time = time.time() - start
    print(f"Thời gian tính toán trên CPU: {cpu_time:.4f} giây")
    
    # Kiểm tra trên GPU nếu có sẵn
    if torch.cuda.is_available():
        a_gpu = a.to('cuda')
        b_gpu = b.to('cuda')
        
        # Khởi động GPU
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Đảm bảo tính toán GPU hoàn tất
        gpu_time = time.time() - start
        
        print(f"Thời gian tính toán trên GPU: {gpu_time:.4f} giây")
        print(f"GPU nhanh hơn CPU: {cpu_time/gpu_time:.1f}x")

print("\nKiểm tra tốc độ GPU vs CPU:")
compare_speed()