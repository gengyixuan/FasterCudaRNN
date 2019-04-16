import torch
import time

device = torch.device('cuda:0')

batch_size = 32
N = 10000
gru = torch.nn.GRU(input_size=300, hidden_size=64, num_layers=1, bias=True, bidirectional=True, batch_first=True).to(device)
h0 = torch.randn(2, batch_size, 64, requires_grad=True).to(device)
x1 = torch.randn(batch_size,N,300).to(device)
x2 = torch.randn(batch_size,N,300).to(device)
x3 = torch.randn(batch_size,N,300).to(device)


s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s3 = torch.cuda.Stream()
with torch.cuda.stream(s1):
    for _ in range(100):
        y1, hn1 = gru(x1, h0)

with torch.cuda.stream(s2):
    for _ in range(100):
        y2, hn2 = gru(x2, h0)

with torch.cuda.stream(s3):
    for _ in range(100):
        y3, hn3 = gru(x3, h0)
