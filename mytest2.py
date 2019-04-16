import torch
import time
import threading 

device = torch.device('cuda:0')
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s3 = torch.cuda.Stream()

with torch.cuda.stream(s3):
    gru = torch.nn.GRU(input_size=300, hidden_size=64, num_layers=1, bias=True, bidirectional=True, batch_first=True).to(device)
    h0 = torch.randn(2, 4, 64, requires_grad=True).to(device)
    x1 = torch.randn(4,100,300).to(device)
    x2 = torch.randn(4,100,300).to(device)


torch.cuda.synchronize()

def run1(x):
    with torch.cuda.stream(s1):
        for _ in range(10):
            y1, hn1 = gru(x1, h0)

def run2(x):
    with torch.cuda.stream(s2):
        for _ in range(10):
            y2, hn2 = gru(x2, h0)

# creating thread 
t1 = threading.Thread(target=run1, args=(1,)) 
t2 = threading.Thread(target=run2, args=(1,)) 

# starting thread 1 
t1.start() 
# starting thread 2 
t2.start() 

# wait until thread 1 is completely executed 
t1.join() 
# wait until thread 2 is completely executed 
t2.join() 

# both threads completely executed 
print("Done!")
