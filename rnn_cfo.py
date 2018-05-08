'''
RNN regression
compensate for CFO
'''

import torch
from torch import nn
#import numpy as np
from matplotlib import pyplot as plt
from scipy import io
#%%
# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
HIDDEN = 8
LR = 0.01            # learning rate
PI = 3.1416

# load data
data_read = io.loadmat('symbCFOconst.mat')
sig_phase = torch.from_numpy(data_read['sigPhase']).view(-1, 1, 1)
ref_phase = torch.from_numpy(data_read['refPhase']).view(-1, 1)
sig_phase_np = sig_phase.numpy()
ref_phase_np = ref_phase.numpy()
plt.plot(sig_phase_np.squeeze()[-2000:-1:50], 'ro')
plt.plot(ref_phase_np.squeeze()[-2000:-1:50], 'bo')
plt.show()
#%%
rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
        )
lin = nn.Linear(HIDDEN, 1)
x = sig_phase[0:50,::]
prediction, h_state = rnn(x, None)
est = lin(h_state)
print(prediction.shape, est.shape)

#%%
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
        )
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        est = self.out(h_state)
        return est, h_state
        
#        r_out = r_out.view(-1, HIDDEN)
#        outs = self.out(r_out)
#        return outs, h_state

rnn = RNN()


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state

#plt.figure(1, figsize=(12, 5))
#plt.ion()           # continuously plot
#%%
loss_np = []
for step in range(500):
#    start, end = step * np.pi, (step+1)*np.pi   # time range
#    # use sin predicts cos
#    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
#    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
#    y_np = np.cos(steps)
#
#    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
#     # shape (batch, time_step, input_size)
##     y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
#    y = (torch.from_numpy(y_np[:, np.newaxis]))

    x = sig_phase[step*TIME_STEP:(step+1)*TIME_STEP]
    y = ref_phase[step*TIME_STEP:(step+1)*TIME_STEP]
    prediction, h_state = rnn(x, h_state)   # rnn output
    y_est = (torch.ones_like(x)*prediction+x)%(2*PI)-PI

    h_state = h_state.detach()

    loss = loss_func(y_est[::,0], y)         # cross entropy loss
    loss_np.append(loss.detach().numpy())
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
#    plt.plot(steps, y_np.flatten(), 'r-')
#    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
#    plt.draw()
#    plt.pause(0.1)
#
#plt.ioff()
#plt.show()
prediction_np = prediction.detach().numpy()
h_state_np = h_state.numpy()
    
plt.plot(loss_np)
plt.show()

#%%
plt.plot(y_est.detach().squeeze().numpy(),'ro')
plt.show()

