import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
        
# class Conv1D(nn.Module):
#     def __init__(self, input_size,  output_size):
#         super(Conv1D, self).__init__()
#         self.conv1D = nn.Conv1d(input_size, output_size, 3, stride=1, padding=1)
#         # self.linear = nn.Linear(hidden_size, output_size)
#     def forward(self,input):
#         output = self.conv1D(input)
#         # output = self.linear(out)
#         return output
        


