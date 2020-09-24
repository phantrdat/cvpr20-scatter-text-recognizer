import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .sequence_modeling import BidirectionalLSTM
from .selective_decoder import SelectiveDecoder
class SelectiveContextualRefinementBlock(nn.Module):
    def __init__(self,visual_size, hidden_size, num_class):
        super(SelectiveContextualRefinementBlock, self).__init__()
        self.sequence_modeling = nn.Sequential(
            BidirectionalLSTM(visual_size, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, visual_size))

        self.sequence_modeling_output = visual_size
        self.selective_decoder = SelectiveDecoder(2*visual_size, hidden_size, num_class)
    def forward(self, visual_feature, contextual_feature, attn_text, is_train, batch_max_length):
        contextual_feature = self.sequence_modeling(contextual_feature)
        D = torch.cat((contextual_feature, visual_feature), 2)
        block_pred = self.selective_decoder(D, attn_text, is_train, batch_max_length=batch_max_length)
        
        return contextual_feature, block_pred


