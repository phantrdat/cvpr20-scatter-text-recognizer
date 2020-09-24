"""
Author: Trong-Dat Phan
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.visual_feature_refinement import Refinement
from modules.selective_contextual_refinement_block import SelectiveContextualRefinementBlock
import torch
class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        
        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)

        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        """ Visual Features Refinement """

        self.Refiner = Refinement(opt.output_channel, opt.num_class)

        """  Contextual Refinement Block """
        self.ctx_block1 = SelectiveContextualRefinementBlock(self.FeatureExtraction_output, opt.hidden_size, opt.num_class)
        self.ctx_block2 = SelectiveContextualRefinementBlock(self.FeatureExtraction_output, opt.hidden_size, opt.num_class)
        self.ctx_block3 = SelectiveContextualRefinementBlock(self.FeatureExtraction_output, opt.hidden_size, opt.num_class)
        self.ctx_block4 = SelectiveContextualRefinementBlock(self.FeatureExtraction_output, opt.hidden_size, opt.num_class)
        self.ctx_block5 = SelectiveContextualRefinementBlock(self.FeatureExtraction_output, opt.hidden_size, opt.num_class)
        

    def forward(self, input, attn_text, is_train=True):
        """ Transformation stage """
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Refinement branch """
        refiner = self.Refiner(visual_feature.contiguous())

        # Selective Contextual Refinement Block 
        contextual_feature, block_pred1 = self.ctx_block1(visual_feature, visual_feature, attn_text, is_train, self.opt.batch_max_length)
        contextual_feature, block_pred2 = self.ctx_block2(visual_feature, contextual_feature, attn_text, is_train, self.opt.batch_max_length)
        contextual_feature, block_pred3 = self.ctx_block3(visual_feature, contextual_feature, attn_text, is_train, self.opt.batch_max_length)
        contextual_feature, block_pred4 = self.ctx_block4(visual_feature, contextual_feature, attn_text, is_train, self.opt.batch_max_length)
        _, block_pred5 = self.ctx_block5(visual_feature, contextual_feature, attn_text, is_train, self.opt.batch_max_length)

        return (block_pred1, block_pred2, block_pred3, block_pred4, block_pred5), refiner
