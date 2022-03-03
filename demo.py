import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from utils import CTCLabelConverter, AttnLabelConverter, TransLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    # print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
    #       opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
    #       opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()


    with torch.no_grad():    
        for image_tensors, image_path_list in demo_loader:
            all_pred_strs = []
            all_confidence_scores = []
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            predss = model(image, text_for_pred, is_train=False)[0]
            

            for i, preds in enumerate(predss):
                confidence_score_list = []
                pred_str_list = []

                # select max probability (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_str_list.append(pred)
                    pred_max_prob = pred_max_prob[:pred_EOS]
                    
                    # calculate confidence score (= multiply of pred_max_prob)
                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1].cpu().numpy()
                    except:
                        confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                    confidence_score_list.append(confidence_score)
                            
                all_pred_strs.append(pred_str_list)
                all_confidence_scores.append(confidence_score_list)
            
            all_confidence_scores =  np.array(all_confidence_scores)
            all_pred_strs = np.array(all_pred_strs)

            best_pred_index = np.argmax(all_confidence_scores, axis=0)
            best_pred_index = np.expand_dims(best_pred_index, axis=0)

            # Get max predition per image through blocks
            all_pred_strs = np.take_along_axis(all_pred_strs, best_pred_index, axis=0)[0]
            all_confidence_scores = np.take_along_axis(all_confidence_scores, best_pred_index, axis=0)[0]
            
            log = open(f'./log_demo_result.txt', 'w')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
                    
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')
            for img_name, pred, confidence_score in zip(image_path_list, all_pred_strs, all_confidence_scores):
                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='test_image/normal_cases', required=False, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--saved_model', default='weights/scatter-case-sensitive.pth', required=False,  help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default = '''0123456789abcdefghijklmnopqrstuvwxyz''', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=512, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
