import torch


if __name__ == '__main__':

    old_ckp_file = '/root/code/new/trackers/SiamDT/pretrained_weights/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_s_epoch_31 (1).pth'

    old_state_dict = torch.load(old_ckp_file)['state_dict']

    from collections import OrderedDict

    new_state_dict = OrderedDict()

    # 只保留backbone和neck的值
    for key, value in old_state_dict.items():

        print(key)
        if key[0:8] == 'backbone':
            new_state_dict[key] = value
        if key[0:4] == 'neck':
            new_state_dict[key] = value


    # open in torch 1.4.0
    torch.save(new_state_dict,
               'pretrained_weights/siam_ssm_tiny.pth.tar',
               _use_new_zipfile_serialization=False)

