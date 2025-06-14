import init_paths
import libs.data as data
from trackers import *

# conda activate SiamDT
# python -m visdom.server -port=5123
# python tracking_test_demo.py

if __name__ == '__main__':
    cfg_file = 'configs/dt.py' #test no_rpn baseline+mla
    ckp_file = '/root/code/new/trackers/SiamDT/work_dirs/new/epoch_22.pth'
    name_suffix = cfg_file[8:-3]
    visualize = False
    # selected_seq='02_6321_0274-2773'
    selected_seq = 'ALL'

    transforms = data.BasicPairTransforms(train=False)
    tracker = SiamDTTracker(
        cfg_file, ckp_file, transforms,
        name_suffix=name_suffix, visualize=visualize)

    evaluators = [data.EvaluatorDT(root_dir='/root/code/dataset/DT/', subset='train')]#data.EvaluatorDUT(root_dir='/root/code/dataset/DUT/', subset='Anti-UAV-Tracking-V0')]
    #data.EvaluatorDUT(root_dir='/root/code/dataset/DUT/', subset='test')]
        #data.EvaluatorUAVtir(root_dir='/root/code/data/', subset='val')]
        #data.EvaluatorDUT(root_dir='/root/code/dataset/DUT/', subset='test')]
        #data.EvaluatorantiUAV(root_dir='/root/code/dataset/antiuav/infrared/', subset='test')]
        #data.EvaluatorantiUAVV(root_dir='/root/code/dataset/antiuav/visible/', subset='test')]
        #data.EvaluatorantiUAV(root_dir='/root/code/dataset/antiuav/infrared/', subset='test')
        #data.EvaluatorUAVtir(root_dir='/root/code/data/', subset='test')] #/root/code/dataset/antiuav/infrared/   /root/code/data/  /root/code/dataset/antiuav/visible/

    for e in evaluators:
        e.run(tracker, selected_seq=selected_seq)
