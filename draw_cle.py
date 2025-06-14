import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def compute_cle(box1, box2):
    """计算两个矩形框的中心点欧氏距离"""
    x1, y1, w1, h1 = map(float, box1)
    x2, y2, w2, h2 = map(float, box2)
    
    # 计算中心点坐标
    center_x1 = x1 + w1/2
    center_y1 = y1 + h1/2
    center_x2 = x2 + w2/2
    center_y2 = y2 + h2/2
    
    # 计算欧氏距离
    return np.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)

def plot_sequence_comparison(seq_name, tracker_results, output_dir):
    """生成CLE对比图（只显示20像素以下数据）"""
    plt.figure(figsize=(20, 4))
    
    # 为每个追踪器绘制曲线
    for tracker_name, cles in tracker_results.items():
        if len(cles) == 0:
            continue
        
        # 过滤超过20像素的数据
        filtered_cles = [x for x in cles]
        
        # 移动平均窗口
        window_size = 4
        smoothed = np.convolve(filtered_cles, np.ones(window_size)/window_size, 
                             mode='valid', )
        
        # 绘制曲线
        plt.plot(smoothed, 
                linewidth=3,
                color='red',
                marker='',
                label=f"{tracker_name} (MA{window_size})")
        ax = plt.gca()
        ax.set_ylim(0, 20)  # 固定y轴范围0-20
        


    # 坐标轴设置
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 刻度样式设置
    ax.tick_params(axis='both', which='major',
                  labelsize=16,    # 增大字体尺寸
                  width=2,         # 加粗刻度线
                  length=6)        # 加长刻度线
    
    # 刻度标签加粗
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight("bold")
    
    # 坐标范围设置
    plt.ylim(0, 20)  # 固定y轴范围
    plt.xlim(left=0)
    
    # 添加参考线
    plt.axhline(y=15, color='g', linestyle='--', linewidth=4, alpha=0.8)
    
    # 网格设置
    plt.grid(True, 
            linestyle='--', 
            linewidth=1.5,
            alpha=0.7)
    
    # 图例设置
    plt.legend(fontsize=12, loc='upper right')
    
    # 保存图像
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{seq_name}_CLE20.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def process_all_sequences(gt_root, tracker_paths, output_dir="results"):
    """主处理函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    for seq_folder in sorted(os.listdir(gt_root)):
        seq_path = os.path.join(gt_root, seq_folder)
        if not os.path.isdir(seq_path):
            continue
        
        # 初始化结果容器
        tracker_results = {os.path.basename(p): [] for p in tracker_paths}
        
        try:
            # 读取GT标注
            with open(os.path.join(seq_path, "IR_label.json")) as f:
                gt_data = json.load(f)
            
            # 获取有效帧
            valid_indices = [i for i, e in enumerate(gt_data["exist"]) if e == 1]
            gt_boxes = [list(map(float, gt_data["gt_rect"][i])) for i in valid_indices]
            
            # 处理每个追踪器
            for tracker_path in tracker_paths:
                tracker_name = os.path.basename(tracker_path)
                res_file = os.path.join(tracker_path, f"{seq_folder}.txt")
                
                if not os.path.exists(res_file):
                    continue
                
                # 读取追踪结果
                with open(res_file) as f:
                    res_data = json.load(f)
                
                # 对齐有效帧
                res_boxes = [list(map(float, res_data["res"][i])) for i in valid_indices]
                
                # 计算CLE序列
                cle_sequence = []
                for gt, res in zip(gt_boxes, res_boxes):
                    cle = compute_cle(gt, res)
                    cle_sequence.append(cle)
                tracker_results[tracker_name] = cle_sequence
                
        except Exception as e:
            print(f"Error processing {seq_folder}: {str(e)}")
            continue
        
        # 生成对比图
        plot_sequence_comparison(seq_folder, tracker_results, output_dir)


if __name__ == "__main__":
    config = {
        "gt_root": "/root/code/dataset/DUT/test/",
        "trackers": [
            "/root/code/new/trackers/SiamDT/results/DUT_test/dut/",
            ""
        ],
        "output_dir": "/root/code/new/trackers/SiamDT/pic_cle_dut/"
    }
    
    process_all_sequences(
        gt_root=config["gt_root"],
        tracker_paths=config["trackers"],
        output_dir=config["output_dir"]
    )