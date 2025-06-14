import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box1, box2):
    """精确计算两个矩形框的交并比"""
    # 确保处理浮点数
    x1, y1, w1, h1 = map(float, box1)
    x2, y2, w2, h2 = map(float, box2)
    
    # 计算交集区域
    inter_left = max(x1, x2)
    inter_top = max(y1, y2)
    inter_right = min(x1 + w1, x2 + w2)
    inter_bottom = min(y1 + h1, y2 + h2)
    
    # 处理无交集情况
    inter_area = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)
    
    # 计算并集面积
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0
def plot_sequence_comparison(seq_name, tracker_results, output_dir):
    """为单个序列生成对比图并保存"""
    plt.figure(figsize=(20, 4))
    
    # 为每个追踪器绘制曲线（保持原样）
    for tracker_name, ious in tracker_results.items():
        if len(ious) == 0:
            continue
        
        # 动态计算移动平均窗口（保持原样）
        window_size = 3
        smoothed = np.convolve(ious, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(smoothed, 
                linewidth=3, 
                marker='', 
                label=f"{tracker_name} (MA{window_size})")
        plt.ylim(0, 1)
        plt.xlim(left=0)

    # 获取当前轴并设置坐标轴属性
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 设置刻度样式（修正版本）
    ax.tick_params(axis='both', which='major', 
                  labelsize=18,      # 增大字体尺寸到18pt
                  width=2,           # 刻度线宽度
                  length=6)          # 刻度线长度

    # 单独设置刻度标签加粗
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight("bold")

    # 设置网格样式（加粗版本）
    plt.grid(True, 
            linestyle='--', 
            linewidth=1.5,    # 加粗网格线到1.5pt
            alpha=0.7)

    # 移除默认坐标轴标签（保持原样）
    plt.xlabel("")
    plt.ylabel("")

    # 调整边距并保存（保持原样）
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{seq_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def process_all_sequences(gt_root, tracker_paths, output_dir="results"):
    """主处理函数"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有GT序列
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
            
            # 获取有效帧索引
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
                
                # 计算IOU序列
                iou_sequence = []
                for gt, res in zip(gt_boxes, res_boxes):
                    iou = compute_iou(gt, res)
                    iou_sequence.append(iou)
                tracker_results[tracker_name] = iou_sequence
                
        except Exception as e:
            print(f"Error processing {seq_folder}: {str(e)}")
            continue
        
        # 生成序列对比图
        plot_sequence_comparison(seq_folder, tracker_results, output_dir)

# 保持compute_iou函数与之前相同

if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    config = {
        "gt_root": "/root/code/dataset/DUT/test/",       # GT数据根目录
        "trackers": ["/root/code/new/trackers/SiamDT/results/DUT_test/dut", "/root/code/new/trackers/SiamDT/results/DUT_test/siam_ssm"],  # 追踪器结果路径列表
        "output_dir": "/root/code/new/trackers/SiamDT/pic_dut/"       # 输出目录
    }
    
    # 执行处理流程
    process_all_sequences(
        gt_root=config["gt_root"],

        tracker_paths=config["trackers"],
        output_dir=config["output_dir"]
    )