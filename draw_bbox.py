import os
import json
import cv2

def draw_boxes(image_dir, gt_json_path, result_b_path, result_c_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载ground truth数据
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    gt_exist = gt_data['exist']
    gt_rects = gt_data['gt_rect']

    # 加载结果文件B和C
    def load_result_txt(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return [list(map(float, box)) for box in data['res']]
    
    res_b = load_result_txt(result_b_path)
    res_c = load_result_txt(result_c_path)

    # 获取排序后的图片列表
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')],
                         key=lambda x: int(x.split('.')[0]))
    
    # 验证数据一致性
    assert len(image_files) == len(gt_exist) == len(gt_rects) == len(res_b) == len(res_c), \
           "数据长度不一致，请检查输入文件"

    # 颜色定义（BGR格式）
    colors = {
        'A': (0, 0, 255),     # 红色
        'B': (255, 0, 0),     # 蓝色
        'C': (0, 165, 255)    # 橙色
    }

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        # 绘制A的框（ground truth）
        if gt_exist[idx] == 1:
            x, y, w, h = map(int, gt_rects[idx])
            cv2.rectangle(img, (x, y), (x+w, y+h), colors['A'], 2)

        # 绘制B的框
        x_b, y_b, w_b, h_b = map(int, map(round, res_b[idx]))
        cv2.rectangle(img, (x_b, y_b), (x_b+w_b, y_b+h_b), colors['B'], 2)

        # 绘制C的框
        x_c, y_c, w_c, h_c = map(int, map(round, res_c[idx]))
        cv2.rectangle(img, (x_c, y_c), (x_c+w_c, y_c+h_c), colors['C'], 2)

        # 保存结果
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, img)
        print(f"已处理: {img_file}")

if __name__ == "__main__":
    # 输入路径配置（请根据实际情况修改）
    PATH_A = "/root/code/data/test/03_2499_0962-2461"               # 数据集路径（包含图片和IR_label.json）
    FILE_B = "/root/code/new/trackers/SiamDT/results/UAVtir_test/siam_ssm_677/03_2499_0962-2461.txt"             # 算法B结果文件
    FILE_C = "/root/code/new/trackers/SiamDT/results/UAVtir_test/siam_ssm_677/03_2499_0962-2461.txt"             # 算法C结果文件
    OUTPUT_DIR = "/root/code/new/trackers/SiamDT/03_2499_0962-2461_siam_ssm_677"    # 输出目录

    # 执行绘制函数
    draw_boxes(
        image_dir=PATH_A,
        gt_json_path=os.path.join(PATH_A, "IR_label.json"),
        result_b_path=FILE_B,
        result_c_path=FILE_C,
        output_dir=OUTPUT_DIR
    )

    """config = {
        "gt_root": "/root/code/dataset/DUT/test/",
        "trackers": [
            "/root/code/new/trackers/SiamDT/results/DUT_test/dut/",
            ""
        ],
        "output_dir": "/root/code/new/trackers/SiamDT/pic_cle_dut/"
    }
    PATH_A = "/root/code/data/test/20190925_111757_1_4"                # 数据集路径（包含图片和IR_label.json）
    FILE_B = "/root/code/new/trackers/SiamDT/results/UAVtir_test/single/20190925_111757_1_4.txt"             # 算法B结果文件
    FILE_C = "/root/code/new/trackers/SiamDT/results/UAVtir_test/siam_ssm_677/20190926_111509_1_8.txt"             # 算法C结果文件
    OUTPUT_DIR = "/root/code/new/trackers/SiamDT/demo_410"    # 输出目录
    """
    
