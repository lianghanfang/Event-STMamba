# Event-STMamba
A spatial-temporal guided mamba for event segmentation and detection

# 孩子不懂事写着玩的 效果一坨 因为在windows上没有hais的cuda算子
# 所以是手动挡 会占用大量内存 （cpu汗流浃背了）
# 等啥时候用服务器再跑好了

顺便说一下 用的数据集是
https://github.com/ChenYichen9527/EV-UAV/tree/main

浅尝了一下可学习的空间引导的mamba 
 <br 是3D贪吃蛇罢（不是） />

# 如果有勇士（美食家）愿意品鉴 ↓ 下方是品鉴流程

使用方法
## 1. 预计算缓存（推荐先运行）
   
python cache_precompute.py --data-root ./data --memory-optimized

python main.py --mode cache --data-root ./data --memory-optimized

## 2. 训练模型

python main.py --mode train --data-root ./data --memory-optimized

python main.py --mode train --checkpoint ./checkpoints/checkpoint_epoch_10.pth --memory-optimized

## 3. 测试模型

python main.py --mode test --checkpoint ./checkpoints/best_iou.pth --data-root ./data

## 4. 完整流程

python main.py --mode all --data-root ./data --memory-optimized



