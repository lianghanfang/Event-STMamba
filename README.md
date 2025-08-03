# Event-STMamba
A spatial-temporal guided mamba for event segmentation and detection

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

