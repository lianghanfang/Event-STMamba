# Event-STMamba
A spatial-temporal guided mamba for event segmentation and detection

使用方法
1. 预计算缓存（推荐先运行）
bash# 使用独立脚本预计算所有数据集的缓存
python cache_precompute.py --data-root ./data --memory-optimized

# 或者使用main.py
python main.py --mode cache --data-root ./data --memory-optimized
2. 训练模型
bash# 使用内存优化配置进行训练
python main.py --mode train --data-root ./data --memory-optimized

# 从检查点继续训练
python main.py --mode train --checkpoint ./checkpoints/checkpoint_epoch_10.pth --memory-optimized
3. 测试模型
bash# 测试模型
python main.py --mode test --checkpoint ./checkpoints/best_iou.pth --data-root ./data
4. 完整流程
bash# 运行完整的缓存-训练-测试流程
python main.py --mode all --data-root ./data --memory-optimized
