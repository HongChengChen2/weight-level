剪枝前 迁移训练：
python trans_prune_0.py --arch alexnet --pretrained --epochs 60 --save /home/leander/hcc/prunWeight/save/ /home/leander/hcc/prunWeight/data

第一次剪枝
python trans_prune_1.py --arch alexnet --percent 0.9 --resume /home/leander/hcc/prunWeight/save/0911/data3/pruned0c.pth.tar  --save /home/leander/hcc/prunWeight/save/0911/data3 /home/leander/hcc/prunWeight/data3

python trans_main_finetune_1.py --arch alexnet -b 64  --epochs 50 --resume /home/leander/hcc/prunWeight/save/0911/data3/pruned1c.pth.tar --save /home/leander/hcc/prunWeight/save/0911/data3 /home/leander/hcc/prunWeight/data3

第二次剪枝
python trans_prune_2.py --arch alexnet  --percent 0.75 --resume /home/leander/hcc/prunWeight/save/0911/data3/scratch1c.pth.tar  --save /home/leander/hcc/prunWeight/save/0911/data3 /home/leander/hcc/prunWeight/data3

python trans_main_finetune_2.py --arch alexnet -b 64  --epochs 50 --resume /home/leander/hcc/prunWeight/save/0911/data3/pruned2c.pth.tar --save /home/leander/hcc/prunWeight/save/0911/data3 /home/leander/hcc/prunWeight/data3

第三次剪枝
python trans_prune_3.py --arch alexnet --percent 0.6 --resume /home/leander/hcc/prunWeight/save/0911/data3/scratch2c.pth.tar  --save /home/leander/hcc/prunWeight/save/0911/data3 /home/leander/hcc/prunWeight/data3

python trans_main_finetune_3.py --arch alexnet -b 64  --epochs 50 --resume /home/leander/hcc/prunWeight/save/0911/data3/pruned3c.pth.tar --save /home/leander/hcc/prunWeight/save/0911/data3 /home/leander/hcc/prunWeight/data3

第四次剪枝
python trans_prune_4.py --arch alexnet --percent 0.55 --resume /home/leander/hcc/prunWeight/save/new/data1/scratch3c.pth.tar  --save /home/leander/hcc/prunWeight/save/new/data1 /home/leander/hcc/prunWeight/data1

python trans_main_finetune_4.py --arch alexnet -b 64  --epochs 50 --resume /home/leander/hcc/prunWeight/save/new/data1/pruned4c.pth.tar --save /home/leander/hcc/prunWeight/save/new/data1 /home/leander/hcc/prunWeight/data1