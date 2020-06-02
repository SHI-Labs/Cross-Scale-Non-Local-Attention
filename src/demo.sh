# CSNLN Training Code
#python3 main.py --chop --batch_size 16 --model CSNLN --scale 2 --patch_size 96 --save CSNLN_x2 --n_feats 128 --depth 12 --data_train DIV2K --save_models

# CSNLN Test Code
python3 main.py --model CSNLN --data_test Set5+Set14+B100+Urban100 --data_range 801-900 --scale 2 --n_feats 128 --depth 12 --pre_train ../models/model_x2.pt --save_results --test_only --chop

