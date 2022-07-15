import os

# source = ["Art" , "Clipart" , "Product" , "Real_World"]
target = ["real" , "clipart" ,"painting", "sketch"]
source = list(target[:])

accs = []

for s in source:
    for t in target:
        if s!= t:
            # if os.path.exists("snapshot/domainNet/CDAN_%s_%s/best_model.pth.tar"%(s,t)):
            print("#"*10)
            print("Source Only from %s to %s"%(s,t))
            cmd = "export CUDA_VISIBLE_DEVICES=7 ; python eval.py --gpu_id 7 --nClasses 345 --checkpoint snapshot/domainNet_full_ablation/MemSAC_%s%s_QS_48000_BS_32_tau_0-07_lambda_0-1_CAS/best_model.pth.tar --data_dir /newfoundland/tarun/datasets/Adaptation/visDA --batch_size 64 --dataset domainNet_full --target %s"%(s,s,t)
            os.system(cmd)
            print("\n")
