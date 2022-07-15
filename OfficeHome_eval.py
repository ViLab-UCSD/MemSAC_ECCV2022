import os

# source = ["Art" , "Clipart" , "Product" , "Real_World"]
source = ["real" , "clipart" ,"painting", "sketch" ]
target = source

for s in source:
    for t in target:
        # if os.path.exists("snapshot/domainNet/CDAN_%s_%s/best_model.pth.tar"%(s,t)):
        if s!=t:
            print("#"*10)
            print("Source: %s, Target: %s"%(s, t))
            print("Adaptation from %s to %s"%(s,t))
            cmd = "python fg_eval.py --gpu_id 7 --dataset %s --nClasses 126 --checkpoint snapshot/domainNet/CDAN_%s%s/best_model.pth.tar --data_dir /newfoundland/tarun/datasets/Adaptation --batch_size 40"%(t,s,s)
            os.system(cmd)
            print("#"*10)
        else:
            print("$"*20)
