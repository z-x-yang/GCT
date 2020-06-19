首先把下面这些sh文件里面的DATA_DIR改成imagenet的tf_record的路径。
然后用resnet50.sh，resnet101.sh，resnet152.sh来train。分别对应3个不同的实验。
train的过程需要4张v100，2~3天。
train完之后用eval_resnet50.sh，eval_resnet101.sh，eval_resnet152.sh来eval。
