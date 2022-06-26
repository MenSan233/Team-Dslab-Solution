import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='pvt_v2_b2', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument("--path", type = str, default = '/home/zhengll2021/synData/Deep-Wilcoxon-signed-rank-test-for-Covid-19-classfication-master/data/')
    #/home/zhengll2021/synData/Deep-Wilcoxon-signed-rank-test-for-Covid-19-classfication-master/data/
    parser.add_argument("--train_batch", type = int, default = 8)
    parser.add_argument("--train_ct_batch", type = int, default = 1)
    parser.add_argument("--val_batch", type = int, default = 8)
    parser.add_argument("--epoch", type = int, default = 1000)
    parser.add_argument("--val_epoch", type = int, default = 10)
    parser.add_argument("--lr", type = float, default = 0.000002)  # 0.00001  
    
    args = parser.parse_args([])
    return args