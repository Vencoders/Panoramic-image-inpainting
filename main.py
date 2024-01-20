

from train import *

def setup_seed(seed=3047):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="test", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")
parser.add_argument('--data_parallel', action='store_true')

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=2, type=int, dest="batch_size") #128->24, 256->8, new2_256->10
parser.add_argument("--num_epoch", default=2000, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="dataset", type=str, dest="data_dir")

parser.add_argument("--ckpt_dir", default="./checkpoint_test", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log_test_0.1", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result_test_0.1", type=str, dest="result_dir")

parser.add_argument("--input_img", default="./test/image/0006_test.png", type=str, dest="input_img")
parser.add_argument("--input_msk", default="./test/mask/0006_mask.png", type=str, dest="input_msk")
parser.add_argument("--outdir_json", default="./test", type=str, dest="outdir_json")

parser.add_argument("--task", default="PIUnet", choices=['PInet','PIUnet'], type=str, dest="task")

parser.add_argument("--nker", default=32, type=int, dest="nker")
parser.add_argument("--norm",default="bnorm",type=str,dest="norm")

parser.add_argument("--network", default="PIUnet", choices=['PInet','PIUnet'], type=str, dest="network")

args = parser.parse_args()

if __name__ == "__main__":
    setup_seed(3047)
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)

