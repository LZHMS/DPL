import pickle
import argparse
import os
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default="16", help="the number of shots")
    parser.add_argument("--dataset", type=str, default="", help="dataset")
    parser.add_argument("--trainer", type=str, default="", help="trainer")
    parser.add_argument("--config", type=str, default="", help="config")
    parser.add_argument("--nctx", type=int, default=16, help="dataset")
    parser.add_argument(
        '--csc', action='store_true', help='class specific'
    )
    parser.add_argument(
        '--clsequle', action='store_true', help='class specific'
    )
    parser.add_argument(
        '--ctp', type=str, default='end', help='context token position'
    )
    parser.add_argument(
        '--tag', type=str, default='', help=''
    )
    
    args = parser.parse_args()

    results = []
    directory = os.path.join("output", f"{args.dataset}", f"{args.trainer}", f"{args.config}_{args.shots}shots_EQULE_{args.clsequle}__{args.tag}")
    for noise in [0, 2, 4, 8]:
        for seed in [1, 2, 3, 4]:
            for epoch in [10, 20, 30, 40, 50]:
                print(f"------------------- Analyse the noise {noise} of seed {seed} epoch {epoch}----------------------")
                noise_info_path = os.path.join(directory, f"nctx{args.nctx}_csc{args.csc}_ctp{args.ctp}_fp{noise}", f"seed{seed}",
                                           "NoiseAnalysis", f"noise_analysis_epoch{epoch}.pkl")
                with open(noise_info_path, "rb") as file:
                    noise_info = pickle.load(file)
                    total = 0
                    ready = True
                    n_cls = len(noise_info)
                    for i in range(n_cls):
                        total += len(noise_info[i])
                        length = len(noise_info[i])
                        clsname, label = None, None
                        tp, fp = 0, 0
                        for datum in noise_info[i]:
                            clsname = datum[1]
                            label = datum[0]
                            if datum[6]:
                                tp += 1
                            else:
                                fp += 1
                        print(f"* epoch {epoch} label: {label} name: {clsname} length: {length} tp: {tp} fp: {fp} pred noise:{fp/length:.3f} given noise:{noise/args.shots:.3f} *")
                        if fp != noise or length != args.shots:
                            ready = False
                    
                    results.append(f"* the noise {noise} of seed {seed} epoch {epoch}: {ready}")
                    print(f"* epoch {epoch} total: {total}, true total: {n_cls*args.shots} *")

    print("-------------------- Analysis the results together ---------------------------")
    for info in results:
        print(info)