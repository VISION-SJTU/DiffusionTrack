import numpy as np
import os
import shutil
import argparse
import _init_paths


def transform_got10k(file_path):

    src_dir = os.path.join(file_path, "got10k" )
    dest_dir = os.path.join(file_path, "got10k_submit" )
    #if not os.path.exists(dest_dir):
      #  os.makedirs(dest_dir)
    print('src_dir: ' + src_dir)
    items = os.listdir(src_dir)
    for item in items:
        if "all" in item:
            continue
        src_path = os.path.join(src_dir, item)
        if "time" not in item:
            seq_name = item.replace(".txt", '')
            seq_dir = os.path.join(dest_dir, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            new_item = item.replace(".txt", '_001.txt')
            dest_path = os.path.join(seq_dir, new_item)
            bbox_arr = np.loadtxt(src_path, dtype=np.int64, delimiter='\t')
            np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')
        else:
            seq_name = item.replace("_time.txt", '')
            seq_dir = os.path.join(dest_dir, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            dest_path = os.path.join(seq_dir, item)
            os.system("cp %s %s" % (src_path, dest_path))
    # make zip archive
    shutil.make_archive(src_dir, "zip", src_dir)
    shutil.make_archive(dest_dir, "zip", dest_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transform got10k results.')
    parser.add_argument('--path', type=str,
                        default= '/home/xyren/data/fei/save_pth/diffusiontrack/6h/tracking_results/diffusiontrack', help='Name of config file.')

    args = parser.parse_args()
    files = os.listdir(args.path)
    files.sort()
    for i in range(len(files)):
         file_path = os.path.join(args.path, files[i])
         print(file_path)
         transform_got10k(file_path)
         print('done')

