
import os
from builtins import print
import os
import argparse
import ffmpeg
import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
import logging
logging.basicConfig(filename='mosei.log', level=logging.DEBUG)




import os
import argparse
import ffmpeg
import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
import logging
logging.basicConfig(filename='mosei.log', level=logging.DEBUG)
# logging.debug('This message should go to the log file')
# logging.info('So should this')
# logging.warning('And this, too')
# logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

# multiprocessing.freeze_support()

#MOSEI 
# root1 = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented/Combined'
# root2 = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented/compress_seg'
# root1 = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented/Combined'
# root2 = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented/Frames1'

# mosi extract
# root1 = '/disk4/CMU-RAW/MOSI/Raw/Video/compress_seg'
# root2 = '/disk4/CMU-RAW/MOSI/Raw/Video/Frames1'

# root1 = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented-1/Combined'
# root2 = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented/Combined'
# root3 = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented-1/Compress'
# root4 = '/disk4/CMU-RAW/MOSEI/Raw/Videos/Segmented-1/Frames'
path_raw_video = '/disk4/CMU-RAW/MOSI/Raw/Video/Segmented'
path_compress_video = '/disk4/CMU-RAW/MOSI/Raw/Video/Segment-1'

# path_compress_video = '/disk4/CMU-RAW/MOSI/Raw/Video/Compress'
path_frames = '/disk4/CMU-RAW/MOSI/Raw/Video/Frames'


def compress(paras):
    input_video_path, output_video_path = paras
    try:
        command = ['ffmpeg',
                   '-y',  # (optional) overwrite output file if it exists
                   '-i', input_video_path,
                   '-filter:v',
                   'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',  # scale to 224
                   '-map', '0:v',
                   '-r', '30',  # frames per second
                   output_video_path,
                   ]
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        retcode = ffmpeg.poll()
        # print something above for debug
    except Exception as e:
        raise e

def extract_frame(paras):
    input_video_path, output_video_path = paras
    os.makedirs(output_video_path, exist_ok=True)
    try:
        command = ['ffmpeg',
                   '-i', input_video_path,
                   '-r', '5',  # frames per second
                   '-f', 'image2',
                   f'{output_video_path}/image-%2d.png',
                   ]
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        retcode = ffmpeg.poll()
        # print something above for debug
    except Exception as e:
        raise e

def prepare_input_output_pairs_compress(input_root, output_root):
    input_video_path_list = []
    output_video_path_list = []
    for root, dirs, files in os.walk(input_root):
        for file_name in files:
            input_video_path = os.path.join(root, file_name)
            output_video_path = os.path.join(output_root, file_name)
            #input_video_path = input_video_path.replace('Segmented', 'Segmented-1')
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                pass
            else:
                input_video_path_list.append(input_video_path)
                output_video_path_list.append(output_video_path)
            print(input_video_path, output_video_path)
    return input_video_path_list, output_video_path_list

def prepare_input_output_pairs_frames(input_root, output_root):
    input_video_path_list = []
    output_video_path_list = []
    for root, dirs, files in os.walk(input_root):
        for file_name in files:
            input_video_path = os.path.join(root, file_name)
            output_video_path = os.path.join(output_root, file_name)
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                pass
            else:
                input_video_path_list.append(input_video_path)
                output_video_path_list.append(output_video_path[0: -4])
            print(input_video_path, output_video_path[0: -4])
    return input_video_path_list, output_video_path_list

if __name__ == "__main__":
    # compress video
    # parser = argparse.ArgumentParser(description='Compress video for speed-up')
    # parser.add_argument('--input_root', type=str, help='input root', default=path_raw_video)
    # parser.add_argument('--output_root', type=str, help='output root', default=path_compress_video)
    # args = parser.parse_args()
    # input_root = args.input_root
    # output_root = args.output_root
    # assert input_root != output_root
    # if not os.path.exists(output_root):
    #     os.makedirs(output_root, exist_ok=True)
    
    # input_video_path_list, output_video_path_list = prepare_input_output_pairs_compress(input_root, output_root)
    # print("Total video need to process: {}".format(len(input_video_path_list)))
    # num_works = cpu_count()
    # print("Begin with {}-core logical processor.".format(num_works))
    # pool = Pool(num_works)
    # data_dict_list = pool.map(compress,
    #                           [(input_video_path, output_video_path) for
    #                            input_video_path, output_video_path in
    #                            zip(input_video_path_list, output_video_path_list)])
    # pool.close()
    # pool.join()
    
    # extract frames
    parser = argparse.ArgumentParser(description='Compress video for speed-up')
    parser.add_argument('--input_root', type=str, help='input root', default=path_compress_video)
    parser.add_argument('--output_root', type=str, help='output root', default=path_frames)
    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root
    assert input_root != output_root
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    input_video_path_list, output_video_path_list = prepare_input_output_pairs_frames(input_root, output_root)
    print("Total video need to process: {}".format(len(input_video_path_list)))
    num_works = cpu_count()
    print("Begin with {}-core logical processor.".format(num_works))
    pool = Pool(num_works)
    data_dict_list = pool.map(extract_frame,
                              [(input_video_path, output_video_path) for
                               input_video_path, output_video_path in
                               zip(input_video_path_list, output_video_path_list)])
    pool.close()
    pool.join()

    # files = os.listdir(path_frames)
    # for file in files:
    #     sub_dir = os.path.join(path_frames, file)
    #     sub_files = os.listdir(sub_dir)
    #     if len(sub_files) == 0:
    #         print(sub_dir)
    #     else:
    #         print(len(sub_files))