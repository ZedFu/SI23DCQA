import numpy as np
import os
import argparse
from data_utils import read_annotations, feature_dir
import cv2

def background_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray != 255)
    if len(rows) == 0:
        return img
    return img[rows.min():rows.max()+1, cols.min():cols.max()+1]


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)    
        
    return

def extract_frame(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name )
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    if not video_capture.isOpened():
        raise ValueError(f"Cannot open video: {filename}")
    cap=video_capture

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    if video_length < 1 or video_frame_rate < 1:
        video_capture.release()
        raise ValueError(f"Invalid video length or frame rate: {filename}")

    video_read_index = 0

    frame_idx = 0
    
    video_length_min = 8

    save_dir = 'frames'
    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:
            # key frame
            if (video_read_index < video_length) and (frame_idx % video_frame_rate == 0):
                # read_frame = cv2.resize(frame, dim)
                read_frame = background_crop(frame)
                exit_folder(os.path.join(save_folder, save_dir))
                cv2.imwrite(os.path.join(save_folder, save_dir, \
                                         '{:03d}'.format(video_read_index) + '.png'), read_frame)          
                video_read_index += 1
            frame_idx += 1
            
    video_capture.release()
    if video_read_index == 0:
        raise ValueError(f"No decoded frames: {filename}")
    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            cv2.imwrite(os.path.join(save_folder, save_dir, \
                                     '{:03d}'.format(i) + '.png'), read_frame)


    return
            

def main():
    parser = argparse.ArgumentParser(description='Extract cropped video frames at one frame per second (at least 8).')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--ann', default='all-result.txt', help='Annotation filename under the data root, or an absolute path.')
    args = parser.parse_args()
    for record in read_annotations(os.path.join(args.data_root, args.ann)):
        target = feature_dir(args.data_root, record['video_path'])
        extract_frame(args.data_root, record['video_path'], str(target))
        print(target)


if __name__ == '__main__':
    main()
