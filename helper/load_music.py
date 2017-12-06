from pydub import AudioSegment
import numpy as np
import os

def music_load(dir='../Music_sample',load_list=[],length=5000
    ,load_limit=50, load_limit_per_song=10, cut=1, head = 0):
    if load_list == []:
        load_list = os.listdir(dir)
    all_file = os.listdir(dir)
    out_array = []
    load_list = list(filter(lambda x: x[-4:] == '.mp3',load_list))
    num_load = 0
    for el in load_list:
        if num_load >= load_limit:
            break
        if el not in all_file:
            print('Warning, file {} not found'.format(el))
            continue
        file_dir = os.path.join(dir,el)
        num_load_in_song = 0
        sound = AudioSegment.from_mp3(file_dir)
        song_length = len(sound)
        while num_load < load_limit and num_load_in_song < load_limit_per_song:
            start = head + num_load_in_song * length
            end = start + length
            if end >= song_length:
                break
            temp = sound[start:end].get_array_of_samples().tolist()
            out_array.append(temp)
            num_load += 1
            num_load_in_song += 1
    return np.array(out_array, dtype='int16')

#x = music_load(load_limit=20,length=3000,head=1000)
