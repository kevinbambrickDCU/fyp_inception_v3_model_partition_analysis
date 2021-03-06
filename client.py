#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import torchvision
import argparse
import sys
import analysis
import inception
import json
import os
import random
import requests
import errno
import numpy as np

from torch.autograd import Variable
from dahuffman import HuffmanCodec

FLAGS = None
USER_DELTA = True
PATH_TO_IMAGNET = '../imagnet/val/'
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
RESET = b'\x01'
MSE = []
incept = torchvision.models.inception_v3(pretrained=True)


def main():
    delete_previous_frames(FLAGS.frame_file)
    # THIS ARRAY NEEDS TO BE THE SAME ON THE SERVER SIDE
    videos = [
        "videos/n01443537/goldfish_1.mp4",
        "videos/n01443537/goldfish_2.mp4",
        "videos/n01443537/goldfish_3.mp4",

        "videos/n01882714/koala_1.mp4",
        "videos/n01882714/koala_2.mp4",
        "videos/n01882714/koala_3.mp4",

        "videos/n02085620/dog_1.mp4",

        "videos/n02099601/golden_retriever_1.mp4",

        "videos/n02099712/golden_retriever_1.mp4",

        "videos/n02110958/pug_1.mp4",
        "videos/n02110958/pug_3.mp4",
        "videos/n02110958/pug_4.mp4",

        "videos/n02206856/bee_1.mp4",

        "videos/n02391049/zebra_1.mp4",
        "videos/n02391049/zebra_2.mp4",
        "videos/n02391049/zebra_3.mp4",

        "videos/n02510455/panda_1.mp4",
        "videos/n02510455/panda_2.mp4",
        "videos/n02510455/panda_3.mp4",
        "videos/n02510455/panda_4.mp4",
        "videos/n02510455/panda_5.mp4",

        "videos/n02676566/guitar_1.mp4",
        "videos/n02676566/guitar_2.mp4",
        "videos/n02676566/guitar_3.mp4",
        "videos/n02676566/guitar_4.mp4",
        "videos/n02676566/guitar_6.mp4",

        "videos/n02787622/banjo_1.mp4",
        "videos/n02787622/banjo_2.mp4",
        "videos/n02787622/banjo_3.mp4",
        "videos/n02787622/banjo_5.mp4",

        "videos/n03452741/piano_1.mp4",
        "videos/n03452741/piano_2.mp4",

        "videos/n03495258/harp_1.mp4",
        "videos/n03495258/harp_2.mp4",
        "videos/n03495258/harp_3.mp4",

        "videos/n03584254/ipod_1.mp4",
        "videos/n03584254/ipod_2.mp4",

        "videos/n03967562/plough_1.mp4",

        "videos/n04536866/violin_3.mp4",
        "videos/n04536866/violin_4.mp4",

        "videos/n06596364/comic_1.mp4",

        "videos/n01910747/jelly_fish_1.mp4",
        "videos/n01910747/jelly_fish_2.mp4",

        "videos/n02134084/polar_bear_1.mp4",
        "videos/n02134084/polar_bear_3.mp4",

        "videos/n02342885/hamster_1.mp4",
        "videos/n02342885/hamster_2.mp4",
        "videos/n02342885/hamster_4.mp4",
        "videos/n02342885/hamster_5.mp4",

        "videos/n02364673/guinea_pig_1.mp4",
        "videos/n02364673/guinea_pig_2.mp4"
    ]

    test_videos = [
        "videos/n01882714/koala_1.mp4",
        "videos/n02510455/panda_1.mp4",
        "videos/n02676566/guitar_2.mp4",
        "videos/n02133161/bear_1.mp4",
        "videos/n02110958/pug_3.mp4"
    ]

    # classify_list_of_videos_without_partition(test_videos, save_results = False,  write = True)
    classify_list_of_videos(videos)


def delete_previous_frames(path_to_frames):
    for the_file in os.listdir(path_to_frames):
        file_path = os.path.join(path_to_frames, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def classify_list_of_videos_without_partition(videos, save_results=True, write=False):
    index = 0
    for i in range(len(videos)):
        class_id = videos[i].split('/')[1]
        print(class_id)
        cats = json.load(open(FLAGS.cat_json))
        for j in range(len(cats)):
            if cats[j]['id'] == class_id:
                index = cats[j]['index']
        print(index)
        print('classifying: ', videos[i])
        classify_video_without_splitting(videos[i], index, save_results=save_results, write=write)


def classify_list_of_videos(videos):
    LAST_EDGE_LAYER = FLAGS.layer_index
    NUM_BINS = FLAGS.num_bins
    DELTA_VALUE = FLAGS.delta_value
    codec_path = 'huffman_encoding_config/' + 'layer' + str(LAST_EDGE_LAYER) + '/' + 'num_bins_' + str(NUM_BINS)
    delta_hist = analysis.load_huff_dictionary(codec_path + '/delta_hist')
    delta_codec = HuffmanCodec.from_frequencies(delta_hist)
    frame_one_hist = analysis.load_huff_dictionary(codec_path + '/frame_one_hist')
    frame_one_codec = HuffmanCodec.from_frequencies(frame_one_hist)
    codecs = delta_codec, frame_one_codec

    for i in range(len(videos)):
        print('Classifying: ', videos[i])
        MSE = classify_video(videos[i], codecs, write=False)
        send(RESET)
        results_path = "Results" + '/layer' + str(LAST_EDGE_LAYER) + '/num_bins_' + str(NUM_BINS) + '/delta_value' \
                       + str(DELTA_VALUE)
        avg_error = sum(MSE) / len(MSE)
        result = 'file: ' + videos[i] + ', AVG_MSE: ' + str(avg_error) + '\n'
        if not (os.path.isdir(results_path)):
            try:
                os.makedirs(results_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        with open(results_path + '/MSE.txt', 'a') as myfile:
            myfile.write(result)


def classify_video(path_to_file, codecs, write=False):
    print('Classifying Video frame by fame')
    LAST_EDGE_LAYER = FLAGS.layer_index
    NUM_BINS = FLAGS.num_bins
    DELTA_VALUE = FLAGS.delta_value
    MSE = []
    frame_one_codec = codecs[1]
    delta_codec = codecs[0]

    fps, number_of_frames = analysis.get_fps_and_number_of_frames(path_to_file)
    PREVIOUS_ARRAY = None
    incept.eval()  # put model into evaluation mode

    for i in range(int(number_of_frames / fps)):
        try:
            img = analysis.read_in_frame_from_video(path_to_file, i * fps, write=write)
        except Exception as e:
            print(e)
            break

        edge_out = analysis.SplitComputation.forward(self=incept,  # pre-trained model
                                                     x=Variable(img),
                                                     start=0,
                                                     end=LAST_EDGE_LAYER)

        if PREVIOUS_ARRAY is not None and USER_DELTA is True:
            input_to_compute_deltas = edge_out.data.numpy().squeeze(0)
            delta_edge_output = analysis.compute_delta(PREVIOUS_ARRAY, input_to_compute_deltas, DELTA_VALUE)

            delta_encoded_edge_output = analysis.encode(delta_edge_output, NUM_BINS, min_num=-8,
                                                        max_num=8)

            huff_delta_encoded_edge_output = delta_codec.encode(delta_encoded_edge_output.flatten().astype('int8'))
            send(huff_delta_encoded_edge_output)
            server_decoded = analysis.decode(delta_encoded_edge_output, NUM_BINS).squeeze(0)
            PREVIOUS_ARRAY = PREVIOUS_ARRAY - server_decoded
            # GETTING ERROR BETWEEN EDGE OUT AND ENCODED
            error = ((input_to_compute_deltas - PREVIOUS_ARRAY) ** 2).mean()
            MSE.append(error)
        else:
            input_to_encoder = edge_out.data.numpy().squeeze(0)
            encoded_edge_output = analysis.encode(input_to_encoder, NUM_BINS, min_num=-8,
                                                  max_num=8)
            huff_encoded_edge_output = frame_one_codec.encode(encoded_edge_output.flatten().astype('int8'))
            send(huff_encoded_edge_output)
            PREVIOUS_ARRAY = analysis.decode(encoded_edge_output, NUM_BINS).squeeze(0)
            # GETTING ERROR BETWEEN EDGE OUT AND ENCODED
            error = ((input_to_encoder - PREVIOUS_ARRAY) ** 2).mean()
            MSE.append(error)
        print('ERROR: ', error)
    return MSE


# USED FOR TESTING
def classify_on_random_images(path_to_data_set, number_of_images_to_check):
    print('classifying images at: ', path_to_data_set)

    cats = json.load(open(FLAGS.cat_json))
    imagnet_folder = os.listdir(PATH_TO_IMAGNET)
    num_of_folders = len(imagnet_folder)

    print(num_of_folders)

    for i in range(number_of_images_to_check):
        folder = random.choice(imagnet_folder)
        img_folder = PATH_TO_IMAGNET + folder
        images_list = os.listdir(img_folder)
        image = random.choice(images_list)
        full_path_to_image = img_folder + '/' + image
        print('Classifying from: ', full_path_to_image)
        classify_one_image(full_path_to_image)


# USED FOR TESTING
def classify_one_image(path_to_image):
    print('Classifying one image')
    LAST_EDGE_LAYER = FLAGS.layer_index
    NUM_BINS = FLAGS.num_bins
    img = analysis.load_in_image(path_to_image)

    incept = torchvision.models.inception_v3(pretrained=True)
    incept.eval()

    edge_out = analysis.SplitComputation.forward(self=incept, x=Variable(img),
                                                 start=0, end=LAST_EDGE_LAYER)

    input_to_encoder = edge_out.data.numpy().squeeze(0)
    encoded_edge_output = analysis.encode(input_to_encoder, min_num=-8,
                                          max_num=8, num_bins=NUM_BINS)

    print('shape of data: ', encoded_edge_output.shape)

    send(encoded_edge_output)


def classify_video_without_splitting(path_to_file, class_index_number, save_results=True, write=False):
    fps, number_of_frames = analysis.get_fps_and_number_of_frames(path_to_file)
    print(fps, number_of_frames)

    incept.eval()
    passCount = 0
    failCount = 0
    failed_frames = {}
    for i in range(number_of_frames):

        try:
            img = analysis.read_in_frame_from_video(path_to_file, i * fps, write=True)
        except Exception as e:
            break

        fc_out = inception.Inception3.forward(self=incept, x=Variable(img))
        sort = fc_out.data.numpy().argsort()

        # write output of fully connected to file for analysis
        if write is True:
            class_id = path_to_file.split('/')[1]
            video_file_name = path_to_file.split('/')[2]
            video_name = video_file_name.split('.')[0]
            fc_path = 'Results/fc_results/' + class_id + '/'
            if not os.path.isdir(fc_path):
                os.makedirs(fc_path)
            np.save(fc_path + video_name + '_' + str(i), fc_out.data.numpy())

        try:
            # read labels from file
            labels = {int(key): value for (key, value)
                      in json.load(open('config/labels.json')).items()}
        except Exception as e:
            # download labels
            labels = {int(key): value for (key, value)
                      in requests.get(LABELS_URL).json().items()}

        match = False
        print(labels[fc_out.data.numpy().argmax()])

        for i in range(1, 6):
            print('Number ', i, ': ', labels[sort[0][-i]])
            if (sort[0][-i] == class_index_number):
                match = True

        if (match):
            print('ground truth in top5')
            passCount += 1
        else:
            print('Classification incorrect')
            failCount += 1
            failed_frames[failCount - 1] = failCount + passCount
        print('Checking: ', path_to_file)
        print('Total checked: ', passCount + failCount)
        print('Number of correct classifications: ', passCount)
        print('Number Failed: ', failCount)

    # PRINT RESULTS
    if save_results:
        print('failed frames: ', failed_frames)
        passRate = (passCount / (failCount + passCount)) * 100
        print('percentage of passed: ', passRate)
        result = 'file: ' + path_to_file + ', %Passed: ' + str(passRate) + '\n'
        with open("Results/non_split_results.txt", "a") as myfile:
            myfile.write(result)


def send(data):
    print('Type being sent: ', type(data))
    print('Data being sent to server')
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('localhost', 10000)
    print(sys.stderr, 'connecting to %s port %s' % server_address)
    sock.connect(server_address)

    try:
        message = (data)
        sock.sendall(message)

        # Look for the response
        amount_received = 0
        amount_expected = len(message)

        while amount_received < amount_expected:
            data = sock.recv(1024)
            amount_received += len(data)
    finally:

        print(sys.stderr, 'closing socket')
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--frame_file',
        type=str,
        default='frames/',
        help='Relative path to the folder storing the frames'
    )
    parser.add_argument(
        '--cat_json',
        type=str,
        default='config/categories.json',
        help='Path to JSON categories file '
    )
    parser.add_argument(
        '--num_bins',
        type=int,
        default=60,
        help='Number of bins to use in encoding'
    )
    parser.add_argument(
        '--layer_index',
        type=int,
        default=7,
        help='Layer index the CNN is to be partitioned at'
    )
    parser.add_argument(
        '--delta_value',
        type=float,
        default=0.1,
        help='Delta value to be used in encoding data'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
