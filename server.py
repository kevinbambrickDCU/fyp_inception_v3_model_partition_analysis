import socket
import sys
import numpy as np
import torch
import json
import os
import torchvision
import errno
import argparse

from dahuffman import HuffmanCodec
from analysis import server_run, decode_delta, decode, load_huff_dictionary, classify_server_run

FLAGS = None


def main():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('localhost', 10000)
    print(sys.stderr, 'starting up on %s port %s' % server_address)
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)
    connect = True
    failCount = 0
    passCount = 0

    INCEPT = torchvision.models.inception_v3(pretrained=True)
    PREVIOUS_ARRAY = None
    USE_DELTA = True
    LAST_EDGE_LAYER = FLAGS.layer_index
    NUM_BINS = FLAGS.num_bins
    DELTA_VALUE = FLAGS.delta_value
    save_results = FLAGS.save_results
    analyse_fc_results = FLAGS.compare_fc

    print('Number of bins: ', NUM_BINS)
    print('DELTA VALUE: ', DELTA_VALUE)
    print('Last edge index: ', LAST_EDGE_LAYER)
    print('Saving Results: ', save_results)

    if (LAST_EDGE_LAYER == 7):
        RESHAPE_ARRAY_DIMENSIONS = [192, 35, 35]
    elif (LAST_EDGE_LAYER == 11):
        RESHAPE_ARRAY_DIMENSIONS = [768, 17, 17]
    elif (LAST_EDGE_LAYER == 6):
        RESHAPE_ARRAY_DIMENSIONS = [192, 71, 71]
    else:
        RESHAPE_ARRAY_DIMENSIONS = None
        print("Reshape dimensions not defined for layer being partitioned")

    codec_path = 'huffman_encoding_config/' + 'layer' + str(LAST_EDGE_LAYER) + '/' + 'num_bins_' + str(NUM_BINS)
    delta_hist = load_huff_dictionary(codec_path + '/delta_hist')
    delta_codec = HuffmanCodec.from_frequencies(delta_hist)
    frame_one_hist = load_huff_dictionary(codec_path + '/frame_one_hist')
    frame_one_codec = HuffmanCodec.from_frequencies(frame_one_hist)

    sizes = []
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

    # for analysing fc output
    if analyse_fc_results is True:
        test_videos = [
            "videos/n01882714/koala_1.mp4",
            "videos/n02510455/panda_1.mp4",
            "videos/n02676566/guitar_2.mp4",
            "videos/n02133161/bear_1.mp4",
            "videos/n02110958/pug_3.mp4"
        ]
        videos = test_videos

    vid_num = 0
    frame_number = 0

    cats = json.load(open('config/categories.json'))
    class_id = videos[vid_num].split('/')[1]
    for j in range(len(cats)):
        if cats[j]['id'] == class_id:
            index = cats[j]['index']

    while True:
        # Wait for a connection
        received = ''
        arr = bytearray()
        byte_size = 0

        print(sys.stderr, 'waiting for a connection')
        connection, client_address = sock.accept()

        try:
            print(sys.stderr, 'connection from', client_address)

            # Receive the data in small chunks and retransmit it
            while True:
                data = connection.recv(1024)
                byte_size = byte_size + len(data)

                arr.extend(data)
                # print(sys.stderr, 'received "%s"' % data)
                if data:
                    # print(sys.stderr, 'sending data back to the client')
                    connection.sendall(data)
                else:
                    print(sys.stderr, 'no more data from', client_address)
                    connect = False
                    break

        finally:
            # Clean up the connection
            connection.close()
            print('Size of data received: ', byte_size)

            # code for receiving reset to frame one
            if byte_size == 1:
                print('Received reset')
                avg_byte_size = sum(sizes) / len(sizes)
                passRate = (passCount / (failCount + passCount)) * 100
                print('percentage of passed: ', passRate)
                result = 'file: ' + videos[vid_num] + ', %Passed: ' + str(passRate) + ', avg_byte_size: ' \
                         + str(avg_byte_size) + ', layer: ' + str(LAST_EDGE_LAYER) + ', num_bins_used: ' + str(
                    NUM_BINS) + \
                         ', Delta Value: ' + str(DELTA_VALUE) + '\n'
                results_path = "Results" + '/layer' + str(LAST_EDGE_LAYER) + '/num_bins_' + str(
                    NUM_BINS) + '/delta_value' \
                               + str(DELTA_VALUE)

                if save_results is True:
                    if not os.path.isdir(results_path):
                        try:
                            os.makedirs(results_path)
                        except OSError as e:
                            if e.errno != errno.EEXIST:
                                raise
                    with open(results_path + "/results.txt", "a") as myfile:
                        myfile.write(result)

                # Resetting variables
                PREVIOUS_ARRAY = None
                sizes = []
                passCount = 0
                failCount = 0
                vid_num += 1
                frame_number = 0

                cats = json.load(open('config/categories.json'))
                class_id = videos[vid_num].split('/')[1]
                for j in range(len(cats)):
                    if cats[j]['id'] == class_id:
                        index = cats[j]['index']

            elif PREVIOUS_ARRAY is not None and USE_DELTA is True:
                decoded = delta_codec.decode(arr)
                arr = np.reshape(decoded, RESHAPE_ARRAY_DIMENSIONS)

                decoded_arr = decode(arr, NUM_BINS, max_num=8, min_num=-8)
                delta_decoded_arr = decode_delta(PREVIOUS_ARRAY, decoded_arr)
                PREVIOUS_ARRAY = delta_decoded_arr
                fc_out = server_run(torch.Tensor(delta_decoded_arr), LAST_EDGE_LAYER, INCEPT)
                result = classify_server_run(fc_out, class_label=index)
            else:
                decoded = frame_one_codec.decode(arr)
                arr = np.reshape(decoded, RESHAPE_ARRAY_DIMENSIONS)

                decoded_arr = decode(arr, NUM_BINS, max_num=8, min_num=-8)
                PREVIOUS_ARRAY = decoded_arr
                fc_out = server_run(torch.Tensor(decoded_arr), LAST_EDGE_LAYER, INCEPT)
                result = classify_server_run(fc_out, class_label=index)

            # 0 for false 1 for true, str so can be written to file and easily calculate total
            top_five_is_the_same = '0'
            top_one_is_the_same = '0'
            if analyse_fc_results is True:
                video_file = videos[vid_num].split('/')[2]
                video = video_file.split('.')[0]
                saved_fc_dir = 'Results/fc_results/' + class_id
                path_to_saved_fc = saved_fc_dir + '/' + video + '_' + str(frame_number) + '.npy'
                print('looking for: ', path_to_saved_fc)
                if os.path.isdir(saved_fc_dir):
                    print('path exists')
                    unencoded_fc = np.load(path_to_saved_fc)
                    unencoded_top_five = unencoded_fc.argsort()[0][-1:-6:-1]  # Get top 5 classifications.
                    encoded_top_five = fc_out.data.numpy().argsort()[0][-1:-6:-1]
                    unencoded_top = unencoded_fc.argsort()[0][-1]
                    encoded_top = fc_out.data.numpy().argsort()[0][-1]
                    if (np.array_equal(unencoded_top_five, encoded_top_five)):
                        top_five_is_the_same = '1'
                    if (np.array_equal(unencoded_top, encoded_top)):
                        top_one_is_the_same = '1'

                    path_to_results = 'Results/fc_results/comparison_results/layer_' + str(LAST_EDGE_LAYER) + \
                                      '/num_bins_' + str(NUM_BINS) + \
                                      '/delta_value_' + str(DELTA_VALUE)
                    fc_analysis_result = videos[vid_num] + ' ,same top five predictions ,' + top_five_is_the_same + \
                                         ' ,same top one prediction ,' + top_one_is_the_same  + '\n'
                    if not (os.path.isdir(path_to_results)):
                        os.makedirs(path_to_results)
                    with open(path_to_results + '/fc_results.txt', 'a') as myfile:
                        myfile.write(fc_analysis_result)

            frame_number += 1
            if result:
                passCount += 1
            else:
                failCount += 1
            sizes.append(byte_size)
            print('Total checked: ', passCount + failCount)
            print('Number of correct classifications: ', passCount)
            print('Number Failed: ', failCount)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        '--save_results',
        type=str2bool,
        default=True,
        help='Save the output of the test results to a file, True by default'
    )
    parser.add_argument(
        '--compare_fc',
        type=str2bool,
        default=False,
        help='Analyse the output of the fc layer and compare it to output without encoding'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
