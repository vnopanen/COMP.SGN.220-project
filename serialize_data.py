import getting_and_init_the_data
import utils
import pickle
import os
import re

import numpy as np
INST_PATTERN = '(\[)(...)(\])'


def extract_audio_features(file):
    audio_data = utils.get_audio_file_data(file)
    mel = utils.extract_mel_band_energies(audio_data[0], audio_data[1])

    return mel


def create_pickle(dirname: str, array: []):
    for i in range(len(array)):
        pickle.dump(array[i],
                    open(dirname + '/datapoint_0' + i.__str__() + '.pickle',
                         'wb'))


def parse_irmas_trainingset(source, destination, split_percentage):

    file_list = []
    data_path = os.path.abspath(source)

    if not os.path.isdir(os.path.abspath(destination)):
        os.makedirs(os.path.abspath(destination))
    if not os.path.isdir(os.path.abspath(destination + "/Train")):
        os.makedirs(os.path.abspath(destination + "/Train"))
    if not os.path.isdir(os.path.abspath(destination + "/Validation")):
        os.makedirs(os.path.abspath(destination + "/Validation"))

    r = 0
    for root, dir, files in os.walk(data_path, topdown=True):
        r += 1
        print("Processing directory: " + str(r))
        i = 0
        for file in files:
            if file[-4:] != ".wav":
                continue

            match = re.search(INST_PATTERN, file)
            if not match:
                continue

            features = extract_audio_features(root + "/" + file)

            dest_dir = destination + "/Train"
            if np.random.random() > split_percentage:
                dest_dir = destination + "/Validation"

            create_pickle(dest_dir, features)
            # augmentations later

    return


def main():
    data_dir = "../COMP.SGN.220-project/IRMAS-TrainingData"
    training_dir = "../COMP.SGN.220-project/Processed_TrainingData"
    parse_irmas_trainingset(data_dir, training_dir, 0.9)


if __name__ == "__main__":
    main()

# EOF
