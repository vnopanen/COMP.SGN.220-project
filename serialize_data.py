import getting_and_init_the_data
import utils
import pickle
import os
import re

import numpy as np
INST_PATTERN = '(\[)(...)(\])'
NUMBER_OF_INSTRUMENTS = 4
INSTRUMENTS = {
    'cel': 0,
    'flu': 1,
    'pia': 2,
    'sax': 3,
}



def extract_audio_features(file):
    audio_data = utils.get_audio_file_data(file)
    mel = utils.extract_mel_band_energies(audio_data[0], audio_data[1])

    return mel


def create_pickle(dirname: str, data, name):
    pickle.dump(data,
                open(dirname + '/' + name + '.pickle',
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
            create_pickle(dest_dir, features, file[:-4])
            # augmentations later

    return


def parse_irmas_text_label(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    label = ["0"] * NUMBER_OF_INSTRUMENTS

    for line in lines:
        if line.strip() in INSTRUMENTS:
            label[INSTRUMENTS[line.strip()]] = "1"

    return "".join(label)


def parse_irmas_testing_set(source, destination):

    files_list = []

    data_path = os.path.abspath(source)

    if not os.path.isdir(os.path.abspath(destination)):
        os.makedirs(os.path.abspath(destination))

    # Recursive crawl
    r = 0
    for root, dir, files in os.walk(data_path, topdown=True):
        r += 1
        print("Processing directory: " + str(r))
        for file in files:

            if file[-4:] != ".wav":
                continue

            features = extract_audio_features(root + "/" + file)
            # reduce audio length here if necessary

            # Parse labels
            base_name = file[:-4]

            label = parse_irmas_text_label(root + "/" + base_name + '.txt')
            print(label)
            # If file has no relevant instruments skip it
            if label == "0000":
                continue

            new_file = '/[' + label + '] ' + base_name + '.wav'
            create_pickle(destination, features, new_file)
    return


def main():
    training_data_dir = "../COMP.SGN.220-project/IRMAS-TrainingData"
    training_dir = "../COMP.SGN.220-project/Processed_TrainingData"
    testing_data_dir = "../COMP.SGN.220-project/IRMAS-TestingData-Part1"
    testing_dir = "../COMP.SGN.220-project/Processed_TestingData"
    parse_irmas_trainingset(training_data_dir, training_dir, 0.9)
    parse_irmas_testing_set(testing_data_dir, testing_dir)

if __name__ == "__main__":
    main()

# EOF
