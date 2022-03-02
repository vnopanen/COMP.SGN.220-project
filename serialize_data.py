import getting_and_init_the_data
import utils
import pickle
import os
import re
import numpy as np





def extract_audio_features(file):

    audio_data, sr = utils.get_audio_file_data(file)
    mel_right = utils.extract_mel_band_energies(audio_data[0], sr)
    mel_left = utils.extract_mel_band_energies(audio_data[1], sr)
    tuple = (mel_left, mel_right)
    return_arr = np.array(tuple)
    print(return_arr.shape)
    return return_arr


def create_pickle(dirname: str, data, name):
    pickle.dump(data,
                open(dirname + '/' + name + '.pickle',
                     'wb'))


def augment_audio(rate, data):
    pass


def parse_irmas_trainingset(source, destination, split_percentage):

    files_list = []
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
        count = 0
        for file in files:

            if file[-4:] != ".wav":
                continue

            match = re.search(utils.INST_PATTERN, file)
            if not match:
                continue

            features = extract_audio_features(root + "/" + file)

            dest_dir = destination + "/Train"
            if np.random.random() > split_percentage:
                dest_dir = destination + "/Validation"

            match = re.search(utils.INST_PATTERN, file)
            label = utils.create_one_hot_encoding(match.group(2),
                                                  list(utils.INSTRUMENTS
                                                       .keys()))
            if match:
                data_tuple = (features, label)
                create_pickle(dest_dir, data_tuple, '/[' + match.group(2)
                              + ']_' + str(count))
            else:
                print("Dataset error")
            count += 1
            # augmentations later

    return


def parse_irmas_text_label(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    label = ["0"] * utils.NUMBER_OF_INSTRUMENTS

    for line in lines:
        if line.strip() in utils.INSTRUMENTS:
            label[utils.INSTRUMENTS[line.strip()]] = "1"

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
            # Reduce audio length here if necessary

            # Parse labels
            base_name = file[:-4]

            label = parse_irmas_text_label(root + "/" + base_name + '.txt')
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

    validation_split = 0.9
    parse_irmas_trainingset(training_data_dir, training_dir, validation_split)
    parse_irmas_testing_set(testing_data_dir, testing_dir)

if __name__ == "__main__":
    main()

# EOF
