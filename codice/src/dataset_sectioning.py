
from _cfg_params import DATASET_PATH, NUM_CLASS, TRAINING_DATA_PERCENTAGE, VALIDATION_SET_PERCENTAGE, DIM_PATTERN

# read structured data from received file path
def read_data_from(path: str, datanum: bool=True) -> list:
    data = []
    with open(path, mode="r") as fdata:
        lines = fdata.readlines()
        for line in lines:
            line = line.split(';')
            data.append([(int(str) if datanum else str) for str in line])
        fdata.close()
    return data

# write structured data to received file path
def write_data_to(path: str, data: list):
    with open(path, mode="w") as fdata:
        fdata.writelines([';'.join([str(feat) for feat in features]) + '\n' for features in data])
        fdata.close()

# create uniformly training, validation and test sets with set training data percentage
def training_validation_test_sets() -> tuple[list, list, list]:
    training_set, validation_set, test_set = [], [], []
    dataset = read_data_from(DATASET_PATH)
    dissected_dataset = []
    for label in range(NUM_CLASS):
        dissected_dataset.append([features for features in dataset if features[DIM_PATTERN] == label])
    min_len = min([len(section) for section in dissected_dataset]) # some labelled patterns ​​will not be inserted

    num_train_valid_per_section = int(TRAINING_DATA_PERCENTAGE*min_len)
    num_valid_per_section = int(VALIDATION_SET_PERCENTAGE*num_train_valid_per_section)
    num_train_per_section = num_train_valid_per_section - num_valid_per_section
    num_test_per_section = int((1-TRAINING_DATA_PERCENTAGE)*min_len)
    for label in range(NUM_CLASS):
        training_set += dissected_dataset[label][:num_train_per_section]
        validation_set += dissected_dataset[label][num_train_per_section:(num_train_per_section + num_valid_per_section)]
        test_set += dissected_dataset[label][-num_test_per_section:]
    return training_set, validation_set, test_set

def main():
    print('Sectioning...')
    training, validation, test = training_validation_test_sets()
    write_data_to('../data/training_set.dat', training)
    write_data_to('../data/validation_set.dat', validation)
    write_data_to('../data/test_set.dat', test)
    print('finish')


# entry point
if __name__ == '__main__':
    main()