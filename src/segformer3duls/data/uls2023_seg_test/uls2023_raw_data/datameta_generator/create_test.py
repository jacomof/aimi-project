import os
import random
import numpy as np
import pandas as pd


def create_train_val_test_csv_from_data_folder(
    folder_dir: str,
    append_dir: str = "",
    save_dir: str = "./",
    train_split_perc: float = 0.80,
    val_split_perc: float = 0.05,
) -> None:
    """
    create train/validation/test csv file out of the given directory such that each csv file has its split percentage count
    folder_dir: path to the whole corpus of the data
    append_dir: path to be appended to the begining of the directory filed in the csv file
    save_dir: directory to which save the csv files
    train_split_perc: the percentage of the train set by which split the data
    val_split_perc: the percentage of the validation set by which split the data
    """
    assert os.path.exists(folder_dir), f"{folder_dir} does not exist"
    assert (
        train_split_perc < 1.0 and train_split_perc > 0.0
    ), "train split should be between 0 and 1"
    assert (
        val_split_perc < 1.0 and val_split_perc > 0.0
    ), "train split should be between 0 and 1"

    # set the seed
    np.random.seed(0)
    random.seed(0)

    # iterate through the folder to list all the filenames
    case_name = next(os.walk(folder_dir), (None, None, []))[1]
    cropus_sample_count = case_name.__len__()

    # appending append_dir to the case name
    data_dir = []
    for case in case_name:
        data_dir.append(os.path.join(append_dir, case))

    idx = np.arange(0, cropus_sample_count)
    # shuffling idx (inplace operation)
    np.random.shuffle(idx)

    # get the corresponding id from the train,validation and test set
    test_sample_base_dir = np.array(data_dir)[idx]
    test_sample_case_name = np.array(case_name)[idx]

    # create a pandas data frame
    test_df = pd.DataFrame(
        data={"base_dir": test_sample_base_dir, "case_name": test_sample_case_name},
        index=None,
        columns=None,
    )

    # write csv files to the drive!
    test_df.to_csv(
        save_dir + "/test.csv",
        header=["data_path", "case_name"],
        index=False,
    )



if __name__ == "__main__":
    create_train_val_test_csv_from_data_folder(
        # path to the train data folder
        folder_dir="../../ULS2023_Testing_Data",
        # this is inferred from where the actual experiments are run relative to the data folder
        append_dir="../../../data/uls2023_seg_test/ULS2023_Testing_Data/",
        # where to save the train, val and test csv file relative to the current directory
        save_dir=".",
        train_split_perc=0.85,
        val_split_perc=0.15,
    )
