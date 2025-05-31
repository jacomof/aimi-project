import os
import torch
import nibabel
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import animation
from monai.data import MetaTensor
from multiprocessing import Process, Pool
from sklearn.preprocessing import MinMaxScaler 
from monai.transforms import (
    Orientation,
    EnsureType,
)
import SimpleITK as sitk
import blosc2
# whoever wrote this code knew what he was doing (hint: It was me!)

"""
data 
 │
 ├───train
 │      ├──imageTr
 │      │      └──BRATS_001_0000.nii.gz
 │      │      └──BRATS_001_0001.nii.gz
 │      │      └──BRATS_001_0002.nii.gz
 │      │      └──BRATS_001_0003.nii.gz
 │      │      └──BRATS_002_0000.nii.gz
 │      │      └──...
 │      ├──labelsTr
 │      │      └──BRATS_001.nii.gz
 │      │      └──BRATS_002.nii.gz
 │      │      └──...
 │      ├──imageTs
 │      │      └──BRATS_485_000.nii.gz
 │      │      └──BRATS_485_001.nii.gz
 │      │      └──BRATS_485_002.nii.gz
 │      │      └──BRATS_485_003.nii.gz
 │      │      └──BRATS_486_000.nii.gz
 │      │      └──...

"""
class ConvertToMultiChannelBasedOnUls2023Classes(object):
    """
    Convert labels to multi channels based on brats17 classes:
    "0": "background", 
    "1": "edema",
    "2": "non-enhancing tumor",
    "3": "enhancing tumour"
    Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2),
    and the necrotic and non-enhancing tumor (NCR/NET — label 1)
    """
    def __call__(self, img):

        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        # We have two classes: 0 (background) and 1 (lesion)
        result = [img == 1]
        # merge background and lesion
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

class ULS2023Preprocess:
    def __init__(
        self,
        root_dir: str,
        train_folder_name: str = "train",
        save_dir: str = "../ULS2023_Training_Data",
        cpu_count: int = 1,
    ):
        """
        root_dir: path to the data folder where the raw train folder is
        roi: spatiotemporal size of the 3D volume to be resized
        train_folder_name: name of the folder of the training data
        save_dir: path to directory where each case is going to be saved as a single file containing four modalities
        """

        self.train_folder_dir = os.path.join(root_dir, train_folder_name)
        print(f"train folder dir: {self.train_folder_dir}")
        label_folder_dir = os.path.join(root_dir, train_folder_name, "labelsTr")
        assert os.path.exists(self.train_folder_dir)
        assert os.path.exists(label_folder_dir)
        
        self.save_dir = save_dir
        # we only care about case names for which we have label! 
        self.case_name = next(os.walk(label_folder_dir), (None, None, []))[2]
        print(f"Number of cases: {len(self.case_name)}")
        self.cpu_count = cpu_count

    def __len__(self):
        return self.case_name.__len__()

    def normalize(self, x:np.ndarray)->np.ndarray:
        # Transform features by scaling each feature to a given range.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # (H, W, D) -> (H * W, D)
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1D_array.reshape(x.shape)
        return normalized_data

    def orient(self, x: MetaTensor) -> MetaTensor:
        # orient the array to be in (Right, Anterior, Superior) scanner coordinate systems
        assert type(x) == MetaTensor
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert type(x) == MetaTensor
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def crop_brats2021_zero_pixels(self, x: np.ndarray)->np.ndarray:
        # get rid of the zero pixels around mri scan and cut it so that the region is useful
        # crop (240, 240, 155) to (128, 128, 128)
        return x[:, 56:184, 56:184, 13:141]

    def remove_case_name_artifact(self, case_name: str)->str:
        # BRATS_066.nii.gz -> BRATS_066
        return case_name.rsplit(".")[0]

    def get_fp(self, case_name: str, folder: str, mri_code: str = None):
        """
        return the modality file path
        case_name: patient ID
        folder: either [imagesTr, labelsTr]
        mri_code: code of any of the ["Flair", "T1w", "T1gd", "T2w"]
        """
        if mri_code:
            f_name = f"{case_name}_{mri_code}.nii.gz"
        else:
            f_name = f"{case_name}.nii.gz"

        modality_fp = os.path.join(
            self.train_folder_dir,
            folder,
            f_name,
        )
        return modality_fp

    def load_nifti(self, fp):
        """
        load a nifti file
        fp: path to the nifti file with (nii or nii.gz) extension
        """
        nifti_data = nibabel.load(fp)
        # get the floating point array
        nifti_scan = nifti_data.get_fdata()
        # get affine matrix
        affine = nifti_data.affine
        return nifti_scan, affine

    def _2metaTensor(self, nifti_data: np.ndarray, affine_mat: np.ndarray):
        """
        convert a nifti data to meta tensor
        nifti_data: floating point array of the raw nifti object
        affine_mat: affine matrix to be appended to the meta tensor for later application such as transformation
        """
        # creating a meta tensor in which affine matrix is stored for later uses(i.e. transformation)
        scan = MetaTensor(x=nifti_data, affine=affine_mat)
        # adding a new axis
        D, H, W = scan.shape
        # adding new axis
        scan = scan.view(1, D, H, W)
        return scan

    def preprocess_uls(self, data_fp: str, is_label: bool = False)->np.ndarray:
        """
        apply preprocess stage to the modality
        data_fp: directory to the modality
        """
        data, affine = self.load_nifti(data_fp)

        # First orient data, because adding new dimensions before makes affine and data incompatible 
        data = MetaTensor(x=data, affine=affine)
        # for oreinting the coordinate system we need the affine matrix
        data = self.orient(data)
        # detaching the meta values from the oriented array
        data = self.detach_meta(data)

        # label do not the be normalized 
        if is_label:
            # Binary mask does not need to be float64! For saving storage purposes!
            data = data.astype(np.uint8)
            # categorical -> one-hot-encoded 
            # (128, 128, 64, 1) -> (2, 128, 128, 64)
            data = ConvertToMultiChannelBasedOnUls2023Classes()(data)
            # print("Label shape after channel mapping: ", data.shape)
        else:
            data = self.normalize(x=data)
            # (128, 128, 64, 1) -> (1, 128, 128, 64)
            data = np.moveaxis(data, -1, 0)

        # images have already been cropped
        #data = self.crop_brats2021_zero_pixels(data)

        return data

    def __getitem__(self, idx):
        # Example: MIX_00001_0000.nii.gz
        case_name = self.case_name[idx]
        # Remove case name file extension: Example: MIX_00001_0000
        case_name = self.remove_case_name_artifact(case_name)
        
        case_path = self.get_fp(case_name, "imagesTr", mri_code='0000')
        
        # preprocess image
        im = self.preprocess_uls(case_path, is_label=False)
        # print("Preprocessed image shape: ", im.shape)
        im_transv = im.swapaxes(1, 3) # transverse plane
        # print("Preprocessed image shape after swapaxes: ", im_transv.shape)

        # preprocess segmentation label
        label = self.get_fp(case_name, "labelsTr")
        label = self.preprocess_uls(label, is_label=True)
        # print("Preprocessed label shape: ", label.shape)
        label = label.swapaxes(1, 3) # transverse plane 
        # print("Preprocessed label shape after swapaxes: ", label.shape)

        # add channel dimension (1, D, H, W)
        # im = im_transv.unsqueeze(0)
    
        return im_transv, label, case_name

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated_data: str,
            output_filename_truncated_seg: str = None,
            chunks=None,
            blocks=None,
            chunks_seg=None,
            blocks_seg=None,
            clevel: int = 8,
            codec=blosc2.Codec.ZSTD
    ):
        blosc2.set_nthreads(1)
        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks

        cparams = {
            'codec': codec,
            # 'filters': [blosc2.Filter.SHUFFLE],
            # 'splitmode': blosc2.SplitMode.ALWAYS_SPLIT,
            'clevel': clevel,
        }
        # print(output_filename_truncated, data.shape, seg.shape, blocks, chunks, blocks_seg, chunks_seg, data.dtype, seg.dtype)
        blosc2.asarray(np.ascontiguousarray(data), urlpath=output_filename_truncated_data + '.b2nd', chunks=chunks,
                       blocks=blocks, cparams=cparams)
        blosc2.asarray(np.ascontiguousarray(seg), urlpath=output_filename_truncated_seg + '_seg.b2nd', chunks=chunks_seg,
                       blocks=blocks_seg, cparams=cparams)



    def __call__(self):
        print("started preprocessing ULS2023...")
        with Pool(processes=self.cpu_count) as multi_p:
            multi_p.map_async(func=self.process, iterable=range(self.__len__()))
            multi_p.close()
            multi_p.join()

        print("finished preprocessing ULS2023...")

    def process(self, idx):
        full_savedir = os.path.abspath(self.save_dir)
        print("Save directory: ", full_savedir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        tensor, label, case_name = self.__getitem__(idx)
        # creating the folder for the current case id
        data_save_path = os.path.join(self.save_dir, case_name)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        case_path = data_save_path + f"/{case_name}"
        self.save_case(
            data=tensor,
            seg=label,
            properties={},
            output_filename_truncated=case_path,
            chunks=(1, 128, 128, 64),
            chunks_seg=(2, 128, 128, 64),
            clevel=8,
            codec=blosc2.Codec.ZSTD
        )



def animate(input_1, input_2):
    """animate pairs of image sequences of the same length on two conjugate axis"""
    assert len(input_1) == len(
        input_2
    ), f"two inputs should have the same number of frame but first input had {len(input_1)} and the second one {len(input_2)}"
    # set the figure and axis
    fig, axis = plt.subplots(1, 2, figsize=(8, 8))
    axis[0].set_axis_off()
    axis[1].set_axis_off()
    sequence_length = input_1.__len__()
    sequence = []
    for i in range(sequence_length):
        im_1 = axis[0].imshow(input_1[i], cmap="bone", animated=True)
        im_2 = axis[1].imshow(input_2[i], cmap="bone", animated=True)
        if i == 0:
            axis[0].imshow(input_1[i], cmap="bone")  # show an initial one first
            axis[1].imshow(input_2[i], cmap="bone")  # show an initial one first

        sequence.append([im_1, im_2])
    return animation.ArtistAnimation(
        fig,
        sequence,
        interval=25,
        blit=True,
        repeat_delay=100,
    )

# def viz(volume_indx: int = 1, label_indx: int = 1)->None:
#     """
#     pair visualization of the volume and label
#     volume_indx: index for the volume. ["Flair", "t1", "t1ce", "t2"]
#     label_indx: index for the label segmentation ["TC" (Tumor core), "WT" (Whole tumor), "ET" (Enhancing tumor)]
#     """
#     assert volume_indx in [0, 1, 2, 3]
#     assert label_indx in [0, 1, 2]
#     x = volume[volume_indx, ...]
#     y = label[label_indx, ...]
#     ani = animate(input_1=x, input_2=y)
#     plt.show()


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ULS2023 Preprocess")
    parser.add_argument(
        '--cpu_count',
        type=int,
        help='number of cpu cores to use for the preprocessing',
        default=1,
    )

    args = parser.parse_args()
    if args.cpu_count < 1:
        raise ValueError("cpu_count should be greater than 0")
    if args.cpu_count > os.cpu_count():
        raise ValueError(
            f"cpu_count should be less than or equal to {os.cpu_count()}"
        )

    uls2023_prep = ULS2023Preprocess(root_dir="./",
    	train_folder_name = "train",
        save_dir="../ULS2023_Training_Data",
        cpu_count=args.cpu_count
    )
    # run the preprocessing pipeline 
    uls2023_prep()

    # in case you want to visualize the data you can uncomment the following. Change the index to see different data 
    # volume, label, case_name = brats2017_task1_prep[400]
    # viz(volume_indx = 3, label_indx = 1)


