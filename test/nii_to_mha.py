import SimpleITK as sitk
from pathlib import Path

def convert_nii_to_mha(input_dir, output_dir):
    """
    Converts all .nii.gz files in the input directory to .mha files in the output directory.

    :param input_dir: Path to the directory containing .nii.gz files.
    :param output_dir: Path to the directory where .mha files will be saved.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for nii_file in input_dir.glob("*.nii.gz"):
        try:
            # Read the .nii.gz file
            image = sitk.ReadImage(str(nii_file))
            
            # Define the output file path
            output_file = output_dir / (nii_file.stem + ".mha")
            
            # Write the image as .mha
            sitk.WriteImage(image, str(output_file))
            print(f"Converted: {nii_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Failed to convert {nii_file.name}: {e}")

if __name__ == "__main__":
    # Define input and output directories
    input_directory = "./architecture/input/images/DeepLesion3D/imagesTr/"  # Replace with the path to your .nii.gz files
    output_directory = "./architecture/input/images/stacked-3d-ct-lesion-volumes/"  # Replace with the path to save .mha files

    convert_nii_to_mha(input_directory, output_directory)