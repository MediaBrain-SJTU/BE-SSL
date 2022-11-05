from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil

if __name__ == "__main__":
    base = "./data/candi/raw_data/preprocessed_label"

    task_id = 160
    task_name = "CandiBrainSegmentation"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_folder = join(base, "imagesTr")
    label_folder = join(base, "labelsTr")
    test_folder = join(base, "imagesTs")
    test_label_folder = join(base, "labelsTs")

    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(train_folder, join=False, suffix = 'nii.gz')

    for p in train_patients:
        serial_number = int(p[:3])
        train_patient_name = f'{serial_number:03d}.nii.gz'
        label_file = join(label_folder, p)
        image_file = join(train_folder, p)
        shutil.copy(image_file, join(imagestr, f'{train_patient_name[:3]}_0000.nii.gz'))
        shutil.copy(label_file, join(labelstr, train_patient_name))
        train_patient_names.append(train_patient_name)

    test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    for p in test_patients:
        serial_number = int(p[:3])
        test_patient_name = f'{serial_number:03d}.nii.gz'
        label_file = join(test_label_folder, p)
        image_file = join(test_folder, p)
        shutil.copy(image_file, join(imagests, f'{test_patient_name[:3]}_0000.nii.gz'))
        shutil.copy(label_file, join(labelsts, test_patient_name))
        test_patient_names.append(test_patient_name)


    json_dict = OrderedDict()
    json_dict['name'] = "CandiBrainSegmentation"
    json_dict['description'] = "The Child and Adolescent NeuroDevelopment Initiative (CANDI) at UMass Medical School is making available a series of structural brain images, as well as their anatomic segmentations, demographic and behavioral data and a set of related morphometric resources (static and dynamic atlases)."
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "https://www.nitrc.org/projects/candi_share/"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = OrderedDict({
        '00': 'background',
        '01': 'class_01',
        '02': 'class_02',
        '03': 'class_03',
        '04': 'class_04',
        '05': 'class_06',
        '06': 'class_06',
        '07': 'class_07',
        '08': 'class_08',
        '09': 'class_09',
        '10': 'class_10',
        '11': 'class_11',
        '12': 'class_12',
        '13': 'class_13',
        '14': 'class_14',
        '15': 'class_15',
        '16': 'class_16',
        '17': 'class_17',
        '18': 'class_18',
        '19': 'class_19',
        '20': 'class_20',
        '21': 'class_21',
        '22': 'class_22',
        '23': 'class_23',
        '24': 'class_24',
        '25': 'class_25',
        '26': 'class_26',
        '27': 'class_27',
       }
    )
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % train_patient_name} for i, train_patient_name in enumerate(train_patient_names)]
    json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))



