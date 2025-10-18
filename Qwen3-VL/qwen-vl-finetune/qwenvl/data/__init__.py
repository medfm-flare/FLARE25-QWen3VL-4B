import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

# ===== FLARE25 Medical Imaging Datasets =====
# Converted FLARE25 datasets for medical multimodal training
# Using absolute paths in converted JSON files, so data_path is empty
FLARE_NEOJAUNDICE = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/neojaundice_train.json",
    "data_path": "",
}

FLARE_RETINO = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/retino_train.json",
    "data_path": "",
}

FLARE_BUSI_DET = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/BUSI-det_train.json",
    "data_path": "",
}

FLARE_BONERESORPTION = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/boneresorption_train.json",
    "data_path": "",
}

FLARE_BONE_MARROW = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/bone_marrow_train.json",
    "data_path": "",
}

# FLARE25 Validation Datasets
FLARE_NEOJAUNDICE_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/neojaundice_val.json",
    "data_path": "",
}

FLARE_RETINO_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/retino_val.json",
    "data_path": "",
}

FLARE_BUSI_DET_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/BUSI-det_val.json",
    "data_path": "",
}

FLARE_BONERESORPTION_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/boneresorption_val.json",
    "data_path": "",
}

FLARE_BONE_MARROW_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/bone_marrow_val.json",
    "data_path": "",
}

# Additional FLARE25 Training Datasets
FLARE_FUNDUS = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/fundus_train.json",
    "data_path": "",
}

FLARE_BUS_UCLM_DET = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/BUS-UCLM-det_train.json",
    "data_path": "",
}

FLARE_BUSI = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/BUSI_train.json",
    "data_path": "",
}

FLARE_BUS_UCLM = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/BUS-UCLM_train.json",
    "data_path": "",
}

FLARE_IUGC = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/iugc_train.json",
    "data_path": "",
}

FLARE_DENTAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/dental_train.json",
    "data_path": "",
}

FLARE_PERIAPICAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/periapical_train.json",
    "data_path": "",
}

FLARE_IU_XRAY = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/IU_XRay_train.json",
    "data_path": "",
}

FLARE_CHESTDR = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/chestdr_train.json",
    "data_path": "",
}

FLARE_CHROMOSOME = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/chromosome_train.json",
    "data_path": "",
}

FLARE_NEURIPS22CELL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/neurips22cell_train.json",
    "data_path": "",
}

FLARE_ENDO = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/endo_train.json",
    "data_path": "",
}

FLARE_BCN20000 = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/bcn20000_train.json",
    "data_path": "",
}

FLARE_CMMD = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/CMMD_train.json",
    "data_path": "",
}

# Additional FLARE25 Validation Datasets
FLARE_FUNDUS_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/fundus_val.json",
    "data_path": "",
}

FLARE_BUS_UCLM_DET_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/BUS-UCLM-det_val.json",
    "data_path": "",
}

FLARE_BUSI_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/BUSI_val.json",
    "data_path": "",
}

FLARE_BUS_UCLM_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/BUS-UCLM_val.json",
    "data_path": "",
}

FLARE_IUGC_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/iugc_val.json",
    "data_path": "",
}

FLARE_DENTAL_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/dental_val.json",
    "data_path": "",
}

FLARE_PERIAPICAL_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/periapical_val.json",
    "data_path": "",
}

FLARE_IU_XRAY_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/IU_XRay_val.json",
    "data_path": "",
}

FLARE_CHESTDR_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/chestdr_val.json",
    "data_path": "",
}

FLARE_CHROMOSOME_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/chromosome_val.json",
    "data_path": "",
}

FLARE_NEURIPS22CELL_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/neurips22cell_val.json",
    "data_path": "",
}

FLARE_ENDO_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/endo_val.json",
    "data_path": "",
}

FLARE_BCN20000_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/bcn20000_val.json",
    "data_path": "",
}

FLARE_CMMD_VAL = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/CMMD_val.json",
    "data_path": "",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    # FLARE25 Training Datasets (19 total)
    "flare_neojaundice": FLARE_NEOJAUNDICE,
    "flare_retino": FLARE_RETINO,
    "flare_busi_det": FLARE_BUSI_DET,
    "flare_boneresorption": FLARE_BONERESORPTION,
    "flare_bone_marrow": FLARE_BONE_MARROW,
    "flare_fundus": FLARE_FUNDUS,
    "flare_bus_uclm_det": FLARE_BUS_UCLM_DET,
    "flare_busi": FLARE_BUSI,
    "flare_bus_uclm": FLARE_BUS_UCLM,
    "flare_iugc": FLARE_IUGC,
    "flare_dental": FLARE_DENTAL,
    "flare_periapical": FLARE_PERIAPICAL,
    "flare_iu_xray": FLARE_IU_XRAY,
    "flare_chestdr": FLARE_CHESTDR,
    "flare_chromosome": FLARE_CHROMOSOME,
    "flare_neurips22cell": FLARE_NEURIPS22CELL,
    "flare_endo": FLARE_ENDO,
    "flare_bcn20000": FLARE_BCN20000,
    "flare_cmmd": FLARE_CMMD,
    # FLARE25 Validation Datasets (19 total)
    "flare_neojaundice_val": FLARE_NEOJAUNDICE_VAL,
    "flare_retino_val": FLARE_RETINO_VAL,
    "flare_busi_det_val": FLARE_BUSI_DET_VAL,
    "flare_boneresorption_val": FLARE_BONERESORPTION_VAL,
    "flare_bone_marrow_val": FLARE_BONE_MARROW_VAL,
    "flare_fundus_val": FLARE_FUNDUS_VAL,
    "flare_bus_uclm_det_val": FLARE_BUS_UCLM_DET_VAL,
    "flare_busi_val": FLARE_BUSI_VAL,
    "flare_bus_uclm_val": FLARE_BUS_UCLM_VAL,
    "flare_iugc_val": FLARE_IUGC_VAL,
    "flare_dental_val": FLARE_DENTAL_VAL,
    "flare_periapical_val": FLARE_PERIAPICAL_VAL,
    "flare_iu_xray_val": FLARE_IU_XRAY_VAL,
    "flare_chestdr_val": FLARE_CHESTDR_VAL,
    "flare_chromosome_val": FLARE_CHROMOSOME_VAL,
    "flare_neurips22cell_val": FLARE_NEURIPS22CELL_VAL,
    "flare_endo_val": FLARE_ENDO_VAL,
    "flare_bcn20000_val": FLARE_BCN20000_VAL,
    "flare_cmmd_val": FLARE_CMMD_VAL,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
