""" data.py:
Data presets for training and evaluation.

Single Presets:
    musicnet_mt3
    musicnet_em
    musicnet_thickstun
    slakh
    guitarset
    ...

Multi Presets:
    all_mmegs
    ...

"""
from config.vocabulary import *
from config.vocabulary import drum_vocab_presets, program_vocab_presets
from utils.utils import deduplicate_splits, merge_splits, merge_vocab

data_preset_single_cfg = {
    "musicnet_mt3": {
            "eval_vocab": [MUSICNET_INSTR_CLASS],
            "dataset_name": "musicnet",
            "train_split": "train_mt3",
            "validation_split": "validation_mt3_acoustic",
            "test_split": "test_mt3_acoustic",
            "has_stem": False,
    },
    "musicnet_mt3_synth_only": { # sanity-check
            "eval_vocab": [MUSICNET_INSTR_CLASS],
            "dataset_name": "musicnet",
            "train_split": "train_mt3_synth",
            "validation_split": "validation_mt3_synth",
            "test_split": "test_mt3_acoustic",
            "has_stem": False,
    },
    "musicnet_mt3_em": {
            "eval_vocab": [MUSICNET_INSTR_CLASS],
            "dataset_name": "musicnet",
            "train_split": "train_mt3_em",
            "validation_split": "validation_mt3_em",
            "test_split": "test_mt3_em",
            "has_stem": False,
    },
    "musicnet_thickstun": { # exp4
            "eval_vocab": [MUSICNET_INSTR_CLASS],
            "dataset_name": "musicnet",
            "train_split": "train_thickstun",
            "validation_split": "test_thickstun",
            "test_split": "test_thickstun",
            "has_stem": False,
    },
    "musicnet_thickstun_em": { # NOTE: this is not the use of external 'synth' in the paper, but the use of 'synth' within the dataset
            "eval_vocab": [MUSICNET_INSTR_CLASS],
            "dataset_name": "musicnet",
            "train_split": "train_thickstun_em",
            "validation_split": "test_thickstun_em",
            "test_split": "test_thickstun_em",
            "has_stem": False,
    },
    "musicnet_thickstun_ext": { # exp4
            "eval_vocab": [MUSICNET_INSTR_CLASS],
            "dataset_name": "musicnet",
            "train_split": "train_thickstun",
            "validation_split": "test_thickstun_ext",
            "test_split": "test_thickstun_ext",
            "has_stem": False,
    },
    "musicnet_thickstun_ext_em": { # NOTE: this is not the use of external 'synth' in the paper, but the use of 'synth' within the dataset
            "eval_vocab": [MUSICNET_INSTR_CLASS],
            "dataset_name": "musicnet",
            "train_split": "train_thickstun_em",
            "validation_split": "test_thickstun_ext_em",
            "test_split": "test_thickstun_ext_em",
            "has_stem": False,
    },
    "maps_default": {
            "eval_vocab": [PIANO_SOLO_CLASS],
            "dataset_name": "maps",
            "train_split": "train",
            "validation_split": "test",
            "test_split": "test",
            "has_stem": False,
    },
    "maps_all": {
            "eval_vocab": [None],
            "dataset_name": "maps",
            "train_split": "all",
            "validation_split": None,
            "test_split": None,
            "has_stem": False,
    },
    "maestro": {
            "eval_vocab": [PIANO_SOLO_CLASS],
            "dataset_name": "maestro",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "has_stem": False,
    },
    "maestro_final": {
            "eval_vocab": [PIANO_SOLO_CLASS],
            "dataset_name": "maestro",
            "train_split": merge_splits(["train", "validation"], dataset_name="maestro"),
            "validation_split": "test",
            "test_split": "test",
            "has_stem": False,
    },
    "guitarset": { # 4 random players for train, 1 for valid, and 1 for test
            "eval_vocab": [GUITAR_SOLO_CLASS],
            "dataset_name": "guitarset",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "has_stem": False,
    },
    "guitarset_pshift": { # guitarset + pitch shift
            "eval_vocab": [GUITAR_SOLO_CLASS],
            "dataset_name": "guitarset",
            "train_split": "train_pshift",
            "validation_split": "validation",
            "test_split": "test",
            "has_stem": False,
    },
    "guitarset_progression": { # progression 1 and 2 as train, progression 3 as test
            "eval_vocab": [GUITAR_SOLO_CLASS],
            "dataset_name": "guitarset",
            "train_split": merge_splits(["progression_1", "progression_2"], dataset_name="guitarset"),
            "validation_split": "progression_3",
            "test_split": "progression_3",
            "has_stem": False,
    },
    "guitarset_progression_pshift": { # guuitarset_progression + pitch shift
            "eval_vocab": [GUITAR_SOLO_CLASS],
            "dataset_name": "guitarset",
            "train_split": merge_splits(["progression_1_pshift", "progression_2_pshift"], dataset_name="guitarset"),
            "validation_split": "progression_3",
            "test_split": "progression_3",
            "has_stem": False,
    },
    "guitarset_minus_bn": { # guuitarset_style + pitch shift
            "eval_vocab": [GUITAR_SOLO_CLASS],
            "dataset_name": "guitarset",
            "train_split": merge_splits(["Funk_pshift", "SS_pshift", "Jazz_pshift", "Rock_pshift"],
                                         dataset_name="guitarset"),
            "validation_split": "BN",
            "test_split": "BN",
            "has_stem": False,
    },
    "guitarset_minus_funk": { # guuitarset_style + pitch shift
            "eval_vocab": [GUITAR_SOLO_CLASS],
            "dataset_name": "guitarset",
            "train_split": merge_splits(["BN_pshift", "SS_pshift", "Jazz_pshift", "Rock_pshift"],
                                         dataset_name="guitarset"),
            "validation_split": "Funk",
            "test_split": "Funk",
            "has_stem": False,
    },
    "guitarset_minus_ss": { # guuitarset_style + pitch shift
            "eval_vocab": GUITAR_SOLO_CLASS,
            "dataset_name": "guitarset",
            "train_split": merge_splits(["BN_pshift", "Funk_pshift", "Jazz_pshift", "Rock_pshift"],
                                         dataset_name="guitarset"),
            "validation_split": "SS",
            "test_split": "SS",
            "has_stem": False,
    },
    "guitarset_minus_jazz": { # guuitarset_style + pitch shift
            "eval_vocab": [GUITAR_SOLO_CLASS],
            "dataset_name": "guitarset",
            "train_split": merge_splits(["BN_pshift", "Funk_pshift", "SS_pshift", "Rock_pshift"],
                                         dataset_name="guitarset"),
            "validation_split": "Jazz",
            "test_split": "Jazz",
            "has_stem": False,
    },
    "guitarset_minus_rock": { # guuitarset_style + pitch shift
            "eval_vocab": [GUITAR_SOLO_CLASS],
            "dataset_name": "guitarset",
            "train_split": merge_splits(["BN_pshift", "Funk_pshift", "SS_pshift", "Jazz_pshift"],
                                         dataset_name="guitarset"),
            "validation_split": "Rock",
            "test_split": "Rock",
            "has_stem": False,
    },
    "guitarset_all": {
            "eval_vocab": [None],
            "dataset_name": "guitarset",
            "train_split": "all",
            "validation_split": None,
            "test_split": None,
            "has_stem": False,
    },
    "enstdrums_dtp": {
            "eval_vocab": [None],
            "eval_drum_vocab": drum_vocab_presets["ksh"],
            "dataset_name": "enstdrums",
            "train_split": merge_splits(["drummer_1_dtp", "drummer_2_dtp", "drummer_1_dtp", "drummer_2_dtp"], dataset_name="enstdrums"),
            "validation_split": "drummer_1_dtp", # for sanity check
            "test_split": "drummer_3_dtp",
            "has_stem": False,
    },
    "enstdrums_dtm": {
            "eval_vocab": [None],
            "eval_drum_vocab": drum_vocab_presets["ksh"],
            "dataset_name": "enstdrums",
            "train_split": merge_splits(["drummer_1_dtm", "drummer_2_dtm", "drummer_1_dtp", "drummer_2_dtp"], dataset_name="enstdrums"),
            "validation_split": "drummer_3_dtm_r2", # 0.6 * drum
            "test_split": "drummer_3_dtm_r1", # 0.75 * drum
            "has_stem": True,
    },
    "enstdrums_random_dtm": { # single dataset training as a denoising ADT model
            "eval_vocab": [None],
            "eval_drum_vocab": drum_vocab_presets["ksh"],
            "dataset_name": "enstdrums",
            "train_split": "train_dtm",
            "validation_split": "validation_dtm",
            "test_split": "test_dtm",
            "has_stem": True,
    },
    "enstdrums_random": { # multi dataset training with random split of 70:15:15
            "eval_vocab": [None],
            "eval_drum_vocab": drum_vocab_presets["ksh"],
            "dataset_name": "enstdrums",
            "train_split": "train_dtp",
            "validation_split": "test_dtm",
            "test_split": "test_dtm",
            "has_stem": True,
    },
    "enstdrums_random_plus_dtd": { # multi dataset training plus dtd
            "eval_vocab": [None],
            "eval_drum_vocab": drum_vocab_presets["ksh"],
            "dataset_name": "enstdrums",
            "train_split": merge_splits(["train_dtp", "all_dtd"], dataset_name="enstdrums"),
            "validation_split": "test_dtm",
            "test_split": "test_dtm",
            "has_stem": True,
    },
    "mir_st500": {
            "eval_vocab": [SINGING_SOLO_CLASS],
            "dataset_name": "mir_st500",
            "train_split": "train_stem",
            "validation_split": "test",
            "test_split": "test",
            "has_stem": True,
    },
    "mir_st500_voc": {
            "eval_vocab": [SINGING_SOLO_CLASS],
            "dataset_name": "mir_st500",
            "train_split": "train_vocal",
            "validation_split": "test_vocal",
            "test_split": "test_vocal",
            "has_stem": False,
    },
    "mir_st500_voc_debug": { # using train_vocal for test (for debugging)
            "eval_vocab": [SINGING_SOLO_CLASS],
            "dataset_name": "mir_st500",
            "train_split": "train_vocal",
            "validation_split": "test_vocal",
            "test_split": "train_vocal",
            "has_stem": False,
    },
    "slakh": {
            "eval_vocab": [GM_INSTR_CLASS],
            "eval_drum_vocab": drum_vocab_presets["gm"],
            "dataset_name": "slakh",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "has_stem": True,
    },
    "slakh_final": {
            "eval_vocab": [GM_INSTR_CLASS],
            "eval_drum_vocab": drum_vocab_presets["gm"],
            "dataset_name": "slakh",
            "train_split": merge_splits(["train", "validation"], dataset_name="slakh"),
            "validation_split": "test",
            "test_split": "test",
            "has_stem": True,
    },
    "rwc_pop_bass": {
            "eval_vocab": [BASS_SOLO_CLASS],
            "add_pitch_class_metric": ["Bass"],
            "dataset_name": "rwc_pop",
            "train_split": None,
            "validation_split": "bass",
            "test_split": "bass",
            "has_stem": False,
    },
    "rwc_pop_full": {
            "eval_vocab": [GM_INSTR_CLASS_PLUS],
            "add_pitch_class_metric": list(GM_INSTR_CLASS_PLUS.keys()),
            "dataset_name": "rwc_pop",
            "train_split": None,
            "validation_split": "full",
            "test_split": "full",
            "has_stem": False,
    },
    "egmd": {
            "eval_vocab": [None],
            "eval_drum_vocab": drum_vocab_presets["ksh"],
            "dataset_name": "egmd",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test_reduced", # EGMD has 5000+ test files, so we reudce it to 200 files to save time
            # "train_limit_num_files": 4402, #8804, # 17608, # limit the number of files for training to random choice of half.
            "has_stem": False,
    },
    "urmp": {
            "eval_vocab": [GM_INSTR_CLASS],
            "dataset_name": "urmp",
            "train_split": "train",
            "validation_split": "test",
            "test_split": "test",
            "has_stem": True,
    },
    "cmedia": {
            "eval_vocab": [SINGING_SOLO_CLASS],
            "dataset_name": "cmedia",
            "train_split": "train_stem",
            "validation_split": "train",
            "test_split": "train",
            "has_stem": True,
    },
    "cmedia_voc": {
            "eval_vocab": [SINGING_SOLO_CLASS],
            "dataset_name": "cmedia",
            "train_split": "train_vocal",
            "validation_split": "train_vocal",
            "test_split": "train_vocal",
            "has_stem": False,
    },
    "idmt_smt_bass": {
            "eval_vocab": [BASS_SOLO_CLASS],
            "dataset_name": "idmt_smt_bass",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "validation",
            "has_stem": False,
    },
    "geerdes": { # full mix dataset for evaluation
            "eval_vocab": [GM_INSTR_CLASS_PLUS],
            "dataset_name": "geerdes",
            "train_split": None,
            "validation_split": None,
            "test_split": "all",
            "has_stem": False,
    },
    "geerdes_sep": { # Using vocal/accomp separation for evalutation
            "eval_vocab": [GM_INSTR_CLASS_PLUS],
            "dataset_name": "geerdes",
            "train_split": None,
            "validation_split": None,
            "test_split": "all_sep",
            "has_stem": False,
    },
    "geerdes_half": { # Using half dataset for train/val
            "eval_vocab": [GM_INSTR_CLASS_PLUS],
            "dataset_name": "geerdes",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "validation",
            "has_stem": False,
    },
    "geerdes_half_sep": { # Using half dataset with vocal/accomp separation for train/val
            "eval_vocab": [GM_INSTR_CLASS_PLUS],
            "dataset_name": "geerdes",
            "train_split": "train_sep",
            "validation_split": "validation_sep",
            "test_split": "validation_sep",
            "has_stem": False,
    },
}

data_preset_multi_cfg = {
    "musicnet_mt3_em_synth_plus_maps": {
        "presets": ["musicnet_mt3_em_synth", "maps_all"],
        "weights": [0.6, 0.4],
        "eval_vocab": [MUSICNET_INSTR_CLASS],
    },
    "musicnet_em_synth_table2_plus_maps": {
        "presets": ["musicnet_em_synth_table2", "maps_all"],
        "weights": [0.6, 0.4],
        "eval_vocab": [MUSICNET_INSTR_CLASS],
    },
    "musicnet_em_synth_table2_plus_maps_multi": {
        "presets": ["musicnet_em_synth_table2", "maps_default"],
        "weights": [0.6, 0.4],
        "eval_vocab": [MUSICNET_INSTR_CLASS],
    },
    "guitarset_progression_plus_maps": {
        "presets": ["guitarset_progression", "maps_all"],
        "weights": [0.5, 0.5],
        "eval_vocab": [GUITAR_SOLO_CLASS],
    },
    "guitarset_pshift_plus_maps": {
        "presets": ["guitarset_pshift", "maps_default"],
        "weights": [0.6, 0.4],
        "eval_vocab": [merge_vocab([GUITAR_SOLO_CLASS, PIANO_SOLO_CLASS])],
    },
    "guitarset_pshift_plus_musicnet_thick": {
        "presets": ["guitarset_pshift", "musicnet_thickstun_em"],
        "weights": [0.5, 0.5],
        "eval_vocab": [merge_vocab([GUITAR_SOLO_CLASS, PIANO_SOLO_CLASS])],
    },
    "multi_sanity_check": {
        "presets": ["musicnet_mt3_synth_only", "musicnet_mt3_synth_only"],
        "weights": [0.6, 0.4],
        "eval_vocab": [MUSICNET_INSTR_CLASS],
    },
    "all_mmegs": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp", "guitarset_pshift"
        ],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "eval_vocab": [None] * 5,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_gt_cv0": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp", "guitarset_minus_bn"
        ],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "eval_vocab": [None] * 5,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_gt_cv1": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_minus_funk"
        ],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "eval_vocab": [None] * 5,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_gt_cv2": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp", "guitarset_minus_ss"
        ],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "eval_vocab": [None] * 5,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_gt_cv3": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_minus_rock"
        ],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "eval_vocab": [None] * 5,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_gt_cv4": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_minus_jazz"
        ],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "eval_vocab": [None] * 5,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_enstdrums_random": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_random", "guitarset"
        ],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "eval_vocab": [None] * 5,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_plus_egmd": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_random_plus_dtd",
            "guitarset", "egmd"
        ],
        "weights": [0.2, 0.2, 0.2, 0.1, 0.1, 0.2],
        "eval_vocab": [None] * 6,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_dtp_egmd": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp", "guitarset", "egmd"
        ],
        "weights": [0.2, 0.2, 0.2, 0.1, 0.1, 0.2],
        "eval_vocab": [None] * 6,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_weighted_slakh": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp", "guitarset_pshift", "egmd"
        ],
        "weights": [0.5, 0.1, 0.1, 0.05, 0.05, 0.2],
        "eval_vocab": [None] * 6,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_weighted_mt3": { # for comparison with MT3
        "presets": [
            "slakh", "musicnet_mt3", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_progression_pshift", "egmd"
        ],
        "weights": [0.5, 0.1, 0.1, 0.05, 0.05, 0.2],
        "eval_vocab": [None] * 6,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_weighted_mt3_em": { # musicnet_mt3_em
        "presets": [
            "slakh", "musicnet_mt3_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_progression_pshift", "egmd"
        ],
        "weights": [0.5, 0.1, 0.1, 0.05, 0.05, 0.2],
        "eval_vocab": [None] * 6,  # None means instrument-agnoßstic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_urmp": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_pshift", "egmd", "urmp"
        ],
        "weights": [0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1],
        "eval_vocab": [None] * 7,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_urmp_mt3": { # for comparison with MT3 including URMP
        "presets": [
            "slakh", "musicnet_mt3", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_progression", "egmd", "urmp"
        ],
        "weights": [0.5, 0.2, 0.1, 0.05, 0.05, 0.0125, 0.1],
        "eval_vocab": [None] * 7,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_urmp_mt3_em": { # musicnet_mt3_em including URMP
        "presets": [
            "slakh", "musicnet_mt3_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_progression", "egmd", "urmp"
        ],
        "weights": [0.5, 0.2, 0.1, 0.05, 0.05, 0.0125, 0.1],
        "eval_vocab": [None] * 7,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_maestro": { # including Mestro and URMP
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.5, 0.1, 0.125, 0.075, 0.025, 0.01, 0.1, 0.1],
        "eval_vocab": [None] * 8,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_maestro_mt3": { # for comparison with MT3 including URMP
        "presets": [
            "slakh", "musicnet_mt3", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_progression", "egmd", "urmp", "maestro"
        ],
        "weights": [0.5, 0.1, 0.1, 0.05, 0.05, 0.0125, 0.1, 0.1],
        "eval_vocab": [None] * 8,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_maestro_mt3_em": { # musicnet_mt3_em including URMP
        "presets": [
            "slakh", "musicnet_mt3_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_progression", "egmd", "urmp", "maestro"
        ],
        "weights": [0.5, 0.1, 0.1, 0.05, 0.05, 0.0125, 0.1, 0.1],
        "eval_vocab": [None] * 8,  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "singing_v1": { # slakh + mir_st500 without spleeter
        "presets": ["slakh", "mir_st500"],
        "weights": [0.8, 0.2],
        "eval_vocab": [None, SINGING_SOLO_CLASS],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_singing_v1": { # for singing-only task
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_stem", "enstdrums_dtp",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.5, 0.1, 0.1, 0.05, 0.05, 0.0125, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_singing_drum_v1": { # for singing-only and drum-only tasks
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_stem", "enstdrums_dtm",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.5, 0.1, 0.1, 0.05, 0.05, 0.0125, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross": { # including Mestro and URMP
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.5, 0.1, 0.125, 0.075, 0.025, 0.01, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_rebal": { # rebalanced for cross-augment, using spleeter
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.4, 0.15, 0.15, 0.075, 0.025, 0.01, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_rebal2": { # rebalanced for cross-augment, using spleeter
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.275, 0.19, 0.19, 0.1, 0.025, 0.02, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_rebal4": { # rebalanced for cross-augment, using spleeter
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.258, 0.19, 0.2, 0.125, 0.022, 0.005, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_rebal5": { # rebalanced for cross-augment, using spleeter
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.295, 0.19, 0.24, 0.05, 0.02, 0.005, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_stem": { # accomp stem for sub-task learning + rebalanced for cross-augment
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_stem", "enstdrums_dtm",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.4, 0.15, 0.15, 0.075, 0.025, 0.01, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_stem_rebal3": { # accomp stem for sub-task learning + rebalanced for cross-augment
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_stem", "enstdrums_dtm",
            "guitarset_pshift", "egmd", "urmp", "maestro"
        ],
        "weights": [0.265, 0.18, 0.21, 0.1, 0.025, 0.02, 0.1, 0.1],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_v6": { # +cmeida +idmt_smt_bass
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset", "egmd", "urmp", "maestro", "idmt_smt_bass", "cmedia_voc",
        ],
        "weights": [0.295, 0.19, 0.19, 0.05, 0.01, 0.005, 0.1, 0.1, 0.01, 0.05],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None, BASS_SOLO_CLASS, SINGING_SOLO_CLASS],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_v6_geerdes": { # +geerdes_half
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset", "egmd", "urmp", "maestro", "idmt_smt_bass", "cmedia_voc",
            "geerdes_half", "geerdes_half_sep"
        ],
        "weights": [0.295, 0.19, 0.19, 0.05, 0.01, 0.005, 0.075, 0.075, 0.01, 0.05, 0.025, 0.025],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None, BASS_SOLO_CLASS,
            SINGING_SOLO_CLASS, GM_INSTR_CLASS_PLUS, GM_INSTR_CLASS_PLUS],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_cross_v6_geerdes_rebal": { # +geerdes_half
            "presets": [
                "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
                "guitarset", "egmd", "urmp", "maestro", "idmt_smt_bass", "cmedia_voc",
                "geerdes_half", "geerdes_half_sep"
            ],
            "weights": [0.245, 0.175, 0.19, 0.05, 0.01, 0.005, 0.075, 0.05, 0.01, 0.05, 0.075, 0.075],
            "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None, BASS_SOLO_CLASS,
                SINGING_SOLO_CLASS, GM_INSTR_EXT_CLASS_PLUS, GM_INSTR_EXT_CLASS_PLUS],  # None means instrument-agnostic F1 for each dataset
            "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
            "val_max_num_files": 20,  # max 20 files per dataset
            "test_max_num_files": None,
        },
   "all_cross_v7": {
        "presets": [
            "slakh", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_progression_pshift", "egmd", "urmp", "maestro", "idmt_smt_bass", "cmedia_voc",
        ],
        "weights": [0.295, 0.19, 0.191, 0.05, 0.01, 0.004, 0.1, 0.1, 0.01, 0.05],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None, BASS_SOLO_CLASS, SINGING_SOLO_CLASS],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
   "all_cross_final": {
        "presets": [
            "slakh_final", "musicnet_thickstun_em", "mir_st500_voc", "enstdrums_dtp",
            "guitarset_progression_pshift", "egmd", "urmp", "maestro_final", "idmt_smt_bass", "cmedia_voc",
        ],
        "weights": [0.295, 0.19, 0.191, 0.05, 0.01, 0.004, 0.1, 0.1, 0.01, 0.05],
        "eval_vocab": [None, None, SINGING_SOLO_CLASS, None, None, None, None, None, BASS_SOLO_CLASS, SINGING_SOLO_CLASS],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "all_eval_final": { # The final evaluation set
        "presets": [
            "slakh", "musicnet_thickstun", "musicnet_thickstun_em", "musicnet_thickstun_ext",
            "musicnet_thickstun_ext_em", "mir_st500_voc", "mir_st500", "enstdrums_dtp",
            "enstdrums_dtm", "guitarset_progression_pshift", "rwc_pop_bass", "maestro", "urmp",
            "maps_default", "rwc_pop_full", # "geerdes", "geerdes_sep",
        ],
        "eval_vocab": [
            GM_INSTR_CLASS, MUSICNET_INSTR_CLASS, MUSICNET_INSTR_CLASS, MUSICNET_INSTR_CLASS,
            MUSICNET_INSTR_CLASS, SINGING_SOLO_CLASS, SINGING_SOLO_CLASS, None,
            None, None, BASS_SOLO_CLASS, PIANO_SOLO_CLASS, GM_INSTR_CLASS,
            PIANO_SOLO_CLASS, GM_INSTR_CLASS_PLUS, # GM_INSTR_CLASS_PLUS, GM_INSTR_CLASS_PLUS
        ],
        "eval_drum_vocab": drum_vocab_presets["ksh"],
    },
    "geerdes_eval": { # Geerdes evaluation sets for models trained without Geerdes.
        "presets": ["geerdes_sep", "geerdes"],
        "eval_vocab": [GM_INSTR_CLASS_PLUS, GM_INSTR_CLASS_PLUS],
        "eval_drum_vocab": drum_vocab_presets["gm"],
    },
    "geerdes_half_eval": { # Geerdes evaluation sets for models trained with Geerdes-half
        "presets": ["geerdes_half_sep", "geerdes_half"],
        "eval_vocab": [GM_INSTR_CLASS_PLUS, GM_INSTR_CLASS_PLUS],
        "eval_drum_vocab": drum_vocab_presets["gm"],
    },
    "minimal": { # slakh + mir_st500 with spleeter
        "presets": ["slakh", "mir_st500_voc"],
        "weights": [0.8, 0.2],
        "eval_vocab": [None, SINGING_SOLO_CLASS],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
    "singing_debug": { # slakh + mir_st500 with spleeter
        "presets": ["mir_st500_voc_debug"],
        "weights": [1.0],
        "eval_vocab": [SINGING_SOLO_CLASS],  # None means instrument-agnostic F1 for each dataset
        "eval_drum_vocab": drum_vocab_presets["ksh"],  # for drums, kick-snare-hihat metric
        "val_max_num_files": 20,  # max 20 files per dataset
        "test_max_num_files": None,
    },
}
