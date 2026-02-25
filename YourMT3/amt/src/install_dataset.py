# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
""" install_dataset.py """
import os
import argparse
import mirdata
from typing import Optional, Tuple, Union
from utils.preprocess.generate_dataset_stats import generate_dataset_stats_for_all_datasets, update_dataset_stats_for_new_dataset
from utils.mirdata_dev.datasets import slakh16k
from utils.preprocess.preprocess_slakh import preprocess_slakh16k, add_program_and_is_drum_info_to_file_list
from utils.preprocess.preprocess_musicnet import preprocess_musicnet16k
from utils.preprocess.preprocess_maps import preprocess_maps16k
from utils.preprocess.preprocess_maestro import preprocess_maestro16k
from utils.preprocess.preprocess_guitarset import preprocess_guitarset16k, create_filelist_by_style_guitarset16k
from utils.preprocess.preprocess_enstdrums import preprocess_enstdrums16k, create_filelist_dtm_random_enstdrums16k
from utils.preprocess.preprocess_mir_st500 import preprocess_mir_st500_16k
from utils.preprocess.preprocess_cmedia import preprocess_cmedia_16k
from utils.preprocess.preprocess_rwc_pop_full import preprocess_rwc_pop_full16k
from utils.preprocess.preprocess_rwc_pop import preprocess_rwc_pop16k
from utils.preprocess.preprocess_egmd import preprocess_egmd16k
from utils.preprocess.preprocess_mir1k import preprocess_mir1k_16k
from utils.preprocess.preprocess_urmp import preprocess_urmp16k
from utils.preprocess.preprocess_idmt_smt_bass import preprocess_idmt_smt_bass_16k
from utils.preprocess.preprocess_geerdes import preprocess_geerdes16k
from utils.utils import download_and_extract  #, download_and_extract_zenodo_restricted

# zenodo_token = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxMDE1MDYzNywiZXhwIjoxNzEyNzA3MTk5fQ.eyJpZCI6ImRmODA5NzZlLTBjM2QtNDk5NS05YjM0LWFiNGM4NzJhMmZhMSIsImRhdGEiOnt9LCJyYW5kb20iOiIwMzY5ZDcxZjc2NTMyN2UyYmVmN2ExYjJkMmMyYTRhNSJ9.0aHnNC-7ivWQO6l8twjLR0NDH4boC0uOolAAmogVt7XRi2PHU5MEKBQoK7-wgDdnmWEIqEIvoLO6p8KTnsY9dg"


def install_slakh(data_home=os.PathLike, no_down=False) -> None:
    if not no_down:
        ds = slakh16k.Dataset(data_home, version='2100-yourmt3-16k')
        ds.download(partial_download=['2100-yourmt3-16k', 'index'])
        del (ds)
    preprocess_slakh16k(data_home, delete_source_files=False, fix_bass_octave=True)
    add_program_and_is_drum_info_to_file_list(data_home)


def install_musicnet(data_home=os.PathLike, no_down=False) -> None:
    if not no_down:
        url = "https://zenodo.org/record/7811639/files/musicnet_yourmt3_16k.tar.gz?download=1"
        checksum = "a2da7c169e26d452a4e8b9bef498b3d7"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    preprocess_musicnet16k(data_home, dataset_name='musicnet')


def install_maps(data_home=os.PathLike, no_down=False, sanity_check=False) -> None:
    if not no_down:
        url = "https://zenodo.org/record/7812075/files/maps_yourmt3_16k.tar.gz?download=1"
        checksum = "6b070d162c931cd5e69c16ef2398a649"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    preprocess_maps16k(data_home, dataset_name='maps', ignore_pedal=False, sanity_check=sanity_check)


def install_maestro(data_home=os.PathLike, no_down=False, sanity_check=False) -> None:
    if not no_down:
        url = "https://zenodo.org/record/7852176/files/maestro_yourmt3_16k.tar.gz?download=1"
        checksum = "c17c6a188d936e5ff3870ef27144d397"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    preprocess_maestro16k(data_home, dataset_name='maestro', ignore_pedal=False, sanity_check=sanity_check)


def install_guitarset(data_home=os.PathLike, no_down=False) -> None:
    if not no_down:
        url = "https://zenodo.org/record/7831843/files/guitarset_yourmt3_16k.tar.gz?download=1"
        checksum = "e3cfe0cc9394d91d9c290ce888821360"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    preprocess_guitarset16k(data_home, dataset_name='guitarset')
    create_filelist_by_style_guitarset16k(data_home, dataset_name='guitarset')


def install_enstdrums(data_home, no_down=False) -> None:
    if not no_down:
        url = "https://zenodo.org/record/7831843/files/enstdrums_yourmt3_16k.tar.gz?download=1"
        checksum = "7e28c2a923e4f4162b3d83877cedb5eb"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    preprocess_enstdrums16k(data_home, dataset_name='enstdrums')
    create_filelist_dtm_random_enstdrums16k(data_home, dataset_name='enstdrums')


def install_egmd(data_home, no_down=False) -> None:
    if not no_down:
        url = "https://zenodo.org/record/7831072/files/egmc_yourmt3_16k.tar.gz?download=1"
        checksum = "4f615157ea4c52a64c6c9dcf68bf2bde"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    preprocess_egmd16k(data_home, dataset_name='egmd')


def install_mirst500(data_home, zenodo_token, no_down=False, sanity_check=True, apply_correction=False) -> None:
    """ Update Oct 2023: MIR-ST500 with FULL audio files"""
    if not no_down:
        url = "https://zenodo.org/records/10016397/files/mir_st500_yourmt3_16k.tar.gz?download=1"
        checksum = "98eb52eb2456ce4034e21750f309da13"
        download_and_extract(data_home, url, check_sum=checksum, zenodo_token=zenodo_token)
    preprocess_mir_st500_16k(data_home, dataset_name='mir_st500', sanity_check=sanity_check)


def install_cmedia(data_home, zenodo_token, no_down=False, sanity_check=True) -> None:
    if not no_down:
        url = "https://zenodo.org/records/10016397/files/cmedia_yourmt3_16k.tar.gz?download=1"
        checksum = "e6cca23577ba7588e9ed9711a398f7cf"
        download_and_extract(data_home, url, check_sum=checksum, zenodo_token=zenodo_token)
    preprocess_cmedia_16k(data_home, dataset_name='cmedia', sanity_check=sanity_check, apply_correction=True)


def install_rwc_pop(data_home, zenodo_token, no_down=False) -> None:
    if not no_down:
        url = "https://zenodo.org/records/10016397/files/rwc_pop_yourmt3_16k.tar.gz?download=1"
        checksum = "ad459f9fa1b6b87676b2fb37c0ba5dfc"
        download_and_extract(data_home, url, check_sum=checksum, zenodo_token=zenodo_token)
    preprocess_rwc_pop16k(data_home, dataset_name='rwc_pop')  # bass transcriptions
    preprocess_rwc_pop_full16k(data_home, dataset_name='rwc_pop')  # full transcriptions


def install_mir1k(data_home, no_down=False) -> None:
    if not no_down:
        url = "https://zenodo.org/record/7955481/files/mir1k_yourmt3_16k.tar.gz?download=1"
        checksum = "4cbac56a4e971432ca807efd5cb76d67"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    # preprocess_mir1k_16k(data_home, dataset_name='mir1k')


def install_urmp(data_home, no_down=False) -> None:
    if not no_down:
        url = "https://zenodo.org/record/8021437/files/urmp_yourmt3_16k.tar.gz?download=1"
        checksum = "4f539c71678a77ba34f6dfca41072102"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    preprocess_urmp16k(data_home, dataset_name='urmp')


def install_idmt_smt_bass(data_home, no_down=False) -> None:
    if not no_down:
        url = "https://zenodo.org/records/10009959/files/idmt_smt_bass_yourmt3_16k.tar.gz?download=1"
        checksum = "0c95f91926a1e95b1f5d075c05b7eb76"
        download_and_extract(data_home, url, remove_tar_file=True, check_sum=checksum)
    preprocess_idmt_smt_bass_16k(data_home, dataset_name='idmt_smt_bass', sanity_check=True,
                                 edit_audio=False)  # the donwloaded audio has already been edited


def install_random_nsynth(data_home, no_down=False) -> None:
    return


def install_geerdes(data_home) -> None:
    try:
        preprocess_geerdes16k(data_home, dataset_name='geerdes', sanity_check=False)
    except Exception as e:
        print(e)
        print("Geerdes dataset is not available for download. Please contact the dataset provider.")


def regenerate_dataset_stats(data_home) -> None:
    generate_dataset_stats_for_all_datasets(data_home)


def get_cached_zenodo_token() -> str:
    # check if cached token exists
    if not os.path.exists('.cached_zenodo_token'):
        raise Exception("Cached Zenodo token not found. Please enter your Zenodo token.")
    # read cached token
    with open('.cached_zenodo_token', 'r') as f:
        zenodo_token = f.read().strip()
        print(f"Using cached Zenodo token: {zenodo_token}")
    return zenodo_token


def cache_zenodo_token(zenodo_token: str) -> None:
    with open('.cached_zenodo_token', 'w') as f:
        f.write(zenodo_token)
    print("Your Zenodo token is cached.")


def option_prompt(data_home: os.PathLike, no_download: bool = False) -> None:
    print("Select the dataset(s) to install (enter comma-separated numbers):")
    print("1. Slakh")
    print("2. MusicNet")
    print("3. MAPS")
    print("4. Maestro")
    print("5. GuitarSet")
    print("6. ENST-drums")
    print("7. EGMD")
    print("8. MIR-ST500 ** Restricted Access **")
    print("9. CMedia ** Restricted Access **")
    print("10. RWC-Pop (Bass and Full) ** Restricted Access **")
    print("11. MIR-1K (NOT SUPPORTED)")
    print("12. URMP")
    print("13. IDMT-SMT-Bass")
    print("14. Random-NSynth")
    print("15. Geerdes")
    print("16. Regenerate Dataset Stats (experimental)")
    print("17. Request Token for ** Restricted Access **")
    print("18. Exit")

    choice = input("Enter your choices (multiple choices with comma): ")
    choices = [c.strip() for c in choice.split(',')]

    if "18" in choices:
        print("Exiting.")
    else:
        # ask for Zenodo token
        for c in choices:
            if int(c) in [8, 9, 10]:
                if no_download is True:
                    zenodo_token = None
                else:
                    zenodo_token = input("Enter Zenodo token, or press enter to use the cached token:")
                    if zenodo_token == "":
                        zenodo_token = get_cached_zenodo_token()
                    else:
                        cache_zenodo_token(zenodo_token)
                    break

        if "1" in choices:
            install_slakh(data_home, no_down=no_download)
        if "2" in choices:
            install_musicnet(data_home, no_down=no_download)
        if "3" in choices:
            install_maps(data_home, no_down=no_download)
        if "4" in choices:
            install_maestro(data_home, no_down=no_download)
        if "5" in choices:
            install_guitarset(data_home, no_down=no_download)
        if "6" in choices:
            install_enstdrums(data_home, no_down=no_download)
        if "7" in choices:
            install_egmd(data_home, no_down=no_download)
        if "8" in choices:
            install_mirst500(data_home, zenodo_token, no_down=no_download)
        if "9" in choices:
            install_cmedia(data_home, zenodo_token, no_down=no_download)
        if "10" in choices:
            install_rwc_pop(data_home, zenodo_token, no_down=no_download)
        if "11" in choices:
            install_mir1k(data_home, no_down=no_download)
        if "12" in choices:
            install_urmp(data_home, no_down=no_download)
        if "13" in choices:
            install_idmt_smt_bass(data_home, no_down=no_download)
        if "14" in choices:
            install_random_nsynth(data_home, no_down=no_download)
        if "15" in choices:
            install_geerdes(data_home)  # not available for download
        if "16" in choices:
            regenerate_dataset_stats(data_home, no_down=no_download)
        if "17" in choices:
            print("\nPlease visit https://zenodo.org/records/10016397 to request a Zenodo token.")
            print("Upon submitting your request, you will receive an email with a link labeled 'Access the record'.")
            print("Copy the token that follows 'token=' in that link.")
        if not any(int(c) in range(16) for c in choices):
            print("Invalid choice(s). Please enter valid numbers separated by commas.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset installer script.')
    # data home dir
    parser.add_argument(
        'data_home',
        type=str,
        nargs='?',
        default=None,
        help='Path to data home directory. If None, use the default path defined in src/config/config.py')
    # `no_download` option
    parser.add_argument('--nodown',
                        '-nd',
                        action='store_true',
                        help='Flag to control downloading. If set, no downloading will occur.')
    args = parser.parse_args()

    if args.data_home is None:
        from config.config import shared_cfg
        data_home = shared_cfg["PATH"]["data_home"]
    else:
        data_home = args.data_home
    os.makedirs(data_home, exist_ok=True)
    no_download = args.nodown

    option_prompt(data_home, no_download)
