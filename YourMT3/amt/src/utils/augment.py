# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""augment.py"""
import numpy as np
import random
from collections import defaultdict
from typing import Optional, Tuple, Union, Callable, Literal, DefaultDict, Set, Any, Dict, List
from utils.note_event_dataclasses import NoteEvent, NoteEventListsBundle
from utils.note2event import check_event_len_from_bundle, mix_note_event_lists_bundle, separate_by_subunit_programs_from_note_event_lists_bundle
from utils.utils import dict_iterator, extend_dict
from copy import deepcopy

EPS = 1e-7
DRUM_PROGRAM = 128
UNANNOTATED_PROGRAM = 129

# -------------------------------------------------------------------------------------
# shared augmentation helper functions
# -------------------------------------------------------------------------------------


def audio_random_submix_fn(x: np.ndarray,
                           random_amp_range: Optional[List[float]] = None,
                           mask: Optional[np.ndarray] = None,
                           normalize: bool = True,
                           dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly submix audio. This function supports batch-wise matrix processing.

    Parameters:
    - x (np.ndarray): Input audio tensor with shape (b, c, t).
    - random_amp_range (List[float], optional): A list containing [min_amp, max_amp]. 
      Defaults to [0.6, 1.2].
    - mask (np.ndarray, optional): Mask tensor with shape (b, c). Defaults to None.
    - dtype (np.dtype): Data type for computations. Defaults to np.float32.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Processed audio (stems, mix).
    """
    b, c, t = x.shape

    if random_amp_range is None:
        random_amp_range = [0.6, 1.2]

    if len(random_amp_range) == 2:
        min_w, max_w = random_amp_range
        ws = np.random.uniform(min_w, max_w, size=(b, c)).astype(dtype)
    else:
        raise ValueError(
            f"random_amp_range should be a list of two floats, [min_amp, max_amp] or None, but got {random_amp_range}")

    if mask is not None:
        ws *= mask  # (b, c)

    processed_audio_stems = x * ws[:, :, np.newaxis]  # (b, c, t)
    processed_audio_mix = np.sum(processed_audio_stems, axis=1, keepdims=True)  # (b, 1, t)

    # Normalize
    if normalize is True:
        norm_factors = np.max(np.abs(processed_audio_mix), axis=2, keepdims=True) + EPS  # (b, 1, 1)
        processed_audio_stems /= norm_factors  # (b, c, t)
        processed_audio_mix /= norm_factors  # (b, 1, t)
    else:
        pass
    return processed_audio_stems, processed_audio_mix


def audio_random_submix_processor(sampled_data: Dict[str, Any],
                                  random_amp_range: List[float] = [0.6, 1.2],
                                  audio_masks: Optional[List[Optional[np.ndarray]]] = None,
                                  update_audio_segments: bool = True,
                                  create_processed_audio_array: bool = True) -> None:
    """Randomly submix audio from sampled data
    
    Args:
        sampled_data: a dictionary containing sampled data.
            ['audio_segments']: a list of audio segments with length B, each element with shape (1, num_stems, T)            
        random_amp_range: a list of two floats, [min_amp, max_amp]
        audio_masks: a list of masks. Each mask is binary vector with shape (num_stems,).
        update_audio_segments: if True (default), update sampled_data["audio_segments"] in-place.
        create_processed_audio_array: if True (default), create a new key "processed_audio_array" in sampled_data for mix audio.
    
    Returns:
        None (processed audio is stored in sampled_data["processed_audio_array"])

    NOTE:
        - This function creates a new key "processed_audio_array" in sampled_data, in-place of `sampled_data`.
        - Input audio should exist in sampled_data["audio_segments"].
        - The created sampled_data["processed_audio_array"] has shape of (B, 1, T)
    """
    if update_audio_segments is False and create_processed_audio_array is False:
        raise ValueError("At least one of update_audio_segments and create_processed_audio_mix should be True.")

    # create a new key "processed_audio" in sampled_data
    b = len(sampled_data["audio_segments"])  # sub-batch size
    t = sampled_data["audio_segments"][0].shape[2]  # audio length

    if create_processed_audio_array is True:
        sampled_data["processed_audio_array"] = np.zeros((b, 1, t), dtype=np.float32)

    # loop over each audio segment
    if audio_masks is None:
        # no audio mask is provided, randomly submix all audio segments
        for i, audio_segment in enumerate(sampled_data["audio_segments"]):
            processed_audio_stems, processed_audio_mix = audio_random_submix_fn(x=audio_segment,
                                                                                random_amp_range=random_amp_range,
                                                                                mask=None)
            if create_processed_audio_array is True:
                sampled_data["processed_audio_array"][i, :, :] = processed_audio_mix
            if update_audio_segments is True:
                sampled_data["audio_segments"][i] = processed_audio_stems

    else:
        # audio mask is provided, randomly submix audio segments based on the audio mask
        for i, (audio_segment, mask) in enumerate(zip(sampled_data["audio_segments"], audio_masks)):
            processed_audio_stems, processed_audio_mix = audio_random_submix_fn(x=audio_segment,
                                                                                random_amp_range=random_amp_range,
                                                                                mask=mask)
            if create_processed_audio_array is True:
                sampled_data["processed_audio_array"][i, :, :] = processed_audio_mix
            if update_audio_segments is True:
                sampled_data["audio_segments"][i] = processed_audio_stems


def drop_random_stems_from_bundle(sampled_data: Dict[str, Any], prob: float = 0.7) -> None:
    """
    Drop stems with a probability of `prob` from a bundle containing `note_event_segments` and 
    `audio_segments`. It also update `programs`, and add `has_unannotated` info. This function 
    serves as a utility for stem-based data augmentation used by `intra_stem_augment_processor`  
    and `cross_stem_augment_processor`. 

    Args:
        sampled_data: A dict of sampled data.
        prob: The probability of dropping stems from the data.

    Returns:
        None. The processed data is stored in-place within the `sampled_data` dictionary.
    
    Update keys in sampled_data (in-place):    
        sampled_data["note_event_segments"]: NoteEventListsBundle
        sampled_data["audio_segments"]: NoteEventListsBundle
        sampled_data["programs_segments"]: a list of list, drum program is 128. updated.
        sampled_data["has_unannotated_segments"]: a list of bool, True if unannotated program 129 is in use. Newly added.


    Removed kyes in sampled_data (in-place):
        all other keys except for the above are removed.

    Function execution time: 16ms for bsz=36 with single worker
    """
    # Create a deep copy to avoid modifying the original data.
    note_event_segments = deepcopy(sampled_data["note_event_segments"])
    has_unannotated = []  # List of bool, True if unannotated program 129 is in use

    for i, (has_stems, note_events, tie_note_events, audio_segment, programs, is_drum) in enumerate(
            zip(sampled_data["has_stems_segments"], note_event_segments['note_events'],
                note_event_segments['tie_note_events'], sampled_data["audio_segments"],
                sampled_data["programs_segments"], sampled_data["is_drum_segments"])):

        # Make sure that programs is np.ndarray
        if not isinstance(programs, np.ndarray):
            programs = np.array(programs)

        if has_stems is True and UNANNOTATED_PROGRAM not in programs:
            # Get unique and actual presence of instruments. 128 means drums, 129 means unannotated.
            uniq_programs = np.unique([ne.program if not ne.is_drum else 128 for ne in (tie_note_events + note_events)])

            # Debug
            if DRUM_PROGRAM in uniq_programs:
                assert DRUM_PROGRAM in programs, "Drum program 128 not in programs"
            if is_drum.any():
                assert DRUM_PROGRAM in programs, "Drum program 128 not in programs"

            # Vectorized random choice for each unique_program
            rand_sel_prgs = uniq_programs[np.random.rand(len(uniq_programs)) < prob]
            if len(rand_sel_prgs) == 0 and len(uniq_programs) != 0:  # Make sure at least one program is active
                rand_sel_prgs = np.random.choice(uniq_programs, size=1)
            programs_mask = np.isin(programs, rand_sel_prgs).astype(np.int32)
            drums_mask = programs_mask * is_drum  # NOTE: if drums are not annotated as program 128, this would not work properly
            _programs_in_use = programs[programs_mask == 1]
            _drum_in_use = np.any(drums_mask == 1)  # True if any drum is in use

            # Drop note_events and tie_note_events in-place
            note_events[:] = [
                ne for ne in note_events
                if (not ne.is_drum and ne.program in _programs_in_use) or (ne.is_drum and _drum_in_use)
            ]
            tie_note_events[:] = [ne for ne in tie_note_events if ne.program in _programs_in_use]

            # Drop stems from audio_segments, update programs_segments
            sampled_data["audio_segments"][i] = audio_segment[:, programs_mask == 1, :]
            sampled_data["programs_segments"][i] = programs[programs_mask == 1]

            # Create has_unannotated
            has_unannotated.append(False)

        elif has_stems is True and UNANNOTATED_PROGRAM in programs:
            # If unannotated program is included in programs, we only drop 129 with a probability of `prob`.
            # `note_event_segments` remains the same.
            # TODO: Actually, we can drop any annoated programs, but current datasets are not the case.
            uniq_programs = np.unique([ne.program if not ne.is_drum else 128 for ne in (tie_note_events + note_events)])
            if np.random.rand() > prob:
                # keep unannotated program, and this will not allow further cross-stem augmentation.
                has_unannotated.append(True)
            else:
                # drop unannotated program
                assert UNANNOTATED_PROGRAM not in uniq_programs  # 129 is not included here...
                sampled_data["audio_segments"][i] = audio_segment[:, programs != 129, :]
                sampled_data["programs_segments"][i] = programs[programs != 129]
                has_unannotated.append(False)

        elif has_stems is False and UNANNOTATED_PROGRAM in programs:
            # No stems, but has unannoted program: cannot be used for cross-stem augmentation.
            has_unannotated.append(True)

        else:
            # No stems, no unannotated program: nothing to do.
            has_unannotated.append(False)

    # Update sampled_data in-place
    sampled_data["note_event_segments"] = note_event_segments
    sampled_data["has_unannotated_segments"] = has_unannotated

    # Remove all other keys except for the above, because they are not used in the downstream pipeline.
    keys_to_remove = ['is_drum_segments', 'has_stems_segments']
    for key in keys_to_remove:
        del sampled_data[key]


# -------------------------------------------------------------------------------------
# intra stem augmentation processor
# -------------------------------------------------------------------------------------
def intra_stem_augment_processor(sampled_data: Dict[str, Any],
                                 random_amp_range: List[float] = [0.6, 1.2],
                                 prob: float = 0.7,
                                 update_audio_segments: bool = True,
                                 submix_audio: bool = True) -> None:
    """
    Intra_stem_augmentation

    Shape of input:
        sampled_data:
            ['note_event_segments']['note_events']:
                List[List[NoteEvent]] with length B, each element is a list of NoteEvent
                with length num_notes
            ['note_event_segments']['tie_note_events']:
                List[List[NoteEvent]] with length B, each element is a list of NoteEvent
                with length num_tie_notes
            ['note_event_segments']['start_times']:
                List[float] with length B
            
            ['audio_segments']: 
                np.ndarray with shape(B, num_stems, T)
            ['programs_segments']: 
                np.ndarray with shape(num_stems,)
            ['is_drum_segments']: 
                np.ndarray with shape(num_stems,)
            ['has_stems_segments']:
                List[bool] with length B
            
    Output (modified in-place):
        sampled_data:
            ['note_event_segments']:
                ['note_events']:
                ['tie_note_events']: 
                ['start_times']: (not modified)
            ['audio_segments']:
                np.ndarray with shape(1, num_stems, T)
            ['processed_audio_array']: # if submix_audio is True
                np.ndarray with shape(B, 1, T)
            ['programs_segments']:
                List[np.ndarray] with length B, each element is a np.ndarray with shape(num_stems,)
            ['has_unannotated_segments']:
                List[bool] with length B
    Execution time: 27 ms for bsz=36 with single worker, including submix audio
    """

    # Randomly drop stems:
    #   - p (0. < p <= 1.) chances to keep each stem, at least one non-drum is guaranteed to be kept.
    #   - This method modifies the input 'note_event_segments' in-place.
    drop_random_stems_from_bundle(sampled_data, prob=prob)

    # Audio processing
    if submix_audio is True:
        # Randomly submix audio, and update audio_segments in-place with random amplitude applied.
        audio_random_submix_processor(sampled_data=sampled_data,
                                      random_amp_range=random_amp_range,
                                      audio_masks=None,
                                      update_audio_segments=True,
                                      create_processed_audio_array=True)  # mix
        # assert "processed_audio_array" in sampled_data.keys()
    else:
        # NOTE: This is used within the cross-stem augmentation pipeline.
        pass


# -------------------------------------------------------------------------------------
# cross-stem augmentation helper functions
# -------------------------------------------------------------------------------------
def combined_survival_and_stop(max_k: int = 5, tau: float = 0.3, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the survival function and prob_stop for exponential or Weibull distributions based on the value of alpha.
    - S(k) represents the probability of "surviving" up to k-th trial.
    - P_stop(k), the stopping probability at trial k is the difference between the survival probabilities at
      k-1 and k. 
    
    Parameters:
    - max_k (int) : Maximum number of trials. k=0, 1, ..., max_k. k=0 means no cross-stem augmentation.
    - tau (float) : Scale parameter. Represents average time to the first failure for exponential distribution.
                   For Weibull distribution, it influences the spread and shape of the distribution.
    - alpha (float) : Shape parameter. If alpha=1, the function reduces to exponential distribution.
                      Otherwise, it represents the Weibull distribution.
                  
    Returns:
    - survival (array-like) : Computed survival function values.
    - prob_stop (array-like) : Computed stop probabilities.

    Example 1:
    >>> survival_exp, stop_exp = combined_survival_and_stop(max_k=5, tau=0.3, alpha=1.0)
    Exponential Survival: [1.         0.74081822 0.54881164 0.40656966 0.30119421 0.22313016]
    Exponential Stop Prob: [0.22313016 0.25918178 0.19200658 0.14224198 0.10537545 0.07806405]
    
    Example 2:
    max_k = 5
    survival_exp, stop_exp_03 = combined_survival_and_stop(max_k, 0.3, 1)
    survival_weibull, stop_weibull = combined_survival_and_stop(max_k, 0.3, 1.5)

    import matplotlib.pyplot as plt
    plt.plot(range(max_k+1), list(stop_exp_03), 'o-', label='Exponential (tau=0.3)')
    plt.plot(range(max_k+1), list(stop_weibull), 's-', label='Weibull (tau=0.3, alpha=1.5)')
    plt.title("Stop Probabilities"); plt.xlabel("k"); plt.ylabel("Probability")
    plt.legend(); plt.grid(True); plt.show()

    References:
    - Weibull, Waloddi. "A statistical distribution function of wide applicability." Journal of applied mechanics (1951).

    """

    # Generate k values based on max_k
    k_values = np.arange(max_k + 1)

    # Calculate survival function
    if alpha == 1:
        survival = np.exp(-k_values * tau)
    else:
        survival = np.exp(-np.power(k_values * tau, alpha))

    # Calculate prob_stop and normalize
    prob_stop_at_k = -np.diff(np.append(survival, 0.))
    return survival, prob_stop_at_k  # (max_k+1,), (max_k+1,)


def deterministic_random_ux_sampler(prob_stop_at_k, bsz) -> np.ndarray:
    """
    Deterministic random sampler for sampling U\X for cross-stem augmentation.

    Args:
        prob_stop_at_k (array-like): Probabilities of stopping at k-th trial.
        bsz (int) : Batch size. Usually local batch size.

    Returns:
        ux_count_per_item (array-like): Number of U\X to sample for each item in the batch.

    Example:
    >>> max_k = 5; tau = 0.3; alpha = 1.0; bsz = 20
    >>> _, prob_stop_at_k = combined_survival_and_stop(max_k, tau, alpha)
    prob_stop_at_k: [0.22313016 0.25918178 0.19200658 0.14224198 0.10537545 0.07806405]
    >>> np.random.choice(np.arange(max_k+1), size=bsz, p=prob_stop_at_k)
    array([1, 4, 1, 3, 0, 3, 0, 2, 5, 0])

    """
    ux_count_per_item = np.random.choice(np.arange(len(prob_stop_at_k)), size=bsz, p=prob_stop_at_k)
    return ux_count_per_item


def check_programs_overlap(list_programs: List[np.ndarray], programs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check if there is any instrument overlap between two lists of programs.

    Example:
    >>> list_programs = np.array([np.array([1,2,3]), np.array([5,6])], dtype=object)
    >>> print(check_programs_overlap(list_programs, np.array([np.array([1,7])], dtype=object)))  # Expected [1]
    >>> print(check_programs_overlap(list_programs, np.array([np.array([])], dtype=object)))     # Expected []
    """
    list_programs_set = set(item for sublist in list_programs for item in sublist)
    overlaps = [p for p in programs if p in list_programs_set]
    uniq_prg_mask = np.array([p not in list_programs_set for p in programs])
    return np.array(overlaps), uniq_prg_mask


def regroup_program_and_audio_by_minimal_shared_subunits(
    gathered_programs: List[np.ndarray],
    gathered_audio_array: List[np.ndarray],
    max_num_groups: Optional[int] = None
) -> Tuple[List[List[int]], DefaultDict[Tuple[int, ...], List[Tuple[int, int]]]]:
    # Check if each audio has stems
    gathered_has_stem = [
        audio_array.shape[1] > 1 for programs, audio_array in zip(gathered_programs, gathered_audio_array)
    ]

    # Create a dictionary for mapping audio to programs
    audio2prg = defaultdict(list)
    for i, programs in enumerate(gathered_programs):
        for j, value in enumerate(programs):
            if gathered_has_stem[i] is True:
                audio2prg[(i, j)].append(value)
            else:
                audio2prg[(i, 0)].append(value)
    grouped_prg2audio = defaultdict(list)
    for k_tuple, v_list in audio2prg.items():
        grouped_prg2audio[tuple(sorted(v_list))].append(k_tuple)
        # defaultdict(list,
        #     {(61, 69, 71, 72): [(0, 0)],
        #      (128,): [(1, 0)], ...}

    # Limit the number of groups
    if max_num_groups is not None:
        # randomly merge groups
        while len(grouped_prg2audio) > max_num_groups:
            # randomly select two groups to merge
            k1, k2 = random.sample(list(grouped_prg2audio.keys()), 2)
            grouped_prg2audio[k1].extend(grouped_prg2audio[k2])
            del grouped_prg2audio[k2]

    grouped_programs = list(grouped_prg2audio.keys())
    return grouped_programs, grouped_prg2audio  # (List[Tuple[int]], DefaultDict[Tuple[int], List[int]])


def audio_random_submix_by_regroup_program_processor(gathered_programs: List[np.ndarray],
                                                     gathered_audio_array: np.ndarray,
                                                     submix_random_amp_range: List[float] = [0.9, 1.0],
                                                     max_num_stems: int = 12) -> Tuple[List[Tuple[int]], np.ndarray]:
    """Regroup programs into subunit programs, and submix regrouped audio arrays
    Return:
        grouped_programs: List[Tuple[int]]
        submix_audio_array: np.ndarray with shape (1, num_grouped_submix_audio, T)
    """

    # Regroup programs into subunit programs
    grouped_programs, grouped_prg2audio = regroup_program_and_audio_by_minimal_shared_subunits(
        gathered_programs, gathered_audio_array, max_num_groups=max_num_stems)

    # Submix subunit audio arrays, based on the regrouped programs
    n_frames = gathered_audio_array[0].shape[2]
    submix_audio_array = np.zeros((1, max_num_stems, n_frames), dtype=np.float32)
    for i, prgs in enumerate(grouped_programs):
        audio_ids = grouped_prg2audio[prgs]  # id of gathered_audio_array, e.g.:[(i,j),...]
        if len(audio_ids) == 1:
            # no need to submix, already subunits
            src_idx, stem_idx = audio_ids[0]
            submix_audio_array[:, i, :] = gathered_audio_array[src_idx][:, [stem_idx], :]
        else:
            # submix audio from elements of subunit programs
            _submix_audio_list = [gathered_audio_array[src_idx][:, [stem_idx], :] for (src_idx, stem_idx) in audio_ids]
            _submix_audio_arr = np.concatenate(_submix_audio_list, axis=1, dtype=np.float32)  # (1, C, T)
            _, _submix_audio_arr = audio_random_submix_fn(_submix_audio_arr,
                                                          random_amp_range=submix_random_amp_range,
                                                          normalize=False)
            submix_audio_array[:, i, :] = _submix_audio_arr
    return [list(prgs) for prgs in grouped_programs], submix_audio_array


# -------------------------------------------------------------------------------------
# cross stem augmentation processor
# -------------------------------------------------------------------------------------
def cross_stem_augment_processor(
        sampled_data: Dict[str, Any],
        sampled_ids: np.ndarray,
        get_rand_segments_from_cache_fn: Callable,
        random_amp_range: List[float] = [0.6, 1.2],
        stem_iaug_prob: float = 0.7,
        stem_xaug_policy: Dict = {
            "max_k": 3,  # max number of external sources used for cross-stem augmentations
            "tau": 0.3,  # exponential decay rate for cross-stem augmentation
            "alpha": 1.0,  # shape parameter for Weibull distribution. set 1.0 for exponential.
            "max_subunit_stems": 12,  # the number of subunit stems to be reduced to
            "p_include_singing":
                0.8,  # probability of including singing for cross augmented examples. if None, use base probaility.
            "no_instr_overlap": True,
            "no_drum_overlap": True,
            "uhat_intra_stem_augment": True,
        },
        max_l: int = 1024,
        precomputed_prob_stop_at_k: Optional[np.array] = None,
        mix_audio: bool = True,
        create_subunit_note_events: bool = False) -> None:
    """
    Cross-stem augmentation

    Args:   
        sampled_data: a dictionary containing sampled data.
            ['note_event_segments']: a list of NoteEventListsBundle with length B
            ['audio_segments']: a list of audio segments with length B, each element with shape (1, num_stems, T)
            ['programs_segments']: a list of programs with length B, each element with shape (num_stems,)
            ['has_unannotated_segments']: a list of bool with length B
        sampled_ids: a numpy array of sampled ids used in sampled_data. (B,)
        get_rand_segments_from_cache_fn: a function for getting random segments from cache.
        random_amp_range: a list of two floats, [min_amp, max_amp]
        stem_iaug_prob: a float, probability of intra-stem augmentation
        stem_xaug_policy: a dictionary of cross-stem augmentation policy
            - max_k (int) : Maximum number of trials. k=0, 1, ..., max_k. k=0 means no cross-stem augmentation.
            - tau (float) : Scale parameter. Represents average time to the first failure for exponential distribution.
                            For Weibull distribution, it influences the spread and shape of the distribution.
            - alpha (float) : Shape parameter. If alpha=1, the function reduces to exponential distribution.
                                Otherwise, it represents the Weibull distribution.
            - max_subunit_stems (int): Maximum number of subunit stems. If larger, they are reduced to this number
                                       by submix. Default: 12
            - p_include_singing (float): Probability of including singing for cross augmented examples. If None, use
                                         base probaility.
            - no_instr_overlap (bool): If True, do not allow instrument overlap between X and U\X.
            - no_drum_overlap (bool): If True, do not allow drum overlap between X and U\X.
            - uhat_intra_stem_augment (bool): If True, apply intra-stem augmentation to U\X.
        max_l: a int, maximum number of note events in a note event list. Default: 1024
        precomputed_prob_stop_at_k: a numpy array of precomputed prob_stop_at_k. If None, it will be computed every time.
        mix_audio: a bool, if True, mix audio from X and U\X. Default: True
        create_subunit_note_events: a bool, if True, create subunit note events. This is necessary for multi channel 
                                    decoder training. Default is False.

    Returns:
        None (processed data is stored in-place within the `sampled_data` dictionary)
    
    Update keys in sampled_data (in-place):
        sampled_data["subunit_programs_segments"]: List[List[np.ndarray]], with length B
        sampled_data["subunit_note_event_segments"]: List[NoteEventListsBundle], with length B
        sampled_data["subunit_audio_array"]: np.ndarray with shape (B, max_subunit_stems, T)
        sampled_data["programs_segments"]: List[np.ndarray], with length B
        sampled_data["note_event_segments"]: NoteEventListsBundle
        sampled_data["has_unannotated_segments"]: List[bool], with length B
        sampled_data["processed_audio_array"]: np.ndarray with shape (B, 1, T)

    Removed kyes in sampled_data (in-place):
        all other keys except for the above are removed.
    """
    # Setup parameters
    max_k = stem_xaug_policy["max_k"]
    tau = stem_xaug_policy["tau"]
    alpha = stem_xaug_policy.get("alpha", 1.0)
    max_subunit_stems = stem_xaug_policy.get("max_subunit_stems", 12)
    p_include_singing = stem_xaug_policy.get("p_include_singing", None)
    no_instr_overlap = stem_xaug_policy["no_instr_overlap"]
    no_drum_overlap = stem_xaug_policy["no_drum_overlap"]
    uhat_intra_stem_augment = stem_xaug_policy["uhat_intra_stem_augment"]
    bsz = len(sampled_ids)  # local batch size
    n_frames = sampled_data["audio_segments"][0].shape[2]

    if precomputed_prob_stop_at_k is None:
        _, prob_stop_at_k = combined_survival_and_stop(max_k, tau, alpha)
    else:
        prob_stop_at_k = precomputed_prob_stop_at_k

    ux_count_per_item = deterministic_random_ux_sampler(prob_stop_at_k, bsz)
    ux_count_sum = int(np.sum(ux_count_per_item))

    # X_in: sampled_data, which we have already applied intra-stem augmentation

    # U\X: ux_sampled_data, complement of X in U
    ux_sampled_data, _ = get_rand_segments_from_cache_fn(
        num_segments=ux_count_sum,
        use_ordered_read_pos=False,  # fully random sampling segments from cache
        sample_excluding_ids=sampled_ids)

    # Randomly drop stems from U\X, and update audio stems without submixing audio.
    if uhat_intra_stem_augment is True:
        intra_stem_augment_processor(sampled_data=ux_sampled_data,
                                     random_amp_range=random_amp_range,
                                     prob=stem_iaug_prob,
                                     update_audio_segments=True,
                                     submix_audio=False)

    # Loop for creating X_hat
    iter_ux = iter(
        zip(
            ux_sampled_data['audio_segments'],
            dict_iterator(ux_sampled_data['note_event_segments']),
            ux_sampled_data['programs_segments'],
            ux_sampled_data['has_unannotated_segments'],
        ))
    iter_x_in = iter(
        zip(
            sampled_data['audio_segments'],
            dict_iterator(sampled_data['note_event_segments']),
            sampled_data['programs_segments'],
            sampled_data['has_unannotated_segments'],
        ))
    x_hat = {
        "subunit_programs_segments": [],  # List[List[np.ndarray]], with length B
        "subunit_note_event_segments": [],  # List[NoteEventListsBundle], with length B
        "subunit_audio_array": np.zeros((bsz, max_subunit_stems, n_frames),
                                        dtype=np.float32),  # (B, max_submix_stems, T)
        "programs_segments": [],  # List[np.ndarray], with length B
        "note_event_segments": {
            "note_events": [],
            "tie_note_events": [],
            "start_times": []
        },  # NoteEventListsBundle
        "has_unannotated_segments": [],  # List[bool], with length B
        "processed_audio_array": np.zeros((bsz, 1, n_frames), dtype=np.float32),  # mixed audio array, B, 1, T)
    }

    for i, (audio_array, ne_bundle, programs, has_unannotated) in enumerate(iter_x_in):
        num_ux_samples = ux_count_per_item[i]
        if num_ux_samples > 0 and has_unannotated is False:
            # gather the main source and k external sources
            gathered_programs = [programs]
            gathered_ne_bundle = ne_bundle  # mutable, but ok because `dict_iterator` yields new dict
            gathered_audio_array = [audio_array]

            for k in range(num_ux_samples):
                # Get next external source
                ex_audio_array, ex_ne_bundle, ex_programs, ex_has_unannotated = next(iter_ux)
                ex_prg_mask = None  # None: no need to mask external programs
                ex_has_stem = bool(ex_audio_array.shape[1] > 1)
                """Criteria for skipping sources"""
                if ex_has_unannotated is True:
                    continue
                """Criteria for instrument overlap and drum overlap """
                instr_overlap, uniq_ex_prg_mask = check_programs_overlap(gathered_programs, ex_programs)
                if no_instr_overlap is True and len(instr_overlap) > 0:
                    if np.any(uniq_ex_prg_mask) and ex_has_stem is True:
                        # mask out non-unique external programs
                        ex_prg_mask = uniq_ex_prg_mask
                    else:
                        # print(i, k, num_ux_samples, ex_programs,
                        #       'Warning: no unique external programs, skip this source')
                        continue  # no unique external programs, skip this source
                else:
                    # programs is already unique or don't care about overlap
                    pass

                if no_drum_overlap is True and no_instr_overlap is False and DRUM_PROGRAM in instr_overlap:
                    non_drum_ex_prg_mask = np.array([prg != DRUM_PROGRAM for prg in ex_programs])
                    if np.any(non_drum_ex_prg_mask):
                        # mask only drum external programs
                        ex_prg_mask = non_drum_ex_prg_mask
                    else:
                        # print(i, k, num_ux_samples, ex_programs,
                        #       'Warning: no non-drum external programs, skip this source')
                        continue  # drum overlapped, but no non-drum programs, skip this source
                else:
                    pass
                """Criteria for stopping iteration with respect to max length"""
                if check_event_len_from_bundle(gathered_ne_bundle, ex_ne_bundle, max_len=max_l) is False:
                    # print(i, k, num_ux_samples, 'Warning: max length reached, stop iteration')
                    break

                # Apply mask and gather
                if ex_prg_mask is None:
                    gathered_programs.append(ex_programs)
                    extend_dict(gathered_ne_bundle, ex_ne_bundle)
                    gathered_audio_array.append(ex_audio_array)
                else:
                    # apply mask to external programs, and add to list
                    ex_programs = ex_programs[ex_prg_mask]
                    gathered_programs.append(ex_programs)

                    # drop note_events with masked programs, and extend dictionary
                    _ex_has_drum = np.any(ex_programs == DRUM_PROGRAM)
                    ex_ne_bundle["note_events"][0] = [
                        ne for ne in ex_ne_bundle["note_events"][0]
                        if (not ne.is_drum and ne.program in ex_programs) or (ne.is_drum and _ex_has_drum)
                    ]
                    ex_ne_bundle["tie_note_events"][0] = [
                        ne for ne in ex_ne_bundle["tie_note_events"][0] if ne.program in ex_programs
                    ]
                    extend_dict(gathered_ne_bundle, ex_ne_bundle)

                    # apply mask to external audio_array, and add to list
                    gathered_audio_array.append(ex_audio_array[:, ex_prg_mask, :])

            # print(gathered_programs)
            # Regroup gathered programs, and cresate submix by subunits programs
            subunit_programs, subunit_audio_array = audio_random_submix_by_regroup_program_processor(
                gathered_programs, gathered_audio_array, max_num_stems=max_subunit_stems)
            mixed_ne_bundle = mix_note_event_lists_bundle(gathered_ne_bundle,
                                                          sort=True,
                                                          start_time_to_zero=True,
                                                          use_deepcopy=True)  #False)

            if create_subunit_note_events is True:
                subunit_ne_bundle = separate_by_subunit_programs_from_note_event_lists_bundle(mixed_ne_bundle,
                                                                                              subunit_programs,
                                                                                              start_time_to_zero=False,
                                                                                              sort=True)
            else:
                subunit_ne_bundle = None
            x_hat["subunit_note_event_segments"].append(subunit_ne_bundle)

            x_hat["subunit_programs_segments"].append(subunit_programs)
            x_hat["subunit_audio_array"][i, :subunit_audio_array.shape[1], :] = subunit_audio_array  # (B, C, T)

            x_hat["programs_segments"].append(np.concatenate(gathered_programs, axis=0))
            extend_dict(x_hat["note_event_segments"], mixed_ne_bundle)
            x_hat["has_unannotated_segments"].append(has_unannotated)
        else:
            num_stems = audio_array.shape[1]
            if num_stems > max_subunit_stems:
                # If num_stems exceeds max_subunit_stems, randomly select max_subunit_stems stems
                subunit_programs, subunit_audio_array = audio_random_submix_by_regroup_program_processor(
                    [programs], [audio_array], max_num_stems=max_subunit_stems)
            else:
                subunit_programs = [programs]
                subunit_audio_array = audio_array
            x_hat["subunit_programs_segments"].append(subunit_programs)
            x_hat["subunit_audio_array"][i, :subunit_audio_array.shape[1], :] = subunit_audio_array

            if create_subunit_note_events is True:
                subunit_ne_bundle = separate_by_subunit_programs_from_note_event_lists_bundle(ne_bundle,
                                                                                              subunit_programs,
                                                                                              start_time_to_zero=True,
                                                                                              sort=True)
            else:
                subunit_ne_bundle = None
            x_hat["subunit_note_event_segments"].append(subunit_ne_bundle)

            x_hat["programs_segments"].append(programs)
            extend_dict(x_hat["note_event_segments"], ne_bundle)
            x_hat["has_unannotated_segments"].append(has_unannotated)

    # Mix subunit audio and update subunit audio arrays
    if mix_audio is True:
        amp_applied_stem_arr, mix_audio_arr = audio_random_submix_fn(x_hat["subunit_audio_array"],
                                                                     random_amp_range=random_amp_range,
                                                                     mask=None,
                                                                     normalize=True)
        x_hat["subunit_audio_array"] = amp_applied_stem_arr  # (B, C, T)
        x_hat["processed_audio_array"] = mix_audio_arr  # (B, 1, T)

    # Update sampled_data in-place
    sampled_data["subunit_programs_segments"] = x_hat["subunit_programs_segments"]
    sampled_data["subunit_note_event_segments"] = x_hat["subunit_note_event_segments"]
    sampled_data["subunit_audio_array"] = x_hat["subunit_audio_array"]
    sampled_data["programs_segments"] = x_hat["programs_segments"]
    sampled_data["note_event_segments"] = x_hat["note_event_segments"]
    sampled_data["has_unannotated_segments"] = x_hat["has_unannotated_segments"]
    sampled_data["processed_audio_array"] = x_hat["processed_audio_array"]
    del sampled_data["audio_segments"]
