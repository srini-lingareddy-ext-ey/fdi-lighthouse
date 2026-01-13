from typing import Any, Sequence


def flip_dict(dictionary: dict[Any, Any]) -> dict[Any, Any]:
    return {v: k for k, v in dictionary.items()}


def flip_seq_dicts(seq_dicts: Sequence[dict[Any, Any]]) -> list[dict[Any, Any]]:
    new_lst: list[dict[Any, Any]] = []
    for dic in seq_dicts:
        new_lst.append(flip_dict(dic))
    return new_lst
