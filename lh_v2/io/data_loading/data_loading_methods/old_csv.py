import datetime
import pathlib as pth
from collections import deque

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import ymd2pydate

from .abstract_class import AbstractDataLoader


def parse_seg_reg(
    data_dict: dict[str, dict[str, dict[dts.AccountType, dict[str, list[str]]]]],
    accounts: list[dts.AccountType],
    segment: dts.ProductType,
    region: dts.LocationType,
) -> dts.AccountGroupInfo:
    new_dict: dict[dts.AccountType, dict[str, float]] = {}
    for pgroup in data_dict.keys():
        for prod in data_dict[pgroup].keys():
            for acc in data_dict[pgroup][prod].keys():
                if acc not in new_dict.keys():
                    new_dict[acc] = {}
                for date, val in zip(
                    data_dict[pgroup][prod][acc]['date'],
                    data_dict[pgroup][prod][acc]['value'],
                ):
                    new_dict[acc][date] = new_dict[acc].get(date, 0.0) + float(val)

    new_dict2: dict[dts.AccountType, dict[str, list[str] | ArrayF]] = {}
    for acc in new_dict.keys():
        new_dict2[acc] = {}
        dates = sorted(list(new_dict[acc].keys()))
        vals = np.zeros((len(dates),), np.float32)
        for k, date in enumerate(dates):
            vals[k] = new_dict[acc][date]
        new_dict2[acc]['dates'] = dates
        new_dict2[acc]['vals'] = vals

    dates = new_dict2[accounts[0]]['dates']
    base_dates = set(dates)
    for acc in accounts[1:]:
        assert base_dates == set(new_dict2[acc]['dates'])

    arr = np.zeros((len(accounts), len(base_dates)), np.float32)
    accounts_dict = {}
    dates_lst: list[dict[datetime.date, int]] = []

    dates_dict_base: dict[datetime.date, int] = {
        ymd2pydate(date): k for k, date in enumerate(dates)
    }

    for k, acc in enumerate(accounts):
        arr[k] = new_dict2[acc]['vals']
        accounts_dict[acc] = k
        dates_lst.append(dates_dict_base.copy())
    return dts.AccountGroupInfo(
        arr=arr,
        account_map=accounts_dict,
        dates=dates_lst,
        segment_type=dts.HierarchyTree(segment),
        region_type=dts.HierarchyTree(region),
    )


class CSVOldDataLoader(AbstractDataLoader):
    def __init__(self, lh_params: params.LighthouseParams) -> None:
        super().__init__(lh_params)
        return

    def load_account_data(self, source: pth.Path) -> dts.AccountGroupInfo:
        # Implementation for loading account data from old CSV format
        assert isinstance(source, pth.Path), 'Given account source is not a pth.Path.'

        with open(str(source), 'r') as f:
            lines: deque[str] = deque(f.readlines())
        init_line = lines.popleft()[:-1].split(',')
        lines_split = []
        for _ in range(len(lines)):
            lines_split.append(lines.popleft()[:-1].split(','))

        region_idx = init_line.index('plant_key')
        segment_idx = init_line.index('bu_key')
        pgroup_idx = init_line.index('product_group')
        prod_idx = init_line.index('product_key')
        acc_idx = init_line.index('account_key')
        date_idx = init_line.index('period_key')
        val_idx = init_line.index('value')

        # this is an affront to god, but it works for now
        # TODO: fix this abomination
        data_dict: dict[
            dts.LocationType,
            dict[
                dts.ProductType,
                dict[str, dict[str, dict[dts.AccountType, dict[str, list[str]]]]],
            ],
        ] = {}
        for line in lines_split:
            if line[region_idx] not in data_dict.keys():
                data_dict[line[region_idx]] = {}
            if line[segment_idx] not in data_dict[line[region_idx]].keys():
                data_dict[line[region_idx]][line[segment_idx]] = {}
            if (
                line[pgroup_idx]
                not in data_dict[line[region_idx]][line[segment_idx]].keys()
            ):
                data_dict[line[region_idx]][line[segment_idx]][line[pgroup_idx]] = {}
            if (
                line[prod_idx]
                not in data_dict[line[region_idx]][line[segment_idx]][
                    line[pgroup_idx]
                ].keys()
            ):
                data_dict[line[region_idx]][line[segment_idx]][line[pgroup_idx]][
                    line[prod_idx]
                ] = {}
            if (
                line[acc_idx]
                not in data_dict[line[region_idx]][line[segment_idx]][line[pgroup_idx]][
                    line[prod_idx]
                ].keys()
            ):
                data_dict[line[region_idx]][line[segment_idx]][line[pgroup_idx]][
                    line[prod_idx]
                ][line[acc_idx]] = {
                    'date': [],
                    'value': [],
                }
            data_dict[line[region_idx]][line[segment_idx]][line[pgroup_idx]][
                line[prod_idx]
            ][line[acc_idx]]['date'].append(line[date_idx])
            data_dict[line[region_idx]][line[segment_idx]][line[pgroup_idx]][
                line[prod_idx]
            ][line[acc_idx]]['value'].append(line[val_idx])

        return parse_seg_reg(
            data_dict=data_dict[self.lh_params.region][self.lh_params.segment],
            accounts=self.lh_params.accounts,
            segment=self.lh_params.segment,
            region=self.lh_params.region,
        )

    def load_driver_data(
        self, source: pth.Path
    ) -> dts.ClassifiedDriverGroups[dts.DriverClassification]:
        assert isinstance(source, pth.Path), 'Given driver source is not a pth.Path.'

        with open(str(source), 'r') as f:
            lines = deque(f.readlines())
        init_line = lines.popleft()[:-1].split(',')
        lines_split: list[list[str]] = []
        for _ in range(len(lines)):
            lines_split.append(lines.popleft()[:-1].split(','))

        class_idx = init_line.index('driver_classification')
        name_idx = init_line.index('driver_unique_name')
        date_idx = init_line.index('date')
        val_idx = init_line.index('driver_value')
        denorm_idx = init_line.index('denorm')

        # this is also an affront to god, just a lesser sin
        # TODO: also fix this
        # just use polars
        data_dict: dict[
            dts.DriverClassification, dict[dts.DriverName, dict[str, float]]
        ] = {}
        for line in lines_split:
            class_ = dts.DriverClassification(line[class_idx])
            driver_name = dts.DriverName(line[name_idx])
            if class_ not in data_dict.keys():
                data_dict[class_] = {}
            if driver_name not in data_dict[class_].keys():
                data_dict[class_][driver_name] = {}
            data_dict[class_][driver_name][line[date_idx]] = float(
                line[val_idx]
            ) * float(line[denorm_idx])

        classes = sorted(list(data_dict.keys()))
        drivers = sorted(list(data_dict[classes[0]].keys()))
        dates_set = set(data_dict[classes[0]][drivers[0]].keys())
        sorted_drivers: dict[dts.DriverClassification, list[dts.DriverName]] = {}
        c: int = 0
        for class_ in classes:
            drivers = sorted(data_dict[class_].keys())
            sorted_drivers[class_] = drivers
            for driver in drivers:
                assert dates_set == set(data_dict[class_][driver].keys())
                c += 1

        dates = sorted(list(dates_set))
        arr: ArrayF = np.zeros((c, len(dates)), np.float32)

        c: int = 0
        maps: list[dict[dts.DriverName, int]] = []
        for i, class_ in enumerate(classes):
            maps.append({})
            for driver in sorted_drivers[class_]:
                maps[i][driver] = c
                for j, date in enumerate(dates):
                    arr[c, j] = data_dict[class_][driver][date]
                c += 1

        classifications: dict[dts.DriverClassification, int] = {
            class_: idx for idx, class_ in enumerate(classes)
        }
        dates_base_dict = {ymd2pydate(date): idx for idx, date in enumerate(dates)}
        dates_lst: list[dict[datetime.date, int]] = []
        for _ in range(arr.shape[0]):
            dates_lst.append(dates_base_dict.copy())

        return dts.ClassifiedDriverGroups(
            arr=arr,
            classification_groups=classifications,
            maps=maps,
            dates=dates_lst,
        )
