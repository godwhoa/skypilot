"""A script that generates the Vast Cloud catalog. """

#
# Due to the design of the sdk, pylint has a false
# positive for the functions.
#
# pylint: disable=assignment-from-no-return
import collections
import csv
import json
import math
import os
import re
from typing import Any, Dict, List

from sky.adaptors import vast

_map = {
    'TeslaV100': 'V100',
    'TeslaT4': 'T4',
    'TeslaP100': 'P100',
    'QRTX6000': 'RTX6000',
    'QRTX8000': 'RTX8000'
}

# Minimum spec thresholds used to filter and normalize offers
# into consistent instance types. These mirror the Vast SDK's
# 'chunked' mode logic, except we intentionally preserve
# min_bid (the SDK zeros it out, breaking spot pricing).
_MIN_CPU_RAM = 64 * 1024  # 64 GiB in MiB
_MIN_CPU_CORES = 32


def create_instance_type(obj: Dict[str, Any]) -> str:
    stubify = lambda x: re.sub(r'\s', '_', x)
    return '{}x-{}-{}-{}'.format(obj['num_gpus'], stubify(obj['gpu_name']),
                                 obj['cpu_cores'], obj['cpu_ram'])


def dot_get(d: dict, key: str) -> Any:
    for k in key.split('.'):
        d = d[k]
    return d


def _median(values: List[float]) -> float:
    """Return the upper-median of a sorted list."""
    values = sorted(values)
    index = math.ceil(0.5 * len(values)) - 1
    return values[index]


if __name__ == '__main__':
    seen: set = set()
    csvList: List[Dict] = []

    mapped_keys = (('gpu_name', 'InstanceType'), ('gpu_name',
                                                  'AcceleratorName'),
                   ('num_gpus', 'AcceleratorCount'), ('cpu_cores', 'vCPUs'),
                   ('cpu_ram', 'MemoryGiB'), ('gpu_name', 'GpuInfo'),
                   ('search.totalHour', 'Price'), ('min_bid', 'SpotPrice'),
                   ('geolocation', 'Region'), ('hosting_type', 'HostingType'))

    # We query WITHOUT the SDK's 'chunked' flag. The SDK's chunked
    # mode zeros out min_bid for all results, which makes every spot
    # price appear as $0.00. Instead we apply the same filtering and
    # normalization ourselves below, preserving min_bid.
    #
    # georegion: consolidates geographic areas into continent codes
    # inet_down >= 100: only machines with reasonable bandwidth
    # disk_space >= 80: only machines with enough disk
    offerList = vast.vast().search_offers(
        query=('georegion = true '
               'inet_down >= 100 disk_space >= 80'),
        limit=10000)

    priceMap: Dict[str, List] = collections.defaultdict(list)
    for offer in (offerList if isinstance(offerList, list) else []):
        # Apply the same filtering the SDK's chunked mode does:
        # skip machines below minimum spec thresholds.
        if (offer.get('cpu_ram') or 0) < _MIN_CPU_RAM:
            continue
        if (offer.get('cpu_cores') or 0) < _MIN_CPU_CORES:
            continue

        # Normalize cpu_ram and cpu_cores to fixed values so
        # machines with slightly different specs bucket into
        # the same instance type. This is what 'chunked' does,
        # minus the min_bid=0 clobber.
        offer['cpu_ram'] = _MIN_CPU_RAM
        offer['cpu_cores'] = _MIN_CPU_CORES

        entry = {}
        for ours, theirs in mapped_keys:
            field = dot_get(offer, ours)
            entry[theirs] = field

        instance_type = create_instance_type(offer)
        entry['InstanceType'] = instance_type

        entry['MemoryGiB'] /= 1024

        gpu = re.sub('Ada', '-Ada', re.sub(r'\s', '', offer['gpu_name']))
        gpu = re.sub(r'(Ti|PCIE|SXM4|SXM|NVL)$', '', gpu)
        gpu = re.sub(r'(RTX\d0\d0)(S|D)$', r'\1', gpu)

        if gpu in _map:
            gpu = _map[gpu]

        entry['AcceleratorName'] = gpu
        entry['GpuInfo'] = json.dumps({
            'Gpus': [{
                'Name': gpu,
                'Count': offer['num_gpus'],
                'MemoryInfo': {
                    'SizeInMiB': offer['gpu_total_ram']
                }
            }],
            'TotalGpuMemoryInMiB': offer['gpu_total_ram']
        }).replace('"', '\'')

        priceMap[instance_type].append(entry)

    for instanceList in priceMap.values():
        # Compute median on-demand price across all offers
        # for this instance type, then keep only offers at
        # or below the median.
        priceList = sorted([x['Price'] for x in instanceList])
        priceTarget = _median(priceList)
        toList: List = []
        for instance in instanceList:
            if instance['Price'] <= priceTarget:
                instance['Price'] = '{:.2f}'.format(priceTarget)
                toList.append(instance)

        # Compute median spot price (min_bid) for this instance
        # type. Only include positive bids.
        spotBids = [
            x['SpotPrice'] for x in toList
            if x.get('SpotPrice') is not None and x['SpotPrice'] > 0
        ]
        spotPrice = f'{_median(spotBids):.2f}' if spotBids else ''

        # Dedup: emit one representative entry per
        # (instance_type, continent, hosting_type) combination.
        # Requires at least two matching offers to confirm
        # the instance type has real availability.
        for instance in toList:
            hosting_type = instance.get('HostingType', 0)
            stub = (f'{instance["InstanceType"]} '
                    f'{instance["Region"][-2:]} {hosting_type}')
            if stub in seen:
                printstub = f'{stub}#print'
                if printstub not in seen:
                    instance['SpotPrice'] = spotPrice
                    csvList.append(instance)
                    seen.add(printstub)
            else:
                seen.add(stub)

    os.makedirs('vast', exist_ok=True)
    with open('vast/vms.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=[x[1] for x in mapped_keys])
        writer.writeheader()

        for instance in csvList:
            writer.writerow(instance)
