"""Vast Cloud catalog fetcher.

Fetches live offers from the Vast.ai API and transforms them into
a catalog DataFrame/CSV compatible with SkyPilot's catalog schema.

Can be used as:
  - A library: ``fetch_catalog()`` returns a ``pd.DataFrame``
  - A script:  ``python -m sky.catalog.data_fetchers.fetch_vast``
    writes ``vast/vms.csv`` in the current directory.
"""

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
import typing
from typing import Any, Dict, List

if typing.TYPE_CHECKING:
    import pandas as pd

from sky.adaptors import vast

# GPU name normalization map.
_GPU_NAME_MAP = {
    'TeslaV100': 'V100',
    'TeslaT4': 'T4',
    'TeslaP100': 'P100',
    'QRTX6000': 'RTX6000',
    'QRTX8000': 'RTX8000',
}

# Minimum spec thresholds used to filter and normalize offers
# into consistent instance types.  These mirror the Vast SDK's
# ``chunked`` mode logic, except we intentionally preserve
# ``min_bid`` (the SDK zeros it out, breaking spot pricing).
_MIN_CPU_RAM = 64 * 1024  # 64 GiB in MiB
_MIN_CPU_CORES = 32

# Column mapping: (vast_api_field, csv_column).
CATALOG_COLUMNS = (
    ('gpu_name', 'InstanceType'),
    ('gpu_name', 'AcceleratorName'),
    ('num_gpus', 'AcceleratorCount'),
    ('cpu_cores', 'vCPUs'),
    ('cpu_ram', 'MemoryGiB'),
    ('gpu_name', 'GpuInfo'),
    ('search.totalHour', 'Price'),
    ('min_bid', 'SpotPrice'),
    ('geolocation', 'Region'),
    ('hosting_type', 'HostingType'),
)

CATALOG_HEADERS = [col for _, col in CATALOG_COLUMNS]


def _create_instance_type(obj: Dict[str, Any]) -> str:
    stubify = lambda x: re.sub(r'\s', '_', x)
    return '{}x-{}-{}-{}'.format(obj['num_gpus'], stubify(obj['gpu_name']),
                                 obj['cpu_cores'], obj['cpu_ram'])


def _dot_get(d: dict, key: str) -> Any:
    for k in key.split('.'):
        d = d[k]
    return d


def _median(values: List[float]) -> float:
    """Return the upper-median of a sorted list."""
    values_sorted = sorted(values)
    index = math.ceil(0.5 * len(values_sorted)) - 1
    return values_sorted[index]


def _normalize_gpu_name(raw_name: str) -> str:
    """Normalize a Vast GPU name to a canonical accelerator name."""
    gpu = re.sub('Ada', '-Ada', re.sub(r'\s', '', raw_name))
    gpu = re.sub(r'(Ti|PCIE|SXM4|SXM|NVL)$', '', gpu)
    gpu = re.sub(r'(RTX\d0\d0)(S|D)$', r'\1', gpu)
    return _GPU_NAME_MAP.get(gpu, gpu)


def _offer_to_entry(offer: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single Vast API offer into a catalog row dict."""
    entry: Dict[str, Any] = {}
    for api_key, col_name in CATALOG_COLUMNS:
        entry[col_name] = _dot_get(offer, api_key)

    entry['InstanceType'] = _create_instance_type(offer)
    entry['MemoryGiB'] = entry['MemoryGiB'] / 1024

    gpu = _normalize_gpu_name(offer['gpu_name'])
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

    return entry


def _fetch_offers() -> List[Dict[str, Any]]:
    """Query the Vast API for current offers.

    We query WITHOUT the SDK's ``chunked`` flag because the SDK's
    chunked mode zeros out ``min_bid`` for all results, making every
    spot price appear as $0.00.  Instead we apply the same filtering
    and normalization ourselves while preserving ``min_bid``.
    """
    offer_list = vast.vast().search_offers(
        query=('georegion = true '
               'inet_down >= 100 disk_space >= 80'),
        limit=10000)
    if not isinstance(offer_list, list):
        return []
    return offer_list


def _build_catalog_rows(offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform raw Vast offers into deduplicated catalog rows."""
    price_map: Dict[str, List[Dict]] = collections.defaultdict(list)

    for offer in offers:
        # Apply the same filtering the SDK's chunked mode does:
        # skip machines below minimum spec thresholds.
        if (offer.get('cpu_ram') or 0) < _MIN_CPU_RAM:
            continue
        if (offer.get('cpu_cores') or 0) < _MIN_CPU_CORES:
            continue

        # Normalize cpu_ram and cpu_cores to fixed values so
        # machines with slightly different specs bucket into
        # the same instance type.
        offer['cpu_ram'] = _MIN_CPU_RAM
        offer['cpu_cores'] = _MIN_CPU_CORES

        entry = _offer_to_entry(offer)
        price_map[entry['InstanceType']].append(entry)

    rows: List[Dict[str, Any]] = []
    seen: set = set()

    for instance_list in price_map.values():
        # Compute median on-demand price, keep offers at or below median.
        prices = sorted([x['Price'] for x in instance_list])
        price_target = _median(prices)
        filtered = []
        for inst in instance_list:
            if inst['Price'] <= price_target:
                inst['Price'] = f'{price_target:.2f}'
                filtered.append(inst)

        # Compute median spot price (min_bid). Only include positive bids.
        spot_bids = [
            x['SpotPrice']
            for x in filtered
            if x.get('SpotPrice') is not None and x['SpotPrice'] > 0
        ]
        spot_price = f'{_median(spot_bids):.2f}' if spot_bids else ''

        # Dedup: one representative entry per
        # (instance_type, continent, hosting_type).
        # Requires at least two matching offers to confirm availability.
        for inst in filtered:
            hosting_type = inst.get('HostingType', 0)
            stub = (f'{inst["InstanceType"]} '
                    f'{inst["Region"][-2:]} {hosting_type}')
            if stub in seen:
                print_stub = f'{stub}#print'
                if print_stub not in seen:
                    inst['SpotPrice'] = spot_price
                    rows.append(inst)
                    seen.add(print_stub)
            else:
                seen.add(stub)

    return rows


def fetch_catalog() -> 'pd.DataFrame':
    """Fetch the Vast catalog from the live API and return a DataFrame.

    This is the primary entry point for ``vast_catalog.py``.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    offers = _fetch_offers()
    rows = _build_catalog_rows(offers)
    if not rows:
        return pd.DataFrame(columns=CATALOG_HEADERS)
    df = pd.DataFrame(rows, columns=CATALOG_HEADERS)
    # Ensure numeric types for the columns the optimizer needs.
    for col in ('AcceleratorCount', 'vCPUs', 'MemoryGiB'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['SpotPrice'] = pd.to_numeric(df['SpotPrice'], errors='coerce')
    return df


def write_catalog(output_dir: str = '.') -> None:
    """Fetch the catalog and write it to ``<output_dir>/vast/vms.csv``."""
    offers = _fetch_offers()
    rows = _build_catalog_rows(offers)

    out_path = os.path.join(output_dir, 'vast')
    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(out_path, 'vms.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    write_catalog()
