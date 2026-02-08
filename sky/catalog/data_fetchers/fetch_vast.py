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
from typing import Any, Dict, List, Tuple

from sky.adaptors import vast

_map = {
    'TeslaV100': 'V100',
    'TeslaT4': 'T4',
    'TeslaP100': 'P100',
    'QRTX6000': 'RTX6000',
    'QRTX8000': 'RTX8000'
}


def create_instance_type(obj: Dict[str, Any]) -> str:
    stubify = lambda x: re.sub(r'\s', '_', x)
    return '{}x-{}-{}-{}'.format(obj['num_gpus'], stubify(obj['gpu_name']),
                                 obj['cpu_cores'], obj['cpu_ram'])


def dot_get(d: dict, key: str) -> Any:
    for k in key.split('.'):
        d = d[k]
    return d


def _build_spot_price_map(
    spot_offers: List[Dict[str, Any]],
) -> Dict[Tuple, List[float]]:
    """Build a mapping of (gpu_name, num_gpus, region_code, hosting_type)
    to list of min_bid values from a non-chunked query.

    The Vast SDK's 'chunked' mode zeros out the min_bid field,
    so a separate non-chunked query is needed to retrieve real
    spot/bid pricing from the marketplace.
    """
    spot_map: Dict[Tuple, List[float]] = collections.defaultdict(list)
    if not isinstance(spot_offers, list):
        return spot_map
    for offer in spot_offers:
        min_bid = offer.get('min_bid')
        if min_bid is not None and min_bid > 0:
            geolocation = offer.get('geolocation', '')
            region_code = geolocation[-2:] if len(geolocation) >= 2 else ''
            hosting_type = offer.get('hosting_type', 0)
            key = (offer['gpu_name'], offer['num_gpus'], region_code,
                   hosting_type)
            spot_map[key].append(min_bid)
    return spot_map


def _get_spot_price(
    spot_map: Dict[Tuple, List[float]],
    gpu_name: str,
    num_gpus: int,
    region_code: str,
    hosting_type: int,
) -> str:
    """Look up the median spot price from the spot price map.

    Returns a formatted price string, or empty string if no
    spot pricing data is available for the given combination.
    """
    key = (gpu_name, num_gpus, region_code, hosting_type)
    bids = spot_map.get(key, [])
    if not bids:
        return ''
    bids_sorted = sorted(bids)
    index = math.ceil(0.5 * len(bids_sorted)) - 1
    return f'{bids_sorted[index]:.2f}'


if __name__ == '__main__':
    seen = set()
    # InstanceList is the buffered list to emit to
    # the CSV
    csvList = []

    # InstanceType and gpuInfo are basically just stubs
    # so that the dictwriter is happy without weird
    # code.
    mapped_keys = (('gpu_name', 'InstanceType'), ('gpu_name',
                                                  'AcceleratorName'),
                   ('num_gpus', 'AcceleratorCount'), ('cpu_cores', 'vCPUs'),
                   ('cpu_ram', 'MemoryGiB'), ('gpu_name', 'GpuInfo'),
                   ('search.totalHour', 'Price'), ('min_bid', 'SpotPrice'),
                   ('geolocation', 'Region'), ('hosting_type', 'HostingType'))

    # Vast has a wide variety of machines, some of
    # which will have less diskspace and network
    # bandwidth than others.
    #
    # The machine normally have high specificity
    # in the vast catalog - this is fairly unique
    # to Vast and can make bucketing them into
    # instance types difficult.
    #
    # The flags
    #
    #   * georegion consolidates geographic areas
    #
    #   * chunked rounds down specifications (such
    #     as 1025GB to 1024GB disk) in order to
    #     make machine specifications look more
    #     consistent
    #
    #   * inet_down makes sure that only machines
    #     with "reasonable" downlink speed are
    #     considered
    #
    #   * disk_space sets a lower limit of how
    #     much space is availble to be allocated
    #     in order to ensure that machines with
    #     small disk pools aren't listed
    #
    offerList = vast.vast().search_offers(
        query=('georegion = true chunked = true '
               'inet_down >= 100 disk_space >= 80'),
        limit=10000)

    # Second query without 'chunked' to get real min_bid values.
    # The Vast SDK's 'chunked' mode zeros out min_bid for all
    # results, making all spot prices appear as $0.00. This
    # separate call retrieves actual spot/bid pricing.
    spotOfferList = vast.vast().search_offers(
        query='georegion = true inet_down >= 100 disk_space >= 80',
        limit=10000)
    spotPriceMap = _build_spot_price_map(spotOfferList)

    priceMap: Dict[str, List] = collections.defaultdict(list)
    for offer in offerList:
        entry = {}
        for ours, theirs in mapped_keys:
            field = dot_get(offer, ours)
            entry[theirs] = field

        instance_type = create_instance_type(offer)
        entry['InstanceType'] = instance_type

        # the documentation says
        # "{'gpus': [{
        #   'name': 'v100',
        #   'manufacturer': 'nvidia',
        #   'count': 8.0,
        #   'memoryinfo': {'sizeinmib': 16384}
        #   }],
        #   'totalgpumemoryinmib': 16384}",
        # we can do that.
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

        # Store raw offer fields for spot price lookup later.
        # These are removed before writing to CSV.
        entry['_gpu_name'] = offer['gpu_name']
        entry['_num_gpus'] = offer['num_gpus']

        priceMap[instance_type].append(entry)

    for instanceList in priceMap.values():
        priceList = sorted([x['Price'] for x in instanceList])
        index = math.ceil(0.5 * len(priceList)) - 1
        priceTarget = priceList[index]
        toList: List = []
        for instance in instanceList:
            if instance['Price'] <= priceTarget:
                instance['Price'] = '{:.2f}'.format(priceTarget)
                toList.append(instance)

        for instance in toList:
            hosting_type = instance.get('HostingType', 0)
            stub = (f'{instance["InstanceType"]} '
                    f'{instance["Region"][-2:]} {hosting_type}')
            if stub in seen:
                printstub = f'{stub}#print'
                if printstub not in seen:
                    # Look up real spot price from the non-chunked
                    # query instead of using the zeroed min_bid.
                    region_code = instance['Region'][-2:]
                    instance['SpotPrice'] = _get_spot_price(
                        spotPriceMap,
                        instance.pop('_gpu_name'),
                        instance.pop('_num_gpus'),
                        region_code,
                        hosting_type,
                    )
                    csvList.append(instance)
                    seen.add(printstub)
            else:
                seen.add(stub)

    os.makedirs('vast', exist_ok=True)
    with open('vast/vms.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[x[1] for x in mapped_keys])
        writer.writeheader()

        for instance in csvList:
            # Remove internal fields that aren't part of the CSV schema
            instance.pop('_gpu_name', None)
            instance.pop('_num_gpus', None)
            writer.writerow(instance)
