"""Vast | Catalog

This module provides the service catalog for Vast.ai by fetching
live offers from the Vast API.  Results are cached to disk and
refreshed periodically.

Unlike static cloud catalogs (AWS, GCP) that pull a pre-built CSV
from a hosted repository, Vast is a dynamic marketplace where
prices change constantly.  We therefore generate the catalog from
the live API on every refresh.
"""

import os
import time
import typing
from typing import Dict, List, Optional, Tuple, Union

from sky import sky_logging
from sky.catalog import common
from sky.utils import resources_utils
from sky.utils import ux_utils

if typing.TYPE_CHECKING:
    import pandas as pd

    from sky.clouds import cloud

logger = sky_logging.init_logger(__name__)

# How often to refresh the catalog from the Vast API (in seconds).
# Vast is a marketplace — prices and availability change frequently.
_REFRESH_INTERVAL_SECS = 2 * 3600  # 2 hours

_CATALOG_PATH = common.get_catalog_path('vast/vms.csv')

_df: Optional['pd.DataFrame'] = None


def _get_df() -> 'pd.DataFrame':
    """Return the Vast catalog DataFrame, fetching from the API if stale."""
    global _df

    # Fast path: already loaded and file is fresh.
    if _df is not None and os.path.exists(_CATALOG_PATH):
        age = time.time() - os.path.getmtime(_CATALOG_PATH)
        if age < _REFRESH_INTERVAL_SECS:
            return _df

    # Check if the cached file is still fresh.
    if os.path.exists(_CATALOG_PATH):
        age = time.time() - os.path.getmtime(_CATALOG_PATH)
        if age < _REFRESH_INTERVAL_SECS:
            import pandas as pd  # pylint: disable=import-outside-toplevel
            _df = pd.read_csv(_CATALOG_PATH)
            return _df

    # Fetch from the live API.
    try:
        from sky.catalog.data_fetchers import (
            fetch_vast)  # pylint: disable=import-outside-toplevel
        new_df = fetch_vast.fetch_catalog()
        if not new_df.empty:
            _df = new_df
            # Persist to disk for caching.
            os.makedirs(os.path.dirname(_CATALOG_PATH), exist_ok=True)
            _df.to_csv(_CATALOG_PATH, index=False)
            logger.debug('Refreshed Vast catalog from API '
                         f'({len(_df)} entries).')
            return _df
        logger.warning('Vast API returned an empty catalog.')
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f'Failed to fetch Vast catalog from API: {e}')

    # Fall back to cached file if API fails.
    if os.path.exists(_CATALOG_PATH):
        import pandas as pd  # pylint: disable=import-outside-toplevel
        _df = pd.read_csv(_CATALOG_PATH)
        return _df

    # Last resort: empty DataFrame so callers don't crash.
    import pandas as pd  # pylint: disable=import-outside-toplevel
    _df = pd.DataFrame(
        columns=common.CATALOG_COLUMNS) if hasattr(  # type: ignore
            common, 'CATALOG_COLUMNS') else pd.DataFrame()
    return _df


def _apply_datacenter_filter(df: 'pd.DataFrame',
                             datacenter_only: bool) -> 'pd.DataFrame':
    """Filter dataframe by hosting_type if datacenter_only is True.

    hosting_type: 0 = Consumer hosted, 1 = Datacenter hosted
    """
    if not datacenter_only or 'HostingType' not in df.columns:
        return df
    return df[df['HostingType'] >= 1]


def instance_type_exists(instance_type: str) -> bool:
    return common.instance_type_exists_impl(_get_df(), instance_type)


def validate_region_zone(
        region: Optional[str],
        zone: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    return common.validate_region_zone_impl('vast', _get_df(), region, zone)


def get_hourly_cost(instance_type: str,
                    use_spot: bool = False,
                    region: Optional[str] = None,
                    zone: Optional[str] = None) -> float:
    """Returns the cost, or the cheapest cost among all zones for spot."""
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    return common.get_hourly_cost_impl(_get_df(), instance_type, use_spot,
                                       region, zone)


def get_vcpus_mem_from_instance_type(
        instance_type: str) -> Tuple[Optional[float], Optional[float]]:
    return common.get_vcpus_mem_from_instance_type_impl(_get_df(),
                                                        instance_type)


def get_default_instance_type(cpus: Optional[str] = None,
                              memory: Optional[str] = None,
                              disk_tier: Optional[
                                  resources_utils.DiskTier] = None,
                              local_disk: Optional[str] = None,
                              region: Optional[str] = None,
                              zone: Optional[str] = None,
                              datacenter_only: bool = False) -> Optional[str]:
    del disk_tier, local_disk
    df = _apply_datacenter_filter(_get_df(), datacenter_only)
    return common.get_instance_type_for_cpus_mem_impl(df, cpus, memory, region,
                                                      zone)


def get_accelerators_from_instance_type(
        instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
    return common.get_accelerators_from_instance_type_impl(
        _get_df(), instance_type)


def get_instance_type_for_accelerator(
        acc_name: str,
        acc_count: int,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        use_spot: bool = False,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        datacenter_only: bool = False) -> Tuple[Optional[List[str]], List[str]]:
    """Returns a list of instance types that have the given accelerator.

    Args:
        datacenter_only: If True, only return instances hosted in datacenters
            (hosting_type >= 1).
    """
    del local_disk  # unused
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    df = _apply_datacenter_filter(_get_df(), datacenter_only)
    return common.get_instance_type_for_accelerator_impl(df=df,
                                                         acc_name=acc_name,
                                                         acc_count=acc_count,
                                                         cpus=cpus,
                                                         memory=memory,
                                                         use_spot=use_spot,
                                                         region=region,
                                                         zone=zone)


def get_region_zones_for_instance_type(instance_type: str,
                                       use_spot: bool) -> List['cloud.Region']:
    df = _get_df()
    df = df[df['InstanceType'] == instance_type]
    return common.get_region_zones(df, use_spot)


# TODO: this differs from the fluffy catalog version
def list_accelerators(
        gpus_only: bool,
        name_filter: Optional[str],
        region_filter: Optional[str],
        quantity_filter: Optional[int],
        case_sensitive: bool = True,
        all_regions: bool = False,
        require_price: bool = True) -> Dict[str, List[common.InstanceTypeInfo]]:
    """Returns all instance types in Vast offering GPUs."""
    del require_price  # Unused.
    return common.list_accelerators_impl('Vast', _get_df(), gpus_only,
                                         name_filter, region_filter,
                                         quantity_filter, case_sensitive,
                                         all_regions)
