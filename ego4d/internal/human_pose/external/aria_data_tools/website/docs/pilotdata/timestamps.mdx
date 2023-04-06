---
sidebar_position: 7
id: timestamps
title: Timestamps Mapping Data
---
import useBaseUrl from '@docusaurus/useBaseUrl'

# Timestamps Mapping Data

Project Aria devices and multi-view devices operating in proximity to each other (<100m) can leverage [SMPTE timecode](https://en.wikipedia.org/wiki/SMPTE_timecode) to receive a synchronized time clock with sub-millisecond accuracy.

The mapping between local time clock and synchronized time clock for each sequence is stored in the file `synchronization/timestamp_mapping.csv`  which contains two columns of data:

* `deviceTimestampNs` - Timestamps in the device’s local time clock. All devices have their own time clocks which start at different times and potentially progress at different rates.
* `syncedTimestampNs` - Timestamps in the synchronized time clock common to all devices.

This mapping data provides a way to convert timestamps from device local time clock to synchronized time clock and, by extension this also mean that data from multiple devices can be expressed with respect to a common time. The timestamps in `timestamp_mapping.csv` are increased monotonically.

To translate the local timestamp of an arbitrary piece of data recorded by the device, you can use the offset obtained by  searching in the mapping file for the nearest local timestamp and calculating its delta to the synchronized time clock. An implementation of this mechanism is provided in [Aria Data Tool’s code](https://github.com/facebookresearch/Aria_data_tools/blob/main/src/desktop_activities_viewer/DesktopActivitiesViewer.h.)

**Table 1:** *`timestamp_mapping.csv` Structure*

|deviceTimestampNs   |syncedTimestampNs  |
|---   |---   |
|monotonically increasing timestamps in ns    |monotonically increasing timestamps in ns   |


<video width="950" controls>
  <source src={useBaseUrl('video/2wearers_synched.m4v')} type="video/mp4"/>
  Your browser does not support the video tag.
</video>
