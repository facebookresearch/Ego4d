# pyre-unsafe
from dataclasses import dataclass
import csv
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, Callable, Optional, Union
from dataclasses import Field, field as dataclass_field, fields as dataclass_fields
from collections import defaultdict
from credential_s3 import S3Helper
from ffmpeg_utils import create_presigned_url, VideoInfo, get_video_info
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
from threading import Lock

logging.basicConfig(filename='example.log', level=logging.DEBUG)
lock = Lock()

def split_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Splits a full s3_path of the form "s3://bucket/folder/.../file.ext"
    into a tuple of ("bucket", "folder/.../file.ext")

    If this doesn't start with s3:// it will assume that the there is no
    bucket and the path is just returned as the second output
    """
    if s3_path[:5] == "s3://":
        s3_path_components = s3_path[5:].split("/")
        return s3_path_components[0], "/".join(s3_path_components[1:])
    # pyre-fixme[7]: Expected `Tuple[str, str]` but got `Tuple[None, str]`.
    return None, s3_path

class DataclassFieldCaster:
    """
    Class to allow subclasses wrapped in @dataclass to automatically
    cast fields to their relevant type by default.

    Also allows for an arbitrary intialization function to be applied
    for a given field.
    """

    COMPLEX_INITIALIZER = "DataclassFieldCaster__complex_initializer"

    def __post_init__(self) -> None:
        f"""
        This function is run by the dataclass library after '__init__'.

        Here we use this to ensure all fields are casted to their declared types
        and to apply any complex field_initializer functions that have been
        declared via the 'complex_initialized_dataclass_field' method of
        this class.

        A complex field_initializer for a given field would be stored in the
        field.metadata dictionary at:
            key = '{self.COMPLEX_INITIALIZER}' (self.COMPLEX_INITIALIZER)

        """
        for field in dataclass_fields(self):
            value = getattr(self, field.name)
            # First check if the datafield has been set to the declared type or
            # if the datafield has a declared complex field_initializer.
            if (
                not isinstance(value, field.type)
                or DataclassFieldCaster.COMPLEX_INITIALIZER in field.metadata
            ):
                # Apply the complex field_initializer function for this field's value,
                # assert that the resultant type is the declared type of the field.
                if DataclassFieldCaster.COMPLEX_INITIALIZER in field.metadata:
                    setattr(
                        self,
                        field.name,
                        field.metadata[DataclassFieldCaster.COMPLEX_INITIALIZER](value),
                    )
                    assert isinstance(getattr(self, field.name), field.type), (
                        f"'field_initializer' function of {field.name} must return "
                        f"type {field.type} but returned type {type(getattr(self, field.name))}"
                    )
                elif field.type == bool:
                    if value:
                        if type(value) == str and value.lower() in ["true", "false"]:
                            value = value.lower() == "true"
                        else:
                            value = bool(int(value))
                    else:
                        value = False
                    setattr(self, field.name, value)
                elif value is not None:
                    # Otherwise attempt to cast the field's value to its declared type.
                    try:
                        value = field.type(value)
                    except ValueError:
                        value = None
                    setattr(self, field.name, value)
    @staticmethod
    def complex_initialized_dataclass_field(
        field_initializer: Callable, **kwargs
    ) -> Field:
        """
        Allows for the setting of a function to be called on the
        named parameter associated with a field during initialization,
        after __init__() completes.

        Args:
            field_initializer (Callable):
                The function to be called on the field

            **kwargs: To be passed downstream to the dataclasses.field method

        Returns:
            (dataclasses.Field) that contains the field_initializer and kwargs infoÎ
        """
        metadata = kwargs.get("metadata") or {}
        assert DataclassFieldCaster.COMPLEX_INITIALIZER not in metadata
        metadata[DataclassFieldCaster.COMPLEX_INITIALIZER] = field_initializer
        kwargs["metadata"] = metadata
        return dataclass_field(**kwargs)

@dataclass
class Device(DataclassFieldCaster):
    device_id: int
    name: str


@dataclass
class ComponentType(DataclassFieldCaster):
    component_type_id: int
    name: str

@dataclass
class Scenario(DataclassFieldCaster):
    scenario_id: int
    name: str

DATE_FORMAT_STR = "%Y-%m-%d %H:%M:%S"

def DATE_INITIALIZATION_FUNC(a: str):
    # TODO (Support milleseconds if needed)
    a = a.split(".")[0]
    # Fall back to default if not available
    return datetime.strptime(a, DATE_FORMAT_STR) if a else datetime(1900, 1, 1)


def json_load_or_empty_dict(s):
    return json.loads(s) if s else {}


def json_load_or_empty_list(s):
    if s:
        # Some universities double encode thier data as JSON
        # This accounts for that
        while type(s) == str:
            s = json.loads(s)
        return s
    else:
        return []

@dataclass
class VideoMetadata(DataclassFieldCaster):
    university_video_id: str
    university_video_folder_path: str
    number_video_components: int
    # pyre-fixme[8]: Attribute has type `datetime`; used as `Field[typing.Any]`.
    start_date_recorded_utc: datetime = (
        DataclassFieldCaster.complex_initialized_dataclass_field(
            DATE_INITIALIZATION_FUNC
        )
    )
    recording_participant_id: str
    device_id: int
    # pyre-fixme[8]: Attribute has type `Dict[typing.Any, typing.Any]`; used as
    #  `Field[typing.Any]`.
    video_device_settings: dict = (
        DataclassFieldCaster.complex_initialized_dataclass_field(
            json_load_or_empty_dict
        )
    )
    physical_setting_id: str
    # pyre-fixme[8]: Attribute has type `List[typing.Any]`; used as `Field[typing.Any]`.
    video_scenario_ids: list = (
        DataclassFieldCaster.complex_initialized_dataclass_field(  # noqa
            json_load_or_empty_list
        )
    )


@dataclass
class VideoComponentFile(DataclassFieldCaster):
    university_video_id: str
    video_component_relative_path: str
    component_index: int
    is_redacted: bool
    # pyre-fixme[8]: Attribute has type `datetime`; used as `Field[typing.Any]`.
    start_date_recorded_utc: datetime = (
        DataclassFieldCaster.complex_initialized_dataclass_field(
            DATE_INITIALIZATION_FUNC
        )
    )
    # pyre-fixme[8]: Attribute has type `Dict[typing.Any, typing.Any]`; used as
    #  `Field[typing.Any]`.
    compression_settings: dict = (
        DataclassFieldCaster.complex_initialized_dataclass_field(
            json_load_or_empty_dict
        )
    )
    includes_audio: bool
    # pyre-fixme[8]: Attribute has type `Dict[typing.Any, typing.Any]`; used as
    #  `Field[typing.Any]`.
    component_metadata: dict = (
        DataclassFieldCaster.complex_initialized_dataclass_field(  # noqa
            json_load_or_empty_dict
        )
    )
    # pyre-fixme[8]: Attribute has type `Dict[typing.Any, typing.Any]`; used as
    #  `Field[typing.Any]`.
    deidentification_metadata: dict = (
        DataclassFieldCaster.complex_initialized_dataclass_field(
            json_load_or_empty_dict
        )
    )


@dataclass
class AuxiliaryVideoComponentDataFile(DataclassFieldCaster):
    university_video_id: str
    component_index: int
    component_type_id: int
    video_component_relative_path: str


@dataclass
class Particpant(DataclassFieldCaster):
    participant_id: str
    # pyre-fixme[8]: Attribute has type `Dict[typing.Any, typing.Any]`; used as
    #  `Field[typing.Any]`.
    participant_metadata: dict = (
        DataclassFieldCaster.complex_initialized_dataclass_field(
            json_load_or_empty_dict
        )
    )


@dataclass
class SynchronizedVideos(DataclassFieldCaster):
    video_grouping_id: str
    # pyre-fixme[8]: Attribute has type `Dict[typing.Any, typing.Any]`; used as
    #  `Field[typing.Any]`.
    synchronization_metadata: dict = (
        DataclassFieldCaster.complex_initialized_dataclass_field(
            json_load_or_empty_dict
        )
    )
    # pyre-fixme[8]: Attribute has type `Dict[typing.Any, typing.Any]`; used as
    #  `Field[typing.Any]`.
    associated_videos: dict = (
        DataclassFieldCaster.complex_initialized_dataclass_field(  # noqa
            json_load_or_empty_dict
        )
    )


@dataclass
class PhysicalSetting(DataclassFieldCaster):
    setting_id: str
    name: str
    associated_matterport_scan_path: str


@dataclass
class Annotations(DataclassFieldCaster):
    university_video_id: str
    start_seconds: float
    end_seconds: float
    # pyre-fixme[8]: Attribute has type `Dict[typing.Any, typing.Any]`; used as
    #  `Field[typing.Any]`.
    annotation_data: dict = (
        DataclassFieldCaster.complex_initialized_dataclass_field(  # noqa
            json_load_or_empty_dict
        )
    )

def load_dataclass_dict_from_csv(
    input_csv_file_path: str,
    dataclass_class: type,
    dict_key_field: str,
    list_per_key: bool = False,
) -> Dict[Any, Union[Any, List[Any]]]:
    """
    Args:
        input_csv_file_path (str): File path of the csv to read from
        dataclass_class (type): The dataclass to read each row into.
        dict_key_field (str): The field of 'dataclass_class' to use as
            the dictionary key.
        list_per_key (bool) = False: If the output data structure
        contains a list of dataclass objects per key, rather than a
        single unique dataclass object.

    Returns:
        Dict[Any, Union[Any, List[Any]] mapping from the dataclass
        value at attr = dict_key_field to either:

        if 'list_per_key', a list of all dataclass objects that
        have equal values at attr = dict_key_field, equal to the key

        if not 'list_per_key', the unique dataclass object
        for which the value at attr = dict_key_field is equal to the key

    Raises:
        AssertionError: if not 'list_per_key' and there are
        dataclass obejcts with equal values at attr = dict_key_field
    """

    output_dict = defaultdict(list) if list_per_key else {}
    with open(input_csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        column_index = {header: i for i, header in enumerate(next(reader))}
        for line in tqdm(reader):
            datum = dataclass_class(
                *(
                    line[column_index[field.name]]
                    for field in dataclass_fields(dataclass_class)
                    if field.name in column_index
                )
            )
            dict_key = getattr(datum, dict_key_field)
            if list_per_key:
                output_dict[dict_key].append(datum)
            else:
                assert (
                    dict_key not in output_dict
                ), f"Multiple entries for {dict_key} in {dataclass_file}"
                output_dict[dict_key] = datum
    return output_dict

def load_standard_metadata_files(
    standard_metadata_folder: str,
) -> Tuple[Dict[str, Device], Dict[str, ComponentType], Dict[str, Scenario]]:
    # Load consortium-wide metadata into memory
    devices = load_dataclass_dict_from_csv(
        f"{standard_metadata_folder}/device.csv", Device, "device_id"
    )
    component_types = load_dataclass_dict_from_csv(
        f"{standard_metadata_folder}/component_type.csv",
        ComponentType,
        "component_type_id",
    )
    scenarios = load_dataclass_dict_from_csv(
        f"{standard_metadata_folder}/scenario.csv", Scenario, "scenario_id"
    )
    validate_standard_metadata_files(devices, component_types, scenarios)
    return devices, component_types, scenarios


def validate_standard_metadata_files(
    devices: Dict[str, Device],
    component_types: Dict[str, ComponentType],
    scenarios: Dict[str, Scenario],
) -> None:
    # At the time of writing code these are the lengths of the files
    assert len(devices) >= 8
    assert len(component_types) >= 4
    assert len(scenarios) >= 108


def load_university_files(
    s3,
    s3_bucket_name: str,
    metadata_folder_prefix: str,
) -> Tuple[
    Dict[str, VideoMetadata],
    Dict[str, List[VideoComponentFile]],
    Dict[str, List[AuxiliaryVideoComponentDataFile]],
    Dict[str, Particpant],
    Dict[str, SynchronizedVideos],
    Dict[str, PhysicalSetting],
    Dict[str, Annotations],
]:

    if s3_bucket_name:
        s3_bucket = S3Helper(s3, s3_bucket_name)
        truncated, available_metadata_files = s3_bucket.ls(metadata_folder_prefix)
        assert not truncated, (
            f"Folder {s3_bucket_name}/{metadata_folder_prefix} " "has too many entries"
        )
        available_files = {f.key.split("/")[-1] for f in available_metadata_files}
    else:
        available_files = os.listdir(metadata_folder_prefix)



    with tempfile.TemporaryDirectory("consortium_ingester") as tempdir:
            # Load reqiured files
            # Load video_metadata.csv
        file_name = "video_metadata.csv"
        assert file_name in available_files, (
            f"required file {file_name} not found in "
            f"{s3_bucket_name}/{metadata_folder_prefix}"
        )
        if s3_bucket_name:
            local_file_path = f"{tempdir}/{file_name}"
            s3_bucket.get_file(f"{metadata_folder_prefix}/{file_name}", local_file_path)
        else:
            local_file_path = f"{metadata_folder_prefix}/{file_name}"
        video_metadata_dict = load_dataclass_dict_from_csv(
            local_file_path,
            VideoMetadata,
            "university_video_id",
        )

        # Load video_component_file.csv
        file_name = "video_component_file.csv"
        assert file_name in available_files, (
            f"required file {file_name} not found in "
            f"{s3_bucket_name}/{metadata_folder_prefix}"
        )
        if s3_bucket_name:
            local_file_path = f"{tempdir}/{file_name}"
            s3_bucket.get_file(f"{metadata_folder_prefix}/{file_name}", local_file_path)
        else:
            local_file_path = f"{metadata_folder_prefix}/{file_name}"
        video_components_dict = load_dataclass_dict_from_csv(
            local_file_path,
            VideoComponentFile,
            "university_video_id",
            list_per_key=True,
        )

        # Load optional files
        # Load auxiliary_video_component_data_file.csv, if available
        file_name = "auxiliary_video_component_data_file.csv"
        auxiliary_video_component_dict = {}
        if file_name in available_files:
            if s3_bucket_name:
                local_file_path = f"{tempdir}/{file_name}"
                s3_bucket.get_file(f"{metadata_folder_prefix}/{file_name}", local_file_path)
            else:
                local_file_path = f"{metadata_folder_prefix}/{file_name}"
            auxiliary_video_component_dict = load_dataclass_dict_from_csv(
                local_file_path,
                AuxiliaryVideoComponentDataFile,
                "university_video_id",
                list_per_key=True,
            )

        # Load participant.csv, if available
        file_name = "participant.csv"
        participant_dict = {}
        if file_name in available_files:
            if s3_bucket_name:
                local_file_path = f"{tempdir}/{file_name}"
                s3_bucket.get_file(f"{metadata_folder_prefix}/{file_name}", local_file_path)
            else:
                local_file_path = f"{metadata_folder_prefix}/{file_name}"
            participant_dict = load_dataclass_dict_from_csv(
                local_file_path,
                Particpant,
                "participant_id",
            )

        # Load synchronized_videos.csv, if available
        file_name = "synchronized_videos.csv"
        synchronized_video_dict = {}
        if file_name in available_files:
            if s3_bucket_name:
                local_file_path = f"{tempdir}/{file_name}"
                s3_bucket.get_file(f"{metadata_folder_prefix}/{file_name}", local_file_path)
            else:
                local_file_path = f"{metadata_folder_prefix}/{file_name}"
            synchronized_video_dict = load_dataclass_dict_from_csv(
                local_file_path,
                SynchronizedVideos,
                "video_grouping_id",
            )

            # Load physical_setting.csv, if available
        file_name = "physical_setting.csv"
        physical_setting_dict = {}
        if file_name in available_files:
            if s3_bucket_name:
                local_file_path = f"{tempdir}/{file_name}"
                s3_bucket.get_file(f"{metadata_folder_prefix}/{file_name}", local_file_path)
            else:
                local_file_path = f"{metadata_folder_prefix}/{file_name}"
            physical_setting_dict = load_dataclass_dict_from_csv(
                local_file_path,
                PhysicalSetting,
                "setting_id",
            )

            # Load annotations.csv, if available
        file_name = "annotations.csv"
        annotations_dict = {}
        if file_name in available_files:
            if s3_bucket_name:
                local_file_path = f"{tempdir}/{file_name}"
                s3_bucket.get_file(f"{metadata_folder_prefix}/{file_name}", local_file_path)
            else:
                local_file_path = f"{metadata_folder_prefix}/{file_name}"
            annotations_dict = load_dataclass_dict_from_csv(
                local_file_path,
                Annotations,
                "university_video_id",
            )

    return (
        video_metadata_dict,
        video_components_dict,
        auxiliary_video_component_dict,
        participant_dict,
        synchronized_video_dict,
        physical_setting_dict,
        annotations_dict,
    )

def validate_university_files(  # noqa :C901
    video_metadata_dict: Dict[str, VideoMetadata],
    video_components_dict: Dict[str, List[VideoComponentFile]],
    auxiliary_video_component_dict: Dict[str, List[AuxiliaryVideoComponentDataFile]],
    participant_dict: Dict[str, Particpant],
    synchronized_video_dict: Dict[str, SynchronizedVideos],
    physical_setting_dict: Dict[str, PhysicalSetting],
    annotations_dict: Dict[str, Annotations],
    devices: Dict[str, Device],
    component_types: Dict[str, ComponentType],
    scenarios: Dict[str, Scenario],
    bucket_name: str,
    s3
) -> None:
    error_message = ""
    # Check ids in video_components_dict are in video_metadata_dict
    # and the # of components is correct
    s3_bucket = S3Helper(s3, bucket_name)
    video_info_param = []

    for video_id, components in video_components_dict.items():
        if video_id not in video_metadata_dict:
            error_message += (
                "video_component_file points to video_metadata at"
                f" non-existent university_video_id: '{video_id}'\n"
            )
        elif video_metadata_dict[video_id].number_video_components != len(components):
            error_message += (
                f"video_metadata: '{video_id}'has {len(components)} components when it"
                f" should have {video_metadata_dict[video_id].number_video_components}\n"
            )
        components.sort(key=lambda x: x.component_index)
        #check component_index is incremental and starts at 0
        university_video_folder_path = video_metadata_dict[video_id].university_video_folder_path
        for i in range(len(components)):
            if components[i].component_index != i:
                error_message += (
                f"video_metadata: '{video_id}' has component index {components[i].component_index}"
                f" when it should have {i}\n"
            )

            #check component S3 files
            s3_path = f"{university_video_folder_path.rstrip('/')}/{components[i].video_component_relative_path}"
            bucket, key = split_s3_path(s3_path)
            if bucket != bucket_name:
                error_message += (
                f"video_metadata: '{video_id}' has wrong bucket name"
            )
            # if not s3_bucket.exist(bucket, key):
            #     error_message += (
            #     f"video_component: '{s3_path}' does not exist in bucket"
            # )
            video_info_param.append((video_id, key))

    #   check all mp4 data
    video_info_dict = defaultdict(list)

    def thread_helper(x):
        video_info = get_video_info(s3, bucket_name, x[1])
        lock.acquire()
        video_info_dict[x[0]].append(video_info)
        lock.release()

    with ThreadPoolExecutor(max_workers=15) as pool:
        tqdm(
            pool.map(
                    thread_helper,
                    video_info_param
                ),
            total = len(video_info_param)
        )

    for video_id, video_infos in video_info_dict.items():
        total_codec = set()
        total_dimension = set()
        for video_info in video_infos:
            if video_info.codec != None:
                total_codec.add(video_info.codec)
                if len(total_codec) > 1:
                    logging.error(
                        f"video_metadata: '{video_id}' has components with inconsistent codec"
                    )
            if video_info.fps == None:
                logging.warning(
                        f"video_metadata: '{video_id}' has components with null fps value"
                    )



    if synchronized_video_dict:
        for video_grouping_id, components in synchronized_video_dict.items():
            for component in components:
                for video_id, num in component.associated_videos:
                    if video_id not in video_metadata_dict:
                        error_message += (
                            "synchronized_video_file points to video_metadata at"
                            f" non-existent university_video_id: '{video_id}'\n"
                        )

    # Check ids in auxiliary_video_component_dict are in video_metadata_dict
    # and that the component_type is valid
    if auxiliary_video_component_dict:
        for video_id, components in auxiliary_video_component_dict.items():
            if video_id not in video_metadata_dict:
                error_message += (
                    "auxiliary_video_component_data_file points to video_metadata at"
                    f" non-existent university_video_id: '{video_id}'\n"
                )
            elif video_metadata_dict[video_id].number_video_components != len(components):
                error_message += (
                    f"video_metadata: '{video_id}' has {len(components)} auxiliary components when it"
                    f" should have {video_metadata_dict[video_id].number_video_components}\n"
                )
            components.sort(key=lambda x: x.component_index)
            for i in range(len(components)):
                if i != components[i].component_index:
                    error_message += (
                        f"video_metadata: '{video_id}' has component index '{components[i].component_index}'"
                        f" when it should have {i}\n"
                    )
                if component.component_type_id not in component_types:
                    error_message += (
                        f"Invalid component_type_id: '{component.component_type_id}'"
                        " for auxillary component of university_video_id:"
                        f" '{video_id}'\n"
                    )

    if participant_dict:
        for participant_id, components in participant_dict.items():
            if participant_id not in video_metadata_dict:
                error_message += (
                    f" participant '{participant_id}' not exist in "
                    f" video metadata'\n"
                )



    # Check ids in annotations are in video_metadata
    if annotations_dict:
        for video_id in annotations_dict:
            if video_id not in video_metadata_dict:
                error_message += (
                    "annotations points to video_metadata at"
                    f" non-existent university_video_id: '{video_id}'\n"
                )

    # Check from video_metadata:
    video_ids = set()
    for video_id, video_metadata in video_metadata_dict.items():
        #   0. check no duplicate university_video_id
        if video_id in video_ids:
             error_message += (
                    f"Video: {video_id} has duplicate video id \n"
                )
        video_ids.add(video_id)
        #   1. participant_ids are in participant_dict
        if participant_dict:
            if video_metadata.recording_participant_id not in participant_dict:
                error_message += (
                    f"Video: {video_id} points to participant file at"
                    " non-existent participant_id: "
                    f"'{video_metadata.recording_participant_id}'\n"
                )
            if video_metadata.recording_participant_id is None:
                error_message += (
                    f"Video: {video_id} does not have corresponding"
                    " participant_id \n"
                )

        #   2. scenario_ids are in scenarios
        for scenario_id in video_metadata.video_scenario_ids:
            if scenario_id not in scenarios:
                error_message += (
                    f"Video: {video_id} points to scenarios at"
                    f" non-existent scenario_id: '{scenario_id}'\n"
                )

        #   3. device_ids are in devices
        if video_metadata.device_id and video_metadata.device_id not in devices:
            error_message += (
                f"Video: {video_id} points to devices at"
                f" non-existent device_id: '{video_metadata.device_id}'\n"
            )

        #   4. physical_settings are in physical_setting
        if physical_setting_dict:
            if (
                video_metadata.physical_setting_id
                and video_metadata.physical_setting_id not in physical_setting_dict
            ):
                error_message += (
                    f"Video: {video_id} points to physical setting at"
                    " non-existent physical_setting_id: "
                    f"'{video_metadata.physical_setting_id}'\n"
                )

        #   5. university_video_ids are in components
        if video_id not in video_components_dict:
            error_message += (
                f"Video: {video_id} points to component at"
                " non-existent university_video_id: "
                f"'{video_id}'\n"
            )

    if error_message:
        print(error_message)
