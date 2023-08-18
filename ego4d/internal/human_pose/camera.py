import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pycolmap
from numpy.linalg import inv


Vec2 = np.array
Vec3 = np.array
Mat44 = np.array
AriaCameraModel = Any


@dataclass
class Camera:
    name: str
    camera_type: str
    center: Vec3  # in world coords
    T_device_world: Mat44  # project world pt into device frame
    T_world_device: Mat44  # project device pt into world frame
    T_device_camera: Mat44  # project camera pt into device frame
    T_camera_device: Mat44  # project device pt into camera frame
    camera_model: Union[pycolmap.Camera, AriaCameraModel]  # intrinsics
    device_row: dict  # raw data constructed camera from
    extrinsics: Mat44  # project world pt into camera frame


# https://github.com/colmap/colmap/blob/d6f528ab59fd653966e857f8d0c2203212563631/scripts/python/read_write_model.py#L453
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def _create_exo_camera(exo_cam_desc):
    params = [exo_cam_desc[f"intrinsics_{i}"] for i in range(8)]
    camera = pycolmap.Camera(
        model="OPENCV_FISHEYE",
        width=exo_cam_desc["image_width"],
        height=exo_cam_desc["image_height"],
        params=params,
    )
    return camera


def create_camera(camera_data, camera_model):
    ret = copy.deepcopy(camera_data)
    if camera_data["camera_type"] == "aria":
        ret["camera_model"] = camera_model
    else:
        ret["camera_model"] = _create_exo_camera(ret["device_row"])

    for key in [
        "center",
        "T_device_world",
        "T_world_device",
        "T_device_camera",
        "T_camera_device",
    ]:
        ret[key] = np.array(camera_data[key])

    ret["extrinsics"] = (
        ret["T_camera_device"] @ ret["T_device_world"]
    )  ## map world point to camera frame
    return Camera(**ret)


def create_camera_data(
    device_row: Dict[str, Any],
    name: str,
    camera_model: Optional[Any],
    device_row_key: str,
) -> Dict[str, Any]:
    if camera_model is None:
        assert "cam" in name or "gp" in name, f"Unrecognized camera name: {name}"
        T_device_camera = np.eye(4)
        T_camera_device = np.eye(4)
        camera_model = _create_exo_camera(device_row)
        camera_type = "colmap"
    else:
        try:
            R = camera_model.get_transform_device_camera().rotation_matrix()
            t = camera_model.get_transform_device_camera().translation()
        except AttributeError:
            # print("[Warning] fall back to use old version of projectaria_tools")
            R = camera_model.T_Device_Camera.rotationMatrix()
            t = camera_model.T_Device_Camera.translation()
        T_device_camera = np.zeros((4, 4))
        T_device_camera[0:3, 0:3] = R
        T_device_camera[0:3, 3] = t
        T_device_camera[3, 3] = 1.0
        T_camera_device = inv(T_device_camera)
        camera_type = "aria"

    t_world = np.array(
        [
            device_row[f"tx_world_{device_row_key}"],
            device_row[f"ty_world_{device_row_key}"],
            device_row[f"tz_world_{device_row_key}"],
        ]
    )
    R_world = qvec2rotmat(
        [
            device_row[f"qw_world_{device_row_key}"],
            device_row[f"qx_world_{device_row_key}"],
            device_row[f"qy_world_{device_row_key}"],
            device_row[f"qz_world_{device_row_key}"],
        ]
    )
    T_world_device = np.zeros((4, 4))
    T_world_device[0:3, 0:3] = R_world
    T_world_device[0:3, 3] = t_world
    T_world_device[3, 3] = 1.0
    T_device_world = inv(T_world_device)

    return {
        "name": name,
        "center": t_world.tolist(),
        "T_device_world": T_device_world.tolist(),
        "T_world_device": T_world_device.tolist(),
        "T_device_camera": T_device_camera.tolist(),
        "T_camera_device": T_camera_device.tolist(),
        "camera_type": camera_type,
        "device_row": device_row,
    }


def xdevice_to_ximage(pt_device: Vec3, cam: Camera):
    if cam.camera_type == "aria":
        assert cam.camera_model is not None
        ret = cam.camera_model.project_no_checks(pt_device / pt_device[2])
    elif cam.camera_type == "colmap":
        assert cam.camera_model is not None
        ret = cam.camera_model.world_to_image(pt_device[0:2] / pt_device[2])
    else:
        raise AssertionError(f"Unexpected camera type: {cam.camera_type}")
    return ret


def ximage_to_xdevice(pt_img: Vec2, cam: Camera):
    if cam.camera_type == "aria":
        assert cam.camera_model is not None
        ret = cam.camera_model.unproject_no_checks(pt_img)[:2]
    elif cam.camera_type == "colmap":
        assert cam.camera_model is not None
        ret = cam.camera_model.image_to_world(pt_img)
    else:
        raise AssertionError(f"Unexpected camera type: {cam.camera_type}")
    return ret


def xworld_to_yimage(pt3d: Vec3, to_cam: Camera):
    assert pt3d.shape[0] == 3
    T_to_world = np.matmul(to_cam.T_camera_device, to_cam.T_device_world)
    pt_target = np.matmul(T_to_world, np.array(pt3d.tolist() + [1.0]))[0:3]
    return xdevice_to_ximage(pt_target, to_cam)


def batch_xworld_to_yimage(pts3d: Vec3, to_cam: Camera):
    assert pts3d.shape[1] == 3
    pts2d = []

    # TODO: optimize
    for pt3d in pts3d:
        T_to_world = to_cam.extrinsics
        pt_target = np.matmul(T_to_world, np.array(pt3d.tolist() + [1.0]))[0:3]
        pts2d.append(xdevice_to_ximage(pt_target, to_cam).reshape(1, -1))
    pts2d = np.concatenate(pts2d, axis=0)
    return pts2d


def batch_xworld_to_yimage_check_camera_z(pts3d: Vec3, to_cam: Camera):
    assert pts3d.shape[1] == 3
    pts2d = []

    # TODO: optimize
    for pt3d in pts3d:
        T_to_world = to_cam.extrinsics
        pt_target = np.matmul(T_to_world, np.array(pt3d.tolist() + [1.0]))[0:3]
        # For negative z-coor point,
        # assign (-1,-1) as image coordinate (which will be filtered out later)
        if pt_target[-1] < 0:
            pts2d.append(np.array([[-1, -1]]))
        else:
            pts2d.append(xdevice_to_ximage(pt_target, to_cam).reshape(1, -1))
    pts2d = np.concatenate(pts2d, axis=0)
    return pts2d


def xdevice_to_yimage(pt3d: Vec3, from_cam: Camera, to_cam: Camera):
    assert pt3d.shape[0] == 3
    print(to_cam.T_device_camera)
    T_world_camera = np.matmul(to_cam.T_world_device, to_cam.T_device_camera)
    T_camera_world = inv(T_world_camera)

    # device -> world -> camera
    T_view_target = np.matmul(T_camera_world, from_cam.T_world_device)
    pt_target = np.matmul(T_view_target, np.array(pt3d.tolist() + [1.0]))[0:3]
    return xdevice_to_ximage(pt_target, to_cam)


def ximage_to_yimage(pt_img: Vec2, from_cam: Camera, to_cam: Camera, z: float = 1.0):
    pt_dev = ximage_to_xdevice(pt_img, from_cam)
    pt_dev = np.array(pt_dev.tolist() + [z])
    return xdevice_to_yimage(pt_dev, from_cam, to_cam)


def get_aria_camera_models(aria_path):
    try:
        from projectaria_tools.core import data_provider

        vrs_data_provider = data_provider.create_vrs_data_provider(aria_path)
        aria_camera_model = vrs_data_provider.get_device_calibration()
        slam_left = aria_camera_model.get_camera_calib("camera-slam-left")
        slam_right = aria_camera_model.get_camera_calib("camera-slam-right")
        rgb_cam = aria_camera_model.get_camera_calib("camera-rgb")
    except Exception as e:
        print(
            f"[Warning] Hitting exception {e}. Fall back to old projectaria_tools ..."
        )
        import projectaria_tools

        vrs_data_provider = projectaria_tools.dataprovider.AriaVrsDataProvider()
        vrs_data_provider.openFile(aria_path)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(214, 1)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(1201, 1)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(1201, 2)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        assert vrs_data_provider.loadDeviceModel()

        aria_camera_model = vrs_data_provider.getDeviceModel()
        slam_left = aria_camera_model.getCameraCalib("camera-slam-left")
        slam_right = aria_camera_model.getCameraCalib("camera-slam-right")
        rgb_cam = aria_camera_model.getCameraCalib("camera-rgb")

    assert slam_left is not None
    assert slam_right is not None
    assert rgb_cam is not None

    return {
        "1201-1": slam_left,
        "1201-2": slam_right,
        "214-1": rgb_cam,
    }
