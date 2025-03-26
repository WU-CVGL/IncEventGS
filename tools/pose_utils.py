import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import os
import tqdm


##################################
# Pose Conventions
##################################

def pose_batch_make_homogeneous(pose_batch_N_3_4):
    """
    Input: (num_poses, 3, 4)
    Output: (num_poses, 4, 4) with lower row [0, 0, 0, 1] appended for each pose
    """

    pose_batch_N_4_4 = np.asarray([np.vstack((p, np.asarray([0, 0, 0, 1]))) for p in pose_batch_N_3_4])
    return pose_batch_N_4_4


def get_hom_trafos(rots_3_3, trans_3_1):
    N = rots_3_3.shape[0]
    assert rots_3_3.shape == (N, 3, 3)

    if trans_3_1.shape == (N, 3):
        trans_3_1 = np.expand_dims(trans_3_1, axis=-1)
    else:
        assert trans_3_1.shape == (N, 3, 1)
    
    pose_N_4_4 = np.zeros((N, 4, 4))
    hom = np.array([0,0,0,1]).reshape((1, 4)).repeat(N, axis=0).reshape((N, 1, 4))

    pose_N_4_4[:N, :3, :3] = rots_3_3  # (N, 3, 3)
    pose_N_4_4[:N, :3, 3:4] = trans_3_1 # (N, 3, 1)
    pose_N_4_4[:N, 3:4, :] = hom # (N, 1, 4)

    # pose_N_3_4 = np.asarray([np.concatenate((r, t), axis=1) for r, t in zip(rots_3_3, trans_3_1)])
    # pose_N_4_4 = np.asarray([np.vstack((p, np.asarray([0, 0, 0, 1]))) for p in pose_N_3_4])
    return pose_N_4_4

def quatList_to_poses_hom_and_tss(quat_list_us):
    """
    quat_list: [[t, px, py, pz, qx, qy, qz, qw], ...]
    """
    tss_all_poses_us = [t[0] for t in quat_list_us]

    all_rots = [R.from_quat(rot[4:]).as_matrix() for rot in quat_list_us]
    all_trans = [trans[1:4] for trans in quat_list_us]
    all_trafos = get_hom_trafos(np.asarray(all_rots), np.asarray(all_trans))

    return tss_all_poses_us, all_trafos

def quat_dict_to_pose_hom(T_quat_dict):
    R_ = R.from_quat([T_quat_dict["qx"], T_quat_dict["qy"], T_quat_dict["qz"], T_quat_dict["qw"]]).as_matrix()
    trans_ = np.asarray([T_quat_dict["px"], T_quat_dict["py"], T_quat_dict["pz"]])
    T_hom = get_hom_trafos(np.expand_dims(R_, 0), np.expand_dims(trans_, 0))
    return T_hom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def poses_hom_to_quatlist(poses_hom, tss):
    """
    poses_hom: np.array (N, 4, 4)
    """    
    N = poses_hom.shape[0]
    assert poses_hom.shape == (N, 4, 4)
    assert len(tss) == N

    quatlist = []
    for i, p in enumerate(poses_hom):
        px, py, pz = p[:3, 3]
        qx, qy, qz, qw = R.from_matrix(p[:3, :3]).as_quat()
        quatlist.append([tss[i], px, py, pz, qx, qy, qz, qw])

    return quatlist



##################################
# Reading
##################################
def read_poses_bounds(path_poses_bounds, start_frame=None, end_frame=None, skip_frames=None, invert=False):
    """ Returns: 
    #    :poses np.array (num_poses, 3, 5) in  c2w (even for esim, these poses are already inverted to c2w)
         :bds (num_poses, 2) where [:, 0] = min_depth, and [:, 1] = max_depth
    """
    assert os.path.exists(path_poses_bounds)

    poses_arr = np.load(path_poses_bounds)
    assert poses_arr.shape[0] > 10
    assert poses_arr.shape[1] == 17
    # (num_poses, 17), where  17 = (rot | trans | hwf).ravel(), zmin, zmax
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (num_poses, 17) to (num_poses, 3, 5)
    bds = poses_arr[:, -2:]  # (num_poses, 2)
    num_poses = poses.shape[0]

    if invert:
        for i in range(num_poses):
            rot, trans = poses[i, :3, :3], poses[i, :, 3]
            rot, trans = invert_trafo(rot, trans)
            poses[i, :3, :3] = rot
            poses[i, :3, 3] = trans
        print("** Inverted Poses from ** ", path_poses_bounds)

    # check rotation matrix
    for i in range(num_poses):
        rot = poses[i, :3, :3]
        check_rot(rot, right_handed=True)

    if (start_frame is not None) and (end_frame is not None) and (skip_frames is not None):
        assert end_frame > start_frame
        assert start_frame >= 0
        assert skip_frames > 0
        assert skip_frames < 50

        if end_frame == -1:
            end_frame = poses.shape[0] - 1

        poses = poses[start_frame:end_frame:skip_frames, ...]  # (num_poses, 3, 5)
        bds = bds[start_frame:end_frame:skip_frames, :]

    print("Got total of %d poses" % (poses.shape[0]))
    return poses, bds



##################################
# Interpolating
##################################
def interpol_poses_slerp(tss_poses_ns, poses_rots, poses_trans, tss_query_ns):
    """
    Input
    :tss_poses_ns list of known tss
    :poses_rots list of 3x3 np.arrays
    :poses_trans list of 3x1 np.arrays
    :tss_query_ns list of query tss

    Returns:
    :rots list of rots at tss_query_ns
    :trans list of translations at tss_query_ns
    """
    # Setup Rot interpolator
    rot_interpolator = Slerp(tss_poses_ns, R.from_matrix(poses_rots))
    # Query rot interpolator
    rots = rot_interpolator(tss_query_ns).as_matrix()

    # Setup trans interpolator
    trans_interpolator = interp1d(x=tss_poses_ns, y=poses_trans, axis=0, kind="cubic", bounds_error=True)
    # Query trans interpolator
    trans = trans_interpolator(tss_query_ns)

    return rots, trans

def interpol_pose_nn(tss_all_ns, poses_all, ts_query_ns, tol_dt_ms=8):
    """
    Description: 
    nearest-neighbor-pose association

    Input:
    :tss_all_ns: timestamps where poses are defined
    :poses_all list of (stamp_ns, pos.x, pos.y, pos.z, or.x, or.y, or.z, or.w)
    :ts_query_ns ts where we want to interpolate from

    Output:
    Rot: (3,3) cam2w if invert_pose = False
    trans: (3, 1) cam2w if invert_pose = False
    """

    closest_pose_idx = np.abs(ts_query_ns - tss_all_ns).argmin()
    dT_ms = (tss_all_ns[closest_pose_idx] - ts_query_ns) * 1e-6
    assert np.abs(dT_ms) < tol_dt_ms

    closest_pose = poses_all[closest_pose_idx]
    rot = R.from_quat([closest_pose[4:]]).as_matrix().squeeze()
    trans = np.array((closest_pose[1:4]))

    return rot, trans


##################################
# Check rotation matrix
##################################
def check_rot_batch(poses, right_handed=True):
    """
    Input: Either (num_poses, 3, 5)-array or list of poses (3, 5) or (3,4)
    """
    assert len(poses) > 0
    assert np.all([p.shape[0] == 3 for p in poses])
    assert np.all([p.shape[1] >= 4 for p in poses])

    for i in range(len(poses)):
        rot = poses[i][:3, :3]
        check_rot(rot, right_handed=right_handed)


def check_rot(rot, right_handed=True, eps=1e-6):
    """
    Input: 3x3 rotation matrix
    """
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3

    assert np.allclose(rot.transpose() @ rot, np.eye(3), atol=1e-6)
    assert np.linalg.det(rot) - 1 < eps * 2

    if right_handed:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) - 1.0) < 1e-3
    else:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) + 1.0) < 1e-3


def check_rots_close(rot, rot2, prec_angle=1e-2, prec_elements=1e-3):

    rot_vec = R.from_matrix(rot).as_rotvec(degrees=True)
    rot_vec2 = R.from_matrix(rot2).as_rotvec(degrees=True)

    diff_angle_degree = np.linalg.norm(rot_vec) - np.linalg.norm(rot_vec2)
    # print("diff angle degree = ", diff_angle_degree)
    assert np.abs(diff_angle_degree) < prec_angle
    assert np.all(np.abs(rot - rot2) < prec_elements)


##################################
# Transform Transforms
##################################
def invert_trafo(rot, trans):
    # invert transform from w2cam (esim, colmap) to cam2w
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3
    assert trans.shape[0] == 3

    rot_ = rot.transpose()
    trans_ = -1.0 * np.matmul(rot_, trans)

    check_rot(rot_)
    return rot_, trans_


##################################
# Coordinate System Conventions
##################################
def rub_from_rdf(poses):
    """
    Input
        :poses (num_poses, 3, 4) as (right, down, front), i.e. the normal convention
    Output: 
        :poses (num_poses, 3, 4) reordered as (right, up, back)
    """
    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] >= 4

    poses_ = np.zeros_like(poses)
    poses_ = np.concatenate([poses[:, :, 0:1], -poses[:, :, 1:2:], -poses[:, :, 2:3], poses[:, :, 3:]], 2)

    check_rot_batch(poses_)
    return poses_


def rub_from_luf(poses):
    """
    Input
        :poses (num_poses, 3, 4) as (left, up, front)
    Output: 
        :poses (num_poses, 3, 4) reordered as (right, up, back)
    """
    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] >= 4

    poses_ = np.zeros_like(poses)
    poses_ = np.concatenate([-poses[:, :, 0:1], poses[:, :, 1:2:], -poses[:, :, 2:3], poses[:, :, 3:]], 2)

    check_rot_batch(poses_)
    return poses_

def rdf_from_drb(poses):
    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] >= 4

    poses_ = np.zeros_like(poses)
    poses_ = np.concatenate([poses[:, :, 1:2], poses[:, :, 0:1], -poses[:, :, 2:3], poses[:, :, 3:]], 2)

    check_rot_batch(poses_)
    return poses_

def rub_from_drb(poses):
    """
    This is the conversion used in original code (COLMAP ouputs (right, down, front) 
    --> save_poses_nerf.py creates poses_bounds.py with (down, right, back)
    Original code had this:  poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    Input
        :poses (num_poses, 3, 4) as (down, right, back)
    Output: 
        :poses (num_poses, 3, 4) reordered as (right, up, back)
    """
    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] >= 4

    poses_ = np.zeros_like(poses)
    poses_ = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)

    check_rot_batch(poses_)
    return poses_


def rub_from_luf(poses):
    """
    Input
        :poses (num_poses, 3, 4) as (left, up, front)
    Output: 
        :poses (num_poses, 3, 4) reordered as (right, up, back)
    """
    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] >= 4

    poses_ = np.zeros_like(poses)
    poses_ = np.concatenate([-poses[:, :, 0:1], poses[:, :, 1:2], -poses[:, :, 2:3], poses[:, :, 3:]], 2)

    check_rot_batch(poses_)
    return poses_


def rub_from_drf(poses):
    """
    Input
        :poses (num_poses, 3, 4) as (down, right, front)
    Output: 
        :poses (num_poses, 3, 4) reordered as (right, up, back)
    """
    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] >= 4

    poses_ = np.zeros_like(poses)
    poses_ = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], -poses[:, :, 2:3], poses[:, :, 3:]], 2)

    check_rot_batch(poses_)
    return poses_

def rub_from_ufl(poses):
    """
    Input
        :poses (num_poses, 3, 4) as (up, front, left)
    Output: 
        :poses (num_poses, 3, 4) reordered as (right, up, back)
    """
    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] >= 4

    poses_ = np.zeros_like(poses)
    poses_ = np.concatenate([-poses[:, :, 2:3], poses[:, :, 0:1], -poses[:, :, 1:2], poses[:, :, 3:]], 2)

    check_rot_batch(poses_)
    return poses_

##################################
# Recenter and Averaging
##################################
def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    """ "
    Input:
    z: (3,) first direction
    up: (3,) second direction
    pos: (3,) translational component

    Returns:
    m: (3, 4)
    """

    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    """ "
    Decription: 
    This function puts all poses in a new coordinate system, which is located at the 
    center point of all poses (center / poses[:, :3, 3].mean(0)). 
    The new coordinate´s x-axis points along the crossproduct of r2-sum (up-vector) and average-viewing-direction (vec2).
    The new coordinate´s y-axis points along the crossproduct of average-viewing-direction (vec2) and new-x.
    The new coordinate´s z-axis of average-viewing-direction (vec2).

    Input:
    poses: (num_frames, 3, 5) = [(r1, r2, r3, t, hwf)]

    Intermediate vars:
    center: Average of translational components t
    vec2: sum of r3 rotation vectors (z-axis), normalized vector
    up: sum of r2 rotation vectors (y-axis)

    Returns:
    c2w:
    """
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0) # translational center
    vec2 = normalize(poses[:, :3, 2].sum(0)) # this is in a way the "average viewing direction" of the cameras
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def poses_avg2(poses):
    """ 
    Decription: 
    This function puts all poses in a new coordinate system, which is located at the 
    center point of all poses (center / poses[:, :3, 3].mean(0)). 
    The new coordinate´s x-axis points along the crossproduct of r2-sum (up-vector) and average-viewing-direction (vec2).
    The new coordinate´s y-axis points along the crossproduct of average-viewing-direction (vec2) and new-x.
    The new coordinate´s z-axis of average-viewing-direction (vec2).

    Input:
    poses: (num_frames, 3, 4) = [(r1, r2, r3, t)]

    Intermediate vars:
    center: Average of translational components t
    vec2: sum of r3 rotation vectors (z-axis), normalized vector
    up: sum of r2 rotation vectors (y-axis)

    Returns:
    c2w:
    """
    N = poses.shape[0]
    assert N > 1
    assert poses.shape == (N, 3, 4)

    center = poses[:, :3, 3].mean(0) # translational center
    vec2 = normalize(poses[:, :3, 2].sum(0)) # this is in a way the "average viewing direction" of the cameras
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)

    return c2w


def recenter_poses2(poses):
    """ "
    Decription: 
    The function computes the average trafo of all translational component
    and recenters all poses around this mean, with a new coordinate system (rotated).
    It preserves coordinate convention (e.g. rdf-input => rdf-output) 

    Input:
    poses: (num_frames, 3, 4) = (R, t)

    Intermediate:
    c2w: (4,4) this is the new center (identity) pose
    For rotations, it computes the mean z-axis (Z_mean) and mean y-axis (Y_mean).
    It then sets z-axis = Z_mean, x-axis = Y_mean x z-axis
    and y-axis = Z_mean X x-axis. (see https://github.com/bmild/nerf/issues/34)

    Output: 
    poses: (num_frames, 3, 4) = (R', t', hwf)
    """
    N = poses.shape[0]
    assert N > 1
    assert poses.shape == (N, 3, 4)
    poses_ = np.copy(poses)
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])

    # Recenter
    c2w = poses_avg2(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses

    # use poses_ variable to keep the hwf (fith column)
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_

    assert poses.shape == poses_.shape
    assert poses.shape == (N, 3, 4)
    return poses  # (num_poses, 3, 4)

# See discussion: https://github.com/bmild/nerf/issues/34
def recenter_poses(poses):
    """ "
    Input:
    poses: (num_frames, 3, 5) = (R, t, hwf)

    Intermediate:
    c2w: (4,4) this is the new center (identity) pose

    The function computes the average trafo of all translational component
    and recenters all poses around this mean.
    For rotations, it computes the mean z-axis (Z_mean) and mean y-axis (Y_mean).
    It then sets z-axis = Z_mean, x-axis = Y_mean x z-axis
    and y-axis = Z_mean X x-axis. (see https://github.com/bmild/nerf/issues/34)

    Output: 
    poses: (num_frames, 3, 5) = (R', t', hwf)
    """
    poses_ = np.copy(poses)
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    num_frames = poses_.shape[0]

    # Recenter
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses

    # use poses_ variable to keep the hwf (fith column)
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_

    assert poses.shape == poses_.shape
    assert poses.shape == (num_frames, 3, 5)
    return poses  # (num_poses, 3, 5)

def recenter_poses_jointly(poses_train, poses_hf):
    """ "
    Input:
    poses: (num_frames, 3, 5) = (R, t, zeros)
    poses_hf: list of [{"ts_ns": scalar, "pose_c2w": (3, 4)}]

    Intermediate:
    c2w: (4,4) average trafo of all poses (rotational and translational avg)
         this is the new center (identity) pose

    Output: 
    :poses_ (num_frames, 3, 5)
    :poses_hf_ list of [{"ts_ns": scalar, "pose_c2w": (3, 4)}]
    """

    # prepare train and val poses
    poses_train_ = np.copy(poses_train)
    num_poses_train = poses_train_.shape[0]

    assert poses_train.shape == (num_poses_train, 3, 5)
    # assert num_poses_train * 4 > num_poses_val
    assert len(poses_hf) > num_poses_train

    # prepare hf poses
    tss_tmp = [ts["ts_ns"] for ts in poses_hf]
    hf_shapes = [p["pose_c2w"].shape for p in poses_hf]
    assert np.all([sh == (3, 4) for sh in hf_shapes])
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])  # (1, 4)

    # merge train hf-poses
    poses_ = np.append(poses_train_, np.asarray([np.hstack((p["pose_c2w"], np.zeros((3, 1)))) for p in poses_hf]), 0)

    # Recenter jointly
    c2w = poses_avg(poses_)  # (3, 4)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # (4, 4)
    bottoms = np.tile(
        bottom[np.newaxis, ...], [poses_.shape[0], 1, 1]
    )  # tile((1,1,4), (N_poses, 1, 1)) --> (N_poses, 1, 4)
    poses_ = np.concatenate([poses_[:, :3, :4], bottoms], -2)  # (N_poses, 4, 4)
    poses_ = np.linalg.inv(c2w) @ poses_  # (4, 4) @ (N_poses, 4, 4) -> broadcast to (N_poses, 4, 4)

    # Get new event poses
    for i in range(len(poses_hf)):
        poses_hf[i]["pose_c2w"] = poses_[num_poses_train + i, :3, :4]

    # Get train poses
    poses_train = poses_[0:num_poses_train, :3, :]  # (num_frames, 3, 4)
    hwf = poses_train_[:, :, 4]  # get fifth column
    hwf = hwf[..., np.newaxis]  # (num_frames, 3, 1)
    poses_train = np.concatenate((poses_train, hwf), axis=2)  #  (num_frames, 3, 5)


    assert poses_train.shape == (num_poses_train, 3, 5)
    assert np.all(poses_train[:, :, :4] == poses_[0:num_poses_train, :3, :])
    assert np.all(np.equal(tss_tmp, [ts["ts_ns"] for ts in poses_hf]))
    assert np.all(np.equal(hf_shapes, (3, 4)))
    assert np.all(np.equal(hf_shapes, [p["pose_c2w"].shape for p in poses_hf]))

    return poses_train, poses_hf


##################################
# Compute poses
##################################
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads)

        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def create_poses_bounds_esim(quatlist_ns, tss_imgs_ns, bds, num_imgs, H, W, focal_final=350, prec_angle=0.8, prec_trans=0.04):    
    # save poses in nsff/llff format
    
    assert len(tss_imgs_ns) == num_imgs
    assert len(bds) == num_imgs

    tss_all_poses_ns, poses_hom = quatList_to_poses_hom_and_tss(quatlist_ns)
    all_rots = [r[:3, :3] for r in poses_hom]
    all_trans = [t[:3, 3] for t in poses_hom]

    print("\nCreating poses_bounds ESIM (rot, trans, hwf)")
    poses_bounds = []
    pbar = tqdm.tqdm(total=num_imgs)
    skipped = 0
    for i in range(num_imgs):
        if tss_all_poses_ns[0] - tss_imgs_ns[i] > 0:
            print(f"Moving ts {i}.jpg by {(tss_all_poses_ns[0] - tss_imgs_ns[i])*1e3} ms to FIRST pose-ts (still creating pose_bounds-entry)")
            skipped += 1
            tss_imgs_ns[i] = tss_all_poses_ns[0]
        if tss_all_poses_ns[-1] - tss_imgs_ns[i] < 0:
            print(f"Moving ts {i}.jpg by {(tss_all_poses_ns[0] - tss_imgs_ns[i])*1e3} ms to LAST pose-ts (still creating pose_bounds-entry)")
            skipped += 1
            tss_imgs_ns[i] = tss_all_poses_ns[-1]
            
        rot_slerp, trans_slerp = interpol_poses_slerp(
            tss_all_poses_ns, all_rots, all_trans, tss_imgs_ns[i]
        )

        hwf = np.array((H, W, focal_final))
        rthwf = np.concatenate((rot_slerp, trans_slerp[..., np.newaxis], hwf[..., np.newaxis]), axis=1)
        # (r1, r2, r3, t, hwf).ravel = (3, 5).ravel = 15 + min_depth + max_depth
        poses_bounds.append(np.concatenate([rthwf.ravel(), np.array([bds[i][0], bds[i][1]])], 0))
        pbar.update(1)
    
    print(f"Interpolated {len(poses_bounds)} poses from total of {len(tss_all_poses_ns)}")
    assert skipped <= 2

    return poses_bounds

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    """
    Description: Expects c2w, rub poses => Outputs rdf (open-cv format).
    """
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    return new_pose