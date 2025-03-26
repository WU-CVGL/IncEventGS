import torch
import numpy as np


delt = 0
EPSILON = 1e-6

def se3_2_qt(wu):
    w, u = wu.split([3, 3], dim=-1)  # w:前3个代表旋转，后3个代表平移
    wx = skew_symmetric(w)  # wx=[0 -w(2) w(1);w(2) 0 -w(0);-w(1) w(0) 0]
    theta = w.norm(dim=-1)[..., None, None]  # theta=sqrt(w'*w)
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    # A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    # R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    t = V @ u[..., None]
    q = exp_r2q(w)
    return q, t


def se3_2_qt_parallel(wu):
    w, u = wu.split([3, 3], dim=-1)  # w:前3个代表旋转，后3个代表平移
    wx = skew_symmetric(w)  # wx=[0 -w(2) w(1);w(2) 0 -w(0);-w(1) w(0) 0]
    theta = w.norm(dim=-1)[..., None, None]  # theta=sqrt(w'*w)
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    # A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    # R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    t = V @ u[..., None]
    q = exp_r2q_parallel(w)
    return q, t.squeeze(-1)


def skew_symmetric(w):
    w0,w1,w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                      torch.stack([w2,O,-w0],dim=-1),
                      torch.stack([-w1,w0,O],dim=-1)],dim=-2)
    return wx


def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        if i>0:
            denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/ (denom + EPSILON)
    return ans


def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/ (denom + EPSILON)
    return ans


def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/ (denom + EPSILON)
    return ans


def exp_r2q(r):
    x, y, z = r[0], r[1], r[2]
    theta = 0.5 * torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    if theta < 1e-20:
        qx = (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * x
        qy = (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * y
        qz = (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * z
        qw = 1. - 1. / 2. * theta ** 2 + 1. / 24. * theta ** 4
        q_ = torch.stack([qx+delt, qy+delt, qz+delt, qw-delt], 0)
    else:
        lambda_ = torch.sin(theta) / (2. * theta + EPSILON)
        q_ = torch.stack([lambda_ * x, lambda_ * y, lambda_ * z, torch.cos(theta)], 0)  # xyzw

    return q_

def exp_r2q_parallel(r):
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    theta = 0.5 * torch.sqrt(x ** 2 + y ** 2 + z ** 2)

    lambda_ = torch.sin(theta) / (2. * theta + EPSILON)

    # Bool_criterion_1 = (theta < 1e-20)
    # Bool_criterion_2 = ~Bool_criterion_1
    # Bool_criterion = torch.stack([Bool_criterion_1, Bool_criterion_2], dim=-1)

    Bool_criterion = (theta < 1e-20)

    """
    qx = torch.stack([(1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * x, lambda_ * x], dim=-1)[Bool_criterion]
    qy = torch.stack([(1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * y, lambda_ * y], dim=-1)[Bool_criterion]
    qz = torch.stack([(1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * z, lambda_ * z], dim=-1)[Bool_criterion]
    qw = torch.stack([1. - 1. / 2. * theta ** 2 + 1. / 24. * theta ** 4, torch.cos(theta)], dim=-1)[Bool_criterion]
    """
    qx = Bool_criterion * (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * x + (~Bool_criterion) * lambda_ * x
    qy = Bool_criterion * (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * y + (~Bool_criterion) * lambda_ * y
    qz = Bool_criterion * (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * z + (~Bool_criterion) * lambda_ * z
    qw = Bool_criterion * (1. - 1. / 2. * theta ** 2 + 1. / 24. * theta ** 4) + (~Bool_criterion) * torch.cos(theta)

    """
    qx = [((1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * x)[i] if Bool_value else (lambda_ * x)[i] for i, Bool_value in enumerate(Bool_criterion)]
    qy = [((1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * y)[i] if Bool_value else (lambda_ * y)[i] for i, Bool_value in enumerate(Bool_criterion)]
    qz = [((1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * z)[i] if Bool_value else (lambda_ * z)[i] for i, Bool_value in enumerate(Bool_criterion)]
    qw = [(1. - 1. / 2. * theta ** 2 + 1. / 24. * theta ** 4)[i] if Bool_value else (torch.cos(theta))[i] for i, Bool_value in enumerate(Bool_criterion)]
    """

    q_ = torch.stack([qx, qy, qz, qw], -1)  # xyzw

    return q_

def q_to_R(q):  # xyzw    四元数转化为旋转矩阵
    qb, qc, qd, qa = q.unbind(dim=-1)
    R = torch.stack(
        [torch.stack([1 - 2 * (qc ** 2 + qd ** 2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)], dim=-1),
         torch.stack([2 * (qb * qc + qa * qd), 1 - 2 * (qb ** 2 + qd ** 2), 2 * (qc * qd - qa * qb)], dim=-1),
         torch.stack([2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb ** 2 + qc ** 2)], dim=-1)],
        dim=-2)
    return R

def q_to_R_parallel(q):  # xyzw    四元数转化为旋转矩阵
    qb, qc, qd, qa = q.unbind(dim=-1)
    R = torch.stack(
        [torch.stack([1 - 2 * (qc ** 2 + qd ** 2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)], dim=-1),
         torch.stack([2 * (qb * qc + qa * qd), 1 - 2 * (qb ** 2 + qd ** 2), 2 * (qc * qd - qa * qb)], dim=-1),
         torch.stack([2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb ** 2 + qc ** 2)], dim=-1)],
        dim=-2)
    return R

def q_to_Q(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    Q_0 = torch.stack([w, -z, y, x], 0).reshape(1, 4)
    Q_1 = torch.stack([z, w, -x, y], 0).reshape(1, 4)
    Q_2 = torch.stack([-y, x, w, z], 0).reshape(1, 4)
    Q_3 = torch.stack([-x, -y, -z, w], 0).reshape(1, 4)
    Q_ = torch.cat([Q_0, Q_1, Q_2, Q_3], 0)

    return Q_

def q_to_Q_parallel(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    Q_0 = torch.stack([w, -z, y, x], -1).unsqueeze(-2)
    Q_1 = torch.stack([z, w, -x, y], -1).unsqueeze(-2)
    Q_2 = torch.stack([-y, x, w, z], -1).unsqueeze(-2)
    Q_3 = torch.stack([-x, -y, -z, w], -1).unsqueeze(-2)
    Q_ = torch.cat([Q_0, Q_1, Q_2, Q_3], -2)

    return Q_

def q_to_q_conj(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    q_conj_ = torch.stack([-x, -y, -z, w], 0)
    return q_conj_

def q_to_q_conj_parallel(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_conj_ = torch.stack([-x, -y, -z, w], -1)
    return q_conj_


def log_q2r(q):  # here fixme
    x, y, z, w = q[0], q[1], q[2], q[3]
    theta = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    if theta < 1e-20:
        lambda_ = 2. / (w + EPSILON) - 2. / 3. * (theta ** 2) / (w * w * w + EPSILON)
    elif torch.abs(w) < 1e-10:
        if w > 0:
            lambda_ = torch.pi / (theta + EPSILON)
        else:
            lambda_ = -torch.pi / (theta + EPSILON)
    else:
        lambda_ = 2. * (torch.arctan(theta / (w + EPSILON))) / (theta + EPSILON)

    r_ = torch.stack([lambda_ * x, lambda_ * y, lambda_ * z], 0)

    return r_

def log_q2r_parallel(q):  # here fixme
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    theta = torch.sqrt(x ** 2 + y ** 2 + z ** 2)

    Bool_criterion_1 = (theta < 1e-20)
    Bool_criterion_2 = (theta >= 1e-20) & (torch.abs(w) < 1e-10)
    Bool_criterion_3 = (theta >= 1e-20) & (torch.abs(w) >= 1e-10)

    lambda_ = Bool_criterion_1 * (2. / (w + EPSILON) - 2. / 3. * (theta ** 2) / (w * w * w + EPSILON)) + Bool_criterion_2 * (w / (torch.abs(w) + EPSILON) * torch.pi / (theta + EPSILON)) + Bool_criterion_3 * (2. * (torch.arctan(theta / (w + EPSILON))) / (theta + EPSILON))
    r_ = torch.stack([lambda_ * x, lambda_ * y, lambda_ * z], -1)

    return r_


def SE3_to_se3(Rt, eps=1e-8):  # [...,3,4]    pose->旋转角速度+平移速度 6维变量
    # Rt [3, 4]
    # wu [6]
    R, t = Rt.split([3, 1], dim=-1)
    w = SO3_to_so3(R)    # rotation matrix to 旋转角速度
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    invV = I - 0.5 * wx + (1 - A / (2 * B + EPSILON)) / (theta ** 2 + eps) * wx @ wx
    u = (invV @ t)[..., 0]
    wu = torch.cat([w, u], dim=-1)
    return wu


def SO3_to_so3(R, eps=1e-7):  # [...,3,3]
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
                ..., None, None] % np.pi  # ln(R) will explode if theta==pi
    lnR = 1 / (2 * taylor_A(theta) + EPSILON) * (R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
    w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
    w = torch.stack([w0, w1, w2], dim=-1)
    return w

def SE3_to_se3_N(poses_rt):
    poses_se3_list =[]
    for i in range(poses_rt.shape[0]):
        pose_se3 = SE3_to_se3(poses_rt[i])
        poses_se3_list.append(pose_se3)
    poses = torch.stack(poses_se3_list, 0)

    return poses

def se3_to_SE3(wu): # [...,3]
    w,u = wu.split([3,3],dim=-1)
    wx = skew_symmetric(w)  # wx=[0 -w(2) w(1);w(2) 0 -w(0);-w(1) w(0) 0]
    theta = w.norm(dim=-1)[...,None,None]  # theta=sqrt(w'*w)
    I = torch.eye(3,device=w.device,dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I+A*wx+B*wx@wx
    V = I+B*wx+C*wx@wx
    Rt = torch.cat([R,(V@u[...,None])],dim=-1)
    return Rt

def se3_to_SE3_N(poses_wu):
    poses_se3_list =[]
    for i in range(poses_wu.shape[0]):
        pose_se3 = se3_to_SE3(poses_wu[i])
        poses_se3_list.append(pose_se3)
    poses = torch.stack(poses_se3_list, 0)

    return poses


def SplineN(start_pose, end_pose, NUM, device=None):
    # start_pose & end_pose are se3

    # spline t
    pose_list = []
    interval = 1 / (NUM - 1)
    for i in range(0, start_pose.shape[0]):
        q_start, t_start = se3_2_qt(start_pose[i])  # N poses
        q_end, t_end = se3_2_qt(end_pose[i])
        # print('test')

        for j in range(0, NUM):
            # five variables
            if j == 0:
                sample_time = j * interval + 0.000001
            elif j == (NUM-1):
                sample_time = j * interval - 0.000001
            else:
                sample_time = j * interval
            # sample_time = j * interval

            # sample t_vector
            t_t = (1 - sample_time) * t_start + sample_time * t_end
            # print(t_t[0], t_t[1], t_t[2])

            # sample rotation_vector
            q_tau_0 = q_to_Q(q_to_q_conj(q_start)) @ q_end  # equation 50 shape:[4]

            r = sample_time * log_q2r(q_tau_0)  # equation 51 shape:[3]
            q_t_0 = exp_r2q(r)  # equation 52 shape:[4]
            q_t = q_to_Q(q_start) @ q_t_0  # equation 53 shape:[4]

            # convert q&t to RT
            R = q_to_R(q_t)  # [3,3]
            t = t_t.reshape(3, 1)
            # R = torch.Tensor(R).to(device)
            # t = torch.Tensor(t).to(device)
            pose_spline = torch.cat([R, t], -1)  # [3, 4]
            pose_list.append(pose_spline)

    poses = torch.stack(pose_list, 0)  # [N, 3, 4]

    return poses


def SplineN_new(start_pose, end_pose, poses_number, NUM, delay_time=0):
    # start_pose & end_pose are se3
    pose_time = poses_number / (NUM-1)

    # parallel

    pos_0 = torch.where(pose_time == 0)
    pose_time[pos_0] = pose_time[pos_0] + 0.000001
    pos_1 = torch.where(pose_time == 1)
    pose_time[pos_1] = pose_time[pos_1] - 0.000001

    # pose_time = pose_time.reshape([-1, 1])   # [6] --> [6,1]

    q_start, t_start = se3_2_qt_parallel(start_pose)    # t_start:[35, 3] (35 imgs * 3 dims)    q_start:[35,4] (35 imgs * 4 dims)
    q_end, t_end = se3_2_qt_parallel(end_pose)
    # sample t_vector
    t_t = (1 - pose_time)[..., None] * t_start + pose_time[..., None] * t_end     # [35, 6, 3] (35 imgs * 6 poses_per_img * 3 dims)
    # print(t_t[0], t_t[1], t_t[2])

    # sample rotation_vector
    q_tau_0 = q_to_Q_parallel(q_to_q_conj_parallel(q_start)) @ q_end[..., None]    # [35, 4, 1]  # equation 50 shape:[4]
    r = pose_time[..., None] * log_q2r_parallel(q_tau_0.squeeze(-1))    # [35, 6, 3]   # [6,1] * [35, 1, 3] = [35, 6, 3]  # equation 51 shape:[3]
    q_t_0 = exp_r2q_parallel(r)  # equation 52 shape:[4]    # [35, 6, 4]
    q_t = q_to_Q_parallel(q_start) @ q_t_0[..., None]    # [35, 6, 4, 1]  # equation 53 shape:[4]

    # convert q&t to RT
    R = q_to_R_parallel(q_t.squeeze(dim=-1))  # [3,3]    # [35, 6, 3, 3]
    t = t_t.unsqueeze(dim=-1)    # [35, 6, 3, 1]
    pose_spline = torch.cat([R, t], -1)  # [3, 4]

    poses = pose_spline.reshape([-1, 3, 4])  # [35, 6, 3, 4]

    return poses


def Spline4N_new(pose0, pose1, pose2, pose3, tau):
    q0, t0 = se3_2_qt_parallel(pose0)
    q1, t1 = se3_2_qt_parallel(pose1)
    q2, t2 = se3_2_qt_parallel(pose2)
    q3, t3 = se3_2_qt_parallel(pose3)

    u = tau
    uu = tau ** 2
    uuu = tau ** 3
    one_over_six = 1. / 6.
    half_one = 0.5

    # t
    coeff0 = one_over_six - half_one * u + half_one * uu - one_over_six * uuu
    coeff1 = 4 * one_over_six - uu + half_one * uuu
    coeff2 = one_over_six + half_one * u + half_one * uu - half_one * uuu
    coeff3 = one_over_six * uuu

    # spline t
    t_t = coeff0 * t0 + coeff1 * t1 + coeff2 * t2 + coeff3 * t3

    # R
    coeff1_r = 5 * one_over_six + half_one * u - half_one * uu + one_over_six * uuu
    coeff2_r = one_over_six + half_one * u + half_one * uu - 2 * one_over_six * uuu
    coeff3_r = one_over_six * uuu

    # spline R
    q_01 = q_to_Q_parallel(q_to_q_conj_parallel(q0)) @ q1[..., None]  # [1]
    q_12 = q_to_Q_parallel(q_to_q_conj_parallel(q1)) @ q2[..., None]  # [2]
    q_23 = q_to_Q_parallel(q_to_q_conj_parallel(q2)) @ q3[..., None]  # [3]

    r_01 = log_q2r_parallel(q_01.squeeze(-1)) * coeff1_r  # [4]
    r_12 = log_q2r_parallel(q_12.squeeze(-1)) * coeff2_r  # [5]
    r_23 = log_q2r_parallel(q_23.squeeze(-1)) * coeff3_r  # [6]

    q_t_0 = exp_r2q_parallel(r_01)  # [7]
    q_t_1 = exp_r2q_parallel(r_12)  # [8]
    q_t_2 = exp_r2q_parallel(r_23)  # [9]

    q_product1 = q_to_Q_parallel(q_t_1) @ q_t_2[..., None]  # [10]
    q_product2 = q_to_Q_parallel(q_t_0) @ q_product1  # [10]
    q_t = q_to_Q_parallel(q0) @ q_product2  # [10]

    R = q_to_R_parallel(q_t.squeeze(-1))  # [3,3]
    t = t_t.unsqueeze(dim=-1)

    pose_spline = torch.cat([R, t], -1)  # [3, 4]
    poses = pose_spline.reshape([-1, 3, 4])  # [35, 6, 3, 4]

    return poses


def SplineRollingShutter(start_pose, end_pose, poses_number, H, delay_time=0):
    # start_pose & end_pose are se3

    # spline t
    pose_list = []
    # pose_time = poses_number.reshape([-1, 1])/(H-1)    # 选择的 pose 对应行数 / 总行数 = 此时对应的 T

    pose_time = poses_number / (H-1+delay_time)

    # parallel

    pos_0 = torch.where(pose_time == 0)
    pose_time[pos_0] = pose_time[pos_0] + 0.000001
    pos_1 = torch.where(pose_time == 1)
    pose_time[pos_1] = pose_time[pos_1] - 0.000001

    # pose_time = pose_time.reshape([-1, 1])   # [6] --> [6,1]

    q_start, t_start = se3_2_qt_parallel(start_pose)    # t_start:[35, 3] (35 imgs * 3 dims)    q_start:[35,4] (35 imgs * 4 dims)
    q_end, t_end = se3_2_qt_parallel(end_pose)
    # sample t_vector
    t_t = (1 - pose_time)[..., None] * t_start + pose_time[..., None] * t_end     # [35, 6, 3] (35 imgs * 6 poses_per_img * 3 dims)
    # print(t_t[0], t_t[1], t_t[2])

    # sample rotation_vector
    q_tau_0 = q_to_Q_parallel(q_to_q_conj_parallel(q_start)) @ q_end[..., None]    # [35, 4, 1]  # equation 50 shape:[4]
    r = pose_time[..., None] * log_q2r_parallel(q_tau_0.squeeze(-1))    # [35, 6, 3]   # [6,1] * [35, 1, 3] = [35, 6, 3]  # equation 51 shape:[3]
    q_t_0 = exp_r2q_parallel(r)  # equation 52 shape:[4]    # [35, 6, 4]
    q_t = q_to_Q_parallel(q_start) @ q_t_0[..., None]    # [35, 6, 4, 1]  # equation 53 shape:[4]

    # convert q&t to RT
    R = q_to_R_parallel(q_t.squeeze(dim=-1))  # [3,3]    # [35, 6, 3, 3]
    t = t_t.unsqueeze(dim=-1)    # [35, 6, 3, 1]
    pose_spline = torch.cat([R, t], -1)  # [3, 4]

    poses = pose_spline.reshape([-1, 3, 4])  # [35, 6, 3, 4]

    return poses


def SplineEvent(se3_start, se3_end, t_tau, period, delay_time=0):
    # start_pose & end_pose are se3

    pose_time = t_tau / (period+delay_time)

    # parallel

    pos_0 = torch.where(pose_time == 0)
    pose_time[pos_0] = pose_time[pos_0] + 1e-6
    pos_1 = torch.where(pose_time == 1)
    pose_time[pos_1] = pose_time[pos_1] - 1e-6

    # pose_time = pose_time.reshape([-1, 1])   # [6] --> [6,1]

    q_start, t_start = se3_2_qt_parallel(se3_start)    # t_start:[35, 3] (35 imgs * 3 dims)    q_start:[35,4] (35 imgs * 4 dims)
    q_end, t_end = se3_2_qt_parallel(se3_end)
    # sample t_vector
    t_t = (1 - pose_time)[..., None] * t_start + pose_time[..., None] * t_end     # [35, 6, 3] (35 imgs * 6 poses_per_img * 3 dims)
    # print(t_t[0], t_t[1], t_t[2])

    # sample rotation_vector
    q_tau_0 = q_to_Q_parallel(q_to_q_conj_parallel(q_start)) @ q_end[..., None]    # [t_tau.shape[0], 4, 1]  # equation 50 shape:[4]
    r = pose_time[..., None] * log_q2r_parallel(q_tau_0.squeeze(-1))    # [t_tau.shape[0], 3]   # [6,1] * [35, 1, 3] = [35, 6, 3]  # equation 51 shape:[3]
    q_t_0 = exp_r2q_parallel(r)  # [t_tau.shape[0], 4]
    q_t = q_to_Q_parallel(q_start) @ q_t_0[..., None]    # [35, 6, 4, 1]  # equation 53 shape:[4]

    # convert q&t to RT
    R = q_to_R_parallel(q_t.squeeze(dim=-1))  # [3,3]    # [35, 6, 3, 3]
    t = t_t.unsqueeze(dim=-1)    # [35, 6, 3, 1]
    pose_spline = torch.cat([R, t], -1)  # [3, 4]

    poses = pose_spline.reshape([-1, 3, 4])  # [35, 6, 3, 4]

    return poses



def get_pose(start_pose, end_pose, NUM, device):
    return SplineN(start_pose, end_pose, NUM, device)


### here: test ###
def test_Spline(start_pose, end_pose, NUM, device=None):
    # start_pose & end_pose are se3

    # spline t
    pose_list = []
    interval = 1 / (NUM - 1)
    for i in range(0, start_pose.shape[0]):
        # start_pose = start_pose[None, :, :]
        # end_pose = end_pose[None, :, :]
        q_start, t_start = se3_2_qt(start_pose[i])  # N poses
        q_end, t_end = se3_2_qt(end_pose[i])

        for j in range(0, NUM):
            # five variables
            if j==NUM//2:
                sample_time = j * interval + 0.1
            else:
                sample_time = j * interval + 0.1

            # sample t_vector
            t_t = (1 - sample_time) * t_start + sample_time * t_end
            # print(t_t[0], t_t[1], t_t[2])

            # sample rotation_vector
            q_tau_0 = q_to_Q(q_to_q_conj(q_start)) @ q_end  # equation 50 shape:[4]
            r = sample_time * log_q2r(q_tau_0)  # equation 51 shape:[3]
            q_t_0 = exp_r2q(r)  # equation 52 shape:[4]
            q_t = q_to_Q(q_start) @ q_t_0  # equation 53 shape:[4]

            # convert q&t to RT
            R = q_to_R(q_t)  # [3,3]
            t = t_t.reshape(3, 1)
            # R = torch.Tensor(R).to(device)
            # t = torch.Tensor(t).to(device)
            pose_spline = torch.cat([R, t], -1)  # [3, 4]
            pose_list.append(pose_spline)

    poses = torch.stack(pose_list, 0)  # [N, 3, 4]

    return poses[1]


def test_autograd(se3_start, se3_end, deblur_num=3):
    spline_pose = test_Spline(se3_start, se3_end, deblur_num)
    loss = torch.sum(spline_pose)

    loss.backward()
    print(se3_start.grad)

    return loss


def test_math_grad(se3_start, se3_end, deblur_num=3):
    spline_pose = test_Spline(se3_start, se3_end, deblur_num)
    return spline_pose


# a = exp_r2q(torch.tensor([0.1, 0.1, 0.1]))
# print(a)