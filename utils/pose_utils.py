import numpy as np
import torch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def camera_w2c(camera):
    T_w2c = torch.eye(4, device=camera.R.device, dtype=camera.R.dtype)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T
    return T_w2c


def update_camera_from_w2c(camera, T_w2c):
    camera.update_RT(T_w2c[0:3, 0:3], T_w2c[0:3, 3])


def get_virtual_camera_delta(camera, sample):
    sample = torch.as_tensor(
        sample, device=camera.cam_rot_delta.device, dtype=camera.cam_rot_delta.dtype
    )
    rot_delta = camera.cam_rot_delta + sample * camera.blur_rot_delta
    trans_delta = camera.cam_trans_delta + sample * camera.blur_trans_delta
    return rot_delta, trans_delta


def reset_blur_motion(camera):
    if getattr(camera, "blur_rot_delta", None) is None:
        return
    camera.blur_rot_delta.data.fill_(0)
    camera.blur_trans_delta.data.fill_(0)


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = camera_w2c(camera).to(device=tau.device, dtype=tau.dtype)

    new_w2c = SE3_exp(tau) @ T_w2c

    converged = tau.norm() < converged_threshold
    update_camera_from_w2c(camera, new_w2c)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
