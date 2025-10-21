import mujoco
import numpy as np
""""
Controller for inverse kinematics and nullspace control in MuJoCo with Franka Emika Panda Arm. From: https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
"""

class InverseKinematicsController:
    def __init__(
        self,
        model,
        data,
        site_name: str,
        joint_names: list,
        mocap_body_name: str,
        q0: np.ndarray,
        integration_dt: float = 0.1,
        damping: float = 1e-4,
        Kpos: float = 0.95,
        Kori: float = 0.95,
        dt: float = 0.002,
        Kn: np.ndarray = None,
        max_angvel: float = 0.1,
    ):
        """
        Initializes the inverse kinematics controller.

        :param model: Mujoco MjModel
        :param data: Mujoco MjData
        :param site_name: Name of the end effector site to control.
        :param joint_names: List of joint names to control.
        :param mocap_body_name: Name of the mocap body used as the target.
        :param q0: Initial joint configuration (1D-array).
        :param integration_dt: Integration timestep.
        :param damping: Damping for the pseudoinverse Jacobian.
        :param Kpos: Gain for the position correction.
        :param Kori: Gain for the orientation correction.
        :param dt: Simulation timestep.
        :param Kn: Nullspace gain (as an array). Default: [10,10,10,10,5,5,5]
        :param max_angvel: Maximum allowed joint velocity.
        """
        self.model = model
        self.data = data
        self.integration_dt = integration_dt
        self.damping = damping
        self.Kpos = Kpos
        self.Kori = Kori
        self.dt = dt
        self.max_angvel = max_angvel
        self.q0 = q0.copy()

        if Kn is None:
            self.Kn = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        else:
            self.Kn = Kn

        # Get the ID of the end effector site
        self.site_id = model.site(site_name).id

        # Create lists for joint IDs, their corresponding positions in qpos, and actuator IDs
        self.joint_names = joint_names
        self.dof_ids = np.array([model.joint(name).id for name in joint_names])
        self.dof_adr = np.array([model.joint(name).dofadr for name in joint_names]).flatten()
        self.actuator_ids = np.array([model.actuator(name).id for name in joint_names])

        # Get the mocap ID of the target body
        self.mocap_id = model.body(mocap_body_name).mocapid[0]

        # Pre-allocated arrays
        self.twist = np.zeros(6)
        self.diag = damping * np.eye(6)
        self.eye = np.eye(len(joint_names))
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

        # Joint limits (for clipping the qpos values)
        self.jnt_min, self.jnt_max = model.jnt_range[self.dof_ids].T

    def get_ctrl(self) -> np.ndarray:
        """
        Computes the control command based on the current state.
        The method calculates the required joint velocity vector,
        integrates it over integration_dt, and then sets the new joint positions
        as commands for the corresponding actuators.

        :return: New joint configuration (q_new) for the controlled DoFs.
        """
        # Compute the position error between the target (Mocap) and the current site position
        dx = self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos
        self.twist[:3] = self.Kpos * dx / self.integration_dt

        # Compute the orientation error
        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.Kori / self.integration_dt

        # Compute the site Jacobian with respect to the controlled joints
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
        jac_pos = self.jacp[:, self.dof_adr]
        jac_rot = self.jacr[:, self.dof_adr]
        jac_robot = np.vstack([jac_pos, jac_rot])

        # Compute the joint velocities using damped least squares
        dq = jac_robot.T @ np.linalg.solve(jac_robot @ jac_robot.T + self.diag, self.twist)
        # Nullspace control: bias the joint velocities towards the home configuration
        dq += (self.eye - np.linalg.pinv(jac_robot) @ jac_robot) @ (self.Kn * (self.q0 - self.data.qpos[self.dof_adr]))

        # Limit the maximum joint velocity
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > self.max_angvel:
            dq *= self.max_angvel / dq_abs_max

        # Integrate the joint velocities to get new joint positions
        q_new = self.data.qpos[self.dof_adr].copy()
        q_new += dq * self.integration_dt

        # Clip the joint positions within the joint limits
        q_new = np.clip(q_new, self.jnt_min, self.jnt_max)

        # Set the control command for the actuators:
        # Create a signal that acts on the corresponding qpos indices.
        ctrl_signal = np.zeros(self.model.nv)
        ctrl_signal[self.dof_adr] = q_new
        self.data.ctrl[self.actuator_ids] = ctrl_signal[self.dof_adr]

        return q_new


