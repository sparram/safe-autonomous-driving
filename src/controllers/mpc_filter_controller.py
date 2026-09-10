import numpy as np
import osqp
from scipy import sparse
from config import N, DT, GAMMA_CBF
from models.kinematic import KinematicBicycleModel

class MPC_CBF_SafetyFilter:
    def __init__(self, horizon=N):
        self.N = horizon

        # Pesos cuadráticos para el MPC Nominal
        self.w_pos = 0.60 / (2.0 ** 2)
        self.w_spd = 0.20 / (5.0 ** 2)
        self.w_ctrl_steer = 0.20 / (0.5 ** 2)
        self.w_ctrl_acc = 0.20 / (1.0 ** 2)
        self.w_dctrl_steer = 0.20 / (0.1 ** 2)
        self.w_dctrl_acc = 0.20 / (2.0 ** 2)

    def _extract_obstacles(self, env, vehicle_pos, max_dist=18.0):
        obstacles_list = []
        vehicles = env.engine.traffic_manager.vehicles
        for v in vehicles:
            if v != env.agent:
                dist = np.linalg.norm(v.position - vehicle_pos)
                if dist < max_dist:
                    v_m_s = max(v.speed_km_h / 3.6, 0.0)
                    heading = v.heading_theta
                    vx = v_m_s * np.cos(heading)
                    vy = v_m_s * np.sin(heading)
                    obstacles_list.append(np.array([v.position[0], v.position[1], vx, vy]))
        return obstacles_list

    def _get_lane_safe(self, road_net, road_index):
        if road_index[2] < 0:
            return None
        try:
            return road_net.get_lane(road_index)
        except (KeyError, AttributeError, IndexError):
            return None

    def _is_lane_blocked(self, lane_obj, state_real, obstacles_list):
        if lane_obj is None:
            return True
        s_e, _ = lane_obj.local_coordinates(state_real[:2])
        for obs in obstacles_list:
            s_o, lat_o = lane_obj.local_coordinates(obs[:2])
            d_fwd = s_o - s_e
            if 0.0 < d_fwd < 18.0 and abs(lat_o) < 1.5:
                return True
        return False

    def _generate_ref_trajectory(self, target_lane, state_real, should_stop):
        ref_trajectory = []
        s_ego_target, _ = target_lane.local_coordinates(state_real[:2])
        is_curved = abs(target_lane.heading_theta_at(s_ego_target + 5) - target_lane.heading_theta_at(s_ego_target)) > 0.1
        base_speed = 5.0 if is_curved else 8.0
        target_speed = 0.0 if should_stop else base_speed

        for i in range(self.N):
            target_s = s_ego_target + target_speed * (i + 1) * DT
            ref_x, ref_y = target_lane.position(target_s, 0.0)
            ref_trajectory.append((ref_x, ref_y, target_speed))

        return ref_trajectory

    def _predict_trajectory(self, current_state, u_seq):
        x_bar = np.zeros((self.N + 1, 4))
        x_bar[0] = current_state
        A_seq, B_seq = [], []

        for i in range(self.N):
            A = KinematicBicycleModel.jacobian_x(x_bar[i], u_seq[i])
            B = KinematicBicycleModel.jacobian_u(x_bar[i], u_seq[i])
            A_seq.append(A)
            B_seq.append(B)
            x_bar[i + 1] = KinematicBicycleModel.step(x_bar[i], u_seq[i])

        return x_bar, A_seq, B_seq

    # ETAPA 1: Resolver MPC Nominal sin restricciones CBF
    def solve_nominal_mpc(self, u0_warm, current_state, reference_trajectory):
        u_warm = u0_warm.reshape((self.N, 2))
        u_warm_flat = u_warm.flatten()
        x_bar, A_seq, B_seq = self._predict_trajectory(current_state, u_warm)

        S_u = np.zeros((4 * self.N, 2 * self.N))
        for k in range(self.N):
            for j in range(k + 1):
                if k == j:
                    S_u[4*k:4*(k+1), 2*j:2*(j+1)] = B_seq[j]
                else:
                    A_prod = np.eye(4)
                    for m in range(j + 1, k + 1):
                        A_prod = A_seq[m] @ A_prod
                    S_u[4*k:4*(k+1), 2*j:2*(j+1)] = A_prod @ B_seq[j]

        Q_block = np.diag([self.w_pos, self.w_pos, self.w_spd, 0.0])
        Q = sparse.kron(sparse.eye(self.N), Q_block)

        R_block = np.diag([self.w_ctrl_steer, self.w_ctrl_acc])
        R = sparse.kron(sparse.eye(self.N), R_block)

        D_diff = np.zeros((2 * (self.N - 1), 2 * self.N))
        R_d_diag = np.tile([self.w_dctrl_steer, self.w_dctrl_acc], self.N - 1)
        R_d = sparse.diags(R_d_diag)

        for i in range(self.N - 1):
            D_diff[2*i:2*(i+1), 2*i:2*(i+1)] = -np.eye(2)
            D_diff[2*i:2*(i+1), 2*(i+1):2*(i+2)] = np.eye(2)

        E_base = np.zeros(4 * self.N)
        for i in range(self.N):
            ref_x, ref_y, ref_v = reference_trajectory[i]
            E_base[4*i]     = x_bar[i + 1, 0] - ref_x
            E_base[4*i + 1] = x_bar[i + 1, 1] - ref_y
            E_base[4*i + 2] = x_bar[i + 1, 2] - ref_v

        E = E_base - S_u @ u_warm_flat

        P_dense = S_u.T @ Q @ S_u + R.toarray() + D_diff.T @ R_d @ D_diff
        P = sparse.csc_matrix(P_dense)
        q = S_u.T @ Q @ E - D_diff.T @ R_d @ D_diff @ u_warm_flat

        A_bounds = sparse.eye(2 * self.N, format='csc')
        l_bounds = -np.ones(2 * self.N)
        u_bounds = np.ones(2 * self.N)

        prob = osqp.OSQP()
        prob.setup(P, q, A_bounds, l_bounds, u_bounds, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
        res = prob.solve()

        if res.info.status == 'solved':
            u_opt = res.x.reshape((self.N, 2))
            return u_opt, res.x

        return u_warm, u_warm_flat

    # ETAPA 2: Filtro de Proyección CBF (Solo para u_0 en R^2)
    def apply_cbf_filter(self, u_nom_first, current_state, obstacles_list):
        if len(obstacles_list) == 0:
            return np.clip(u_nom_first, -1.0, 1.0)

        # min_u 0.5 * || u - u_nom ||^2  ==>  min_u 0.5 * u^T I u - u_nom^T u
        P_filter = sparse.csc_matrix(np.eye(2))
        q_filter = -u_nom_first

        # Límites del actuador (-1 <= u <= 1)
        A_box = sparse.eye(2, format='csc')
        l_box = -np.ones(2)
        u_box = np.ones(2)

        # Matriz B para k=0
        B_0 = KinematicBicycleModel.jacobian_u(current_state, u_nom_first)
        x_pred_1 = KinematicBicycleModel.step(current_state, u_nom_first)

        M = len(obstacles_list)
        A_cbf = np.zeros((M, 2))
        l_cbf = np.zeros(M)
        u_cbf = np.full(M, np.inf)

        for idx, obs in enumerate(obstacles_list):
            obs_x0, obs_y0 = obs[0], obs[1]
            vx_obs = obs[2] if len(obs) > 2 else 0.0
            vy_obs = obs[3] if len(obs) > 3 else 0.0

            obs_x1 = obs_x0 + vx_obs * DT
            obs_y1 = obs_y0 + vy_obs * DT

            v_pred_1 = max(x_pred_1[2], 0.1)
            R_margin = 1.4 + 0.2 * v_pred_1

            h_0 = (current_state[0] - obs_x0)**2 + (current_state[1] - obs_y0)**2 - R_margin**2
            h_1 = (x_pred_1[0] - obs_x1)**2 + (x_pred_1[1] - obs_y1)**2 - R_margin**2

            grad_h_1 = np.array([2 * (x_pred_1[0] - obs_x1), 2 * (x_pred_1[1] - obs_y1), 0.0, 0.0])

            A_cbf[idx, :] = grad_h_1 @ B_0
            l_cbf[idx] = (1.0 - GAMMA_CBF) * h_0 - h_1 + A_cbf[idx, :] @ u_nom_first

        A_qp = sparse.vstack([A_box, sparse.csc_matrix(A_cbf)], format='csc')
        l_qp = np.hstack([l_box, l_cbf])
        u_qp = np.hstack([u_box, u_cbf])

        prob = osqp.OSQP()
        prob.setup(P_filter, q_filter, A_qp, l_qp, u_qp, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
        res = prob.solve()

        if res.info.status == 'solved':
            return res.x
        else:
            # Fallback si la proyección falla
            v_actual = current_state[2]
            return np.array([0.4, 0.15]) if v_actual < 0.8 else np.array([0.0, -1.0])

    def get_action(self, env, state_real, u0_warm):
        vehicle = env.agent
        obstacles_list = self._extract_obstacles(env, vehicle.position)

        road_net = env.engine.map_manager.current_map.road_network
        current_road = vehicle.lane_index
        current_lane = vehicle.lane

        target_lane = current_lane
        should_stop = False

        if self._is_lane_blocked(current_lane, state_real, obstacles_list):
            left_index = (current_road[0], current_road[1], current_road[2] - 1)
            right_index = (current_road[0], current_road[1], current_road[2] + 1)

            left_lane = self._get_lane_safe(road_net, left_index)
            right_lane = self._get_lane_safe(road_net, right_index)

            if left_lane is not None and not self._is_lane_blocked(left_lane, state_real, obstacles_list):
                target_lane = left_lane
            elif right_lane is not None and not self._is_lane_blocked(right_lane, state_real, obstacles_list):
                target_lane = right_lane
            else:
                should_stop = True

        ref_trajectory = self._generate_ref_trajectory(target_lane, state_real, should_stop)

        # 1. Obtener secuencia nominal del MPC
        u_nom_seq, u0_warm_flat = self.solve_nominal_mpc(u0_warm, state_real, ref_trajectory)

        # 2. Aplicar el Filtro de Seguridad CBF solo a u_0
        u_safe_first = self.apply_cbf_filter(u_nom_seq[0], state_real, obstacles_list)

        # Actualizar la secuencia warm start
        u0_warm_next = np.roll(u0_warm_flat, -2)
        u0_warm_next[:2] = u_safe_first
        u0_warm_next[-2:] = u0_warm_next[-4:-2]

        return u_safe_first, u0_warm_next, obstacles_list