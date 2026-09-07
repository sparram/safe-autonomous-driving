import os
import numpy as np
import osqp
from scipy import sparse
from stable_baselines3 import PPO
from config import DT, GAMMA_CBF
from models.kinematic import KinematicBicycleModel


class SafeRLController:
    def __init__(self, model_path="models_checkpoints/ppo_metadrive.zip"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found in path: {model_path}.")
        self.model = PPO.load(model_path)

    # Extrae obstáculos dinámicos en un radio de seguridad
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

    # Filtro CBF-QP de 1 paso: Proyecta u_nom al conjunto seguro
    def _filter_cbf_qp(self, u_nom, state_real, obstacles_list):
        """
        Minimiza 1/2 || u - u_nom ||^2
        Sujeto a:
          - Límites del actuador: -1.0 <= u <= 1.0
          - Restricción CBF: h_k1(u) >= (1 - GAMMA_CBF) * h_k
        """
        if not obstacles_list:
            return np.clip(u_nom, -1.0, 1.0)

        # Matriz Hessiana P = I y gradiente q = -u_nom (Minimiza ||u - u_nom||^2)
        P = sparse.csc_matrix(np.eye(2))
        q = -u_nom.astype(np.float64)

        # Estado nominal predicho a 1 paso y Jacobiano B respecto al control
        x_next_nom = KinematicBicycleModel.step(state_real, u_nom)
        B = KinematicBicycleModel.jacobian_u(state_real, u_nom)

        A_cbf_list = []
        l_cbf_list = []
        u_cbf_list = []

        # 1. Limites fisicos del actuador [-1, 1]
        A_cbf_list.append(np.eye(2))
        l_cbf_list.append(np.array([-1.0, -1.0]))
        u_cbf_list.append(np.array([1.0, 1.0]))

        # 2. Construcción de restricciones CBF por obstáculo
        for obs in obstacles_list:
            obs_x0, obs_y0 = obs[0], obs[1]
            vx_obs = obs[2] if len(obs) > 2 else 0.0
            vy_obs = obs[3] if len(obs) > 3 else 0.0

            # Posición proyectada del obstáculo
            obs_k = np.array([obs_x0, obs_y0])
            obs_k1 = np.array([obs_x0 + vx_obs * DT, obs_y0 + vy_obs * DT])

            v_next = max(x_next_nom[2], 0.1)
            R_margin = 1.4 + 0.2 * v_next

            # Valores de CBF en k y k+1 nominal
            h_k = (state_real[0] - obs_k[0])**2 + (state_real[1] - obs_k[1])**2 - R_margin**2
            h_k1_nom = (x_next_nom[0] - obs_k1[0])**2 + (x_next_nom[1] - obs_k1[1])**2 - R_margin**2

            # Gradiente de h respecto al estado
            dh_dx = 2 * (x_next_nom[0] - obs_k1[0])
            dh_dy = 2 * (x_next_nom[1] - obs_k1[1])
            grad_h = np.array([dh_dx, dh_dy, 0.0, 0.0])

            # Sensibilidad con respecto al control (grad_h @ B)
            a_row = grad_h @ B
            
            # Condición CBF: grad_h @ B @ u >= (1 - GAMMA_CBF) * h_k - h_k1_nom + grad_h @ B @ u_nom
            l_val = (1.0 - GAMMA_CBF) * h_k - h_k1_nom + a_row @ u_nom

            A_cbf_list.append(a_row.reshape(1, 2))
            l_cbf_list.append(np.array([l_val]))
            u_cbf_list.append(np.array([np.inf]))

        A_qp = sparse.csc_matrix(np.vstack(A_cbf_list))
        l_qp = np.hstack(l_cbf_list)
        u_qp = np.hstack(u_cbf_list)

        # Resolver QP con OSQP
        prob = osqp.OSQP()
        prob.setup(P, q, A_qp, l_qp, u_qp, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
        res = prob.solve()

        if res.info.status == 'solved':
            return res.x

        # Fallback si el QP es infactible: frenado de emergencia
        return np.array([0.0, -1.0])

    # Interfaz principal de control
    def get_action(self, obs, env, state_real):
        # 1. Acción nominal inferida por la Red Neuronal (RL)
        u_nom, _ = self.model.predict(obs, deterministic=True)
        u_nom = u_nom.astype(np.float64)

        # 2. Obtener obstáculos cercanos
        vehicle = env.agent
        obstacles_list = self._extract_obstacles(env, vehicle.position)

        # 3. Filtrar la acción con el QP de CBF
        u_safe = self._filter_cbf_qp(u_nom, state_real, obstacles_list)

        return u_safe, obstacles_list