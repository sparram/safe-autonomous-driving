import os
import numpy as np
from scipy.optimize import minimize
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

    # Filtro CBF-NLP estricto (sin slack)
    def _filter_cbf_nlp(self, u_nom, state_real, obstacles_list):
        """
        Minimiza: 1/2 || u - u_nom ||^2
        Sujeto a:
          - Actuadores: -1.0 <= u <= 1.0
          - CBF Estricto: h(f(x_k, u)) >= (1 - GAMMA_CBF) * h(x_k)
        """
        if not obstacles_list:
            return np.clip(u_nom, -1.0, 1.0)

        # Punto inicial u = [u_steer, u_accel]
        u0 = np.clip(u_nom, -1.0, 1.0)

        # Función objetivo: minimizar la desviación de la acción de RL
        def objective(u):
            return 0.5 * np.sum((u - u_nom) ** 2)

        # Gradiente analítico de la función objetivo
        def objective_grad(u):
            return u - u_nom

        # Restricciones CBF no lineales estrictas
        constraints = []
        for obs in obstacles_list:
            obs_k = obs[:2]
            v_obs = obs[2:] if len(obs) > 2 else np.zeros(2)
            obs_k1 = obs_k + v_obs * DT

            def cbf_constraint(u, o_k=obs_k, o_k1=obs_k1):
                # Propagación directa a través del modelo cinemático no lineal
                x_next = KinematicBicycleModel.step(state_real, u)
                v_next = max(x_next[2], 0.1)
                R_margin = 2.0 + 0.2 * v_next

                h_k = np.sqrt((state_real[0] - o_k[0]) ** 2 + (state_real[1] - o_k[1]) ** 2) - R_margin
                h_k1 = np.sqrt((x_next[0] - o_k1[0]) ** 2 + (x_next[1] - o_k1[1]) ** 2) - R_margin

                # Expresión g(u) >= 0 estricta
                return h_k1 - (1.0 - GAMMA_CBF) * h_k

            constraints.append({'type': 'ineq', 'fun': cbf_constraint})

        # Límites directos para los controles u in [-1, 1]
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]

        # Resolver optimización estricta con SLSQP
        res = minimize(
            objective,
            u0,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-4, 'maxiter': 50}
        )

        # Si el solver encontró solución factible, retorna u optimizado
        if res.success:
            return res.x

        # Si es infactible o no convergió: Freno de emergencia automático
        return np.array([u_nom[0], -1.0])

    # Interfaz principal de control
    def get_action(self, obs, env, state_real):
        # 1. Acción nominal inferida por la Red Neuronal (RL)
        u_nom, _ = self.model.predict(obs, deterministic=True)
        u_nom = u_nom.astype(np.float64)

        v_curr = max(state_real[2], 0.0)

        # Moderación de velocidad progresiva
        if v_curr > 8.0:
            u_nom[1] = min(u_nom[1], -0.3)

        # Escaneo de obstáculos para frenado crítico directo
        vehicle = env.agent
        lane = vehicle.navigation.current_lane
        vehicles = env.engine.traffic_manager.vehicles
        dist_critica = 0.0 + 0.2 * v_curr
        s_ego, lat_ego = lane.local_coordinates(state_real[:2])

        obstacles_list = self._extract_obstacles(env, vehicle.position)

        for v in vehicles:
            if v != vehicle:
                s_obs, lat_obs = lane.local_coordinates(v.position)
                d_fwd = s_obs - s_ego
                d_right = lat_obs - lat_ego

                if 0 < d_fwd < dist_critica and abs(d_right) < 1.5:
                    u_nom[1] = -1.0
                    return u_nom, obstacles_list

        # 2. Filtrar la acción con el NLP estricto (cae en -1.0 si falla SLSQP)
        u_filtrado = self._filter_cbf_nlp(u_nom, state_real, obstacles_list)

        return u_filtrado, obstacles_list