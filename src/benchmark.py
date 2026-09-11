import os
import csv
import time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config import N, TOTAL_STEPS, MPC_SKIP_STEPS
from controllers.mpc_cbf_controller import MPC_CBF
from controllers.mpc_filter_controller import MPC_CBF_SafetyFilter
from controllers.rl_controller import RLController
from controllers.safe_rl_controller import SafeRLController
from utils.metrics import EpisodeMetrics
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.engine_utils import close_engine, engine_initialized

# Niveles de ruido por magnitud física
NOISE_LEVELS = {
    "none": {
        "pos_std": 0.0,       # metros (GPS)
        "vel_std": 0.0,       # m/s (Odometría)
        "heading_std": 0.0,   # radianes (IMU/Brújula)
        "obs_std": 0.0        # LiDAR / Observación MetaDrive
    },
    "low": {
        "pos_std": 0.05,      # 5 cm
        "vel_std": 0.1,       # 0.36 km/h
        "heading_std": 0.01,  # ~0.57°
        "obs_std": 0.02
    },
    "med": {
        "pos_std": 0.10,      # 20 cm
        "vel_std": 0.25,       # 1.8 km/h
        "heading_std": 0.03,  # ~2.86°
        "obs_std": 0.04
    },
    "high": {
        "pos_std": 0.20,      # 20 cm
        "vel_std": 0.5,       # 1.8 km/h
        "heading_std": 0.05,  # ~2.86°
        "obs_std": 0.08
    }
}

def evaluar_semilla(control_type, controller, seed, env, noise_params=None):
    if noise_params is None:
        noise_params = NOISE_LEVELS["none"]

    obs, info = env.reset(seed=seed)
    u0_warm = np.zeros(N * 2)
    u_action = np.array([0.0, 0.0])
    metrics = EpisodeMetrics()
    tiempos_inferencia = []

    for step in range(TOTAL_STEPS):
        vehicle = env.agent
        
        # 1. Estado Real (Ground Truth)
        state_real = np.array([
            vehicle.position[0],
            vehicle.position[1],
            vehicle.speed_km_h / 3.6,
            vehicle.heading_theta
        ])

        # 2. Vector de perturbación por cada componente del estado
        noise_vector = np.array([
            np.random.normal(0, noise_params["pos_std"]),      # x
            np.random.normal(0, noise_params["pos_std"]),      # y
            np.random.normal(0, noise_params["vel_std"]),      # v
            np.random.normal(0, noise_params["heading_std"])  # theta
        ])
        state_percibido = state_real + noise_vector

        # Ruido en las lecturas de LiDAR/observación del entorno
        if noise_params["obs_std"] > 0:
            obs_percibido = obs + np.random.normal(0, noise_params["obs_std"], size=obs.shape)
        else:
            obs_percibido = obs

        # 3. Inferencia del controlador sobre la percepción ruidosa
        t0 = time.perf_counter()
        se_calculo = False

        if control_type == "RL":
            u_action = controller.get_action(obs_percibido, env, state_percibido)
            se_calculo = True
        elif control_type == "SafeRL":
            u_action, _ = controller.get_action(obs_percibido, env, state_percibido)
            se_calculo = True
        elif control_type in ["MPC-CBF", "MPC-Filter"]:
            if step % MPC_SKIP_STEPS == 0:
                u_action, u0_warm, _ = controller.get_action(env, state_percibido, u0_warm)
                se_calculo = True

        t1 = time.perf_counter()
        if se_calculo:
            tiempos_inferencia.append((t1 - t0) * 1000.0)

        # 4. Avance del simulador con la acción calculada
        obs, reward, terminated, truncated, info = env.step(u_action)

        # 5. Registro de métricas usando Ground Truth (state_real)
        metrics.update(env, vehicle, state_real, u_action, info, terminated, truncated)

        if terminated or truncated:
            break

    res = metrics.get_summary(control_type, seed)
    res["Tiempo Inf. Prom (ms)"] = float(np.mean(tiempos_inferencia)) if tiempos_inferencia else 0.0
    return res


def ejecutar_benchmark(num_escenarios=50, start_seed=37):
    if engine_initialized():
        close_engine()
    
    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=num_escenarios,
        start_seed=start_seed,
        traffic_density=0.15,
        map=5,
        crash_object_done=False,
        out_of_road_done=False
    ))
    
    # Instanciación de los controladores
    controladores = {
        "MPC-CBF": MPC_CBF(horizon=N),
        #"MPC-Filter": MPC_CBF_SafetyFilter(horizon=N),
        "RL": RLController("models_checkpoints/ppo_metadrive.zip"),
        "SafeRL": SafeRLController("models_checkpoints/ppo_metadrive.zip")
    }
    
    resumen_global = []
    detalles_semillas = []
    
    try:
        for control_type, controller in controladores.items():
            print(f"\n[BENCHMARK] Evaluando {control_type} en {num_escenarios} escenarios...")
    
            resultados_lote = []
            for seed in range(start_seed, start_seed + num_escenarios):
                res = evaluar_semilla(control_type, controller, seed, env, NOISE_LEVELS['med'])
                resultados_lote.append(res)
                detalles_semillas.append(res)
    
                print(f"  -> [{control_type}] Semilla {seed}: Éxito={res['Éxito']} | ErrLatProm={res['Err. Lat. Prom (m)']:.2f}m | T.Inf={res['Tiempo Inf. Prom (ms)']:.2f}ms")
                
                # Pausa de 1 segundo entre semillas para gestión térmica
                time.sleep(1.0)
    
            exitos = sum(1 for r in resultados_lote if r["Éxito"] == "SÍ")
            resumen_global.append({
                "Controlador": control_type,
                "Éxito (%)": (exitos / num_escenarios) * 100.0,
                "Err Lat (m)": float(np.mean([r["Err. Lat. Prom (m)"] for r in resultados_lote])),
                "Salida (m)": float(np.mean([r["Exceso Salida (m)"] for r in resultados_lote])),
                "Dist Mín (m)": float(np.mean([r["Dist. Mín Obs (m)"] for r in resultados_lote])),
                "Jerk Prom": float(np.mean([r["Jerk Prom (1/s)"] for r in resultados_lote])),
                "Steer Rate": float(np.mean([r["Steer Rate Prom (rad/s)"] for r in resultados_lote])),
                "Tiempo Inf (ms)": float(np.mean([r["Tiempo Inf. Prom (ms)"] for r in resultados_lote]))
            })
    
            print(f"[BENCHMARK] Pausa de refrigeración (10s) antes del siguiente controlador...")
            time.sleep(10.0)
    
    finally:
        env.close()
    
    print("\n" + "=" * 105)
    print(f"{'CONTROLADOR':<12} | {'ÉXITO (%)':<10} | {'ERR LAT (m)':<12} | {'SALIDA (m)':<10} | {'DIST MÍN (m)':<12} | {'JERK PROM':<10} | {'STEER RATE':<10} | {'T. INF (ms)':<10}")
    print("=" * 105)
    for g in resumen_global:
        print(f"{g['Controlador']:<12} | {g['Éxito (%)']:<10.1f} | {g['Err Lat (m)']:<12.3f} | {g['Salida (m)']:<10.3f} | {g['Dist Mín (m)']:<12.3f} | {g['Jerk Prom']:<10.3f} | {g['Steer Rate']:<10.3f} | {g['Tiempo Inf (ms)']:<10.2f}")
    print("=" * 105)
    
    # Exportación de CSVs
    with open("benchmark_resumen.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resumen_global[0].keys())
        writer.writeheader()
        writer.writerows(resumen_global)
    
    with open("benchmark_detalles.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detalles_semillas[0].keys())
        writer.writeheader()
        writer.writerows(detalles_semillas)
    
    print("\n¡Resultados guardados en 'benchmark_resumen.csv' y 'benchmark_detalles.csv'!")

if __name__ == "__main__":
    ejecutar_benchmark(num_escenarios=50, start_seed=37)
