import numpy as np
import cupy as cp
from rasterio import open as rio
import matplotlib.pyplot as plt
from imageio import imwrite
import os
import json
from tkinter import Tk, ttk, filedialog, messagebox
from datetime import datetime
import threading
from numba import jit
import tkinter as tk





# 主程序类
class FloodSimulatorApp:
    def __init__(self):
        self.root = Tk()
        self.root.title("二维浅水方程洪水模拟系统")

        self.params = self.load_default_params()
        self.gui = self.create_gui(self.root)
        self.control_panel = self.create_control_panel(self.gui)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 新增线程相关属性
        self.simulation_thread = None
        self.simulator = None

    def load_default_params(self):
        return {
            'dem_path': 'input_dem.tif',
            'vol_curve_path': 'volume_curve.csv',
            'output_dir': './output',
            'dx': 10.0,
            'dy': 10.0,
            'dt_max': 3600,
            'cfl_factor': 0.9,
            'simulation_time': 86400,
            'visualization_interval': 3600,
            'adaptive_dt': True,
            'gpu_threads': 0,
            'mannings_n': 0.01
        }

    def create_gui(self, parent):
        self.notebook = ttk.Notebook(parent)

        self.param_page = self.create_parameter_page(self.notebook)
        self.notebook.add(self.param_page, text='参数配置')

        self.io_page = self.create_io_page(self.notebook)
        self.notebook.add(self.io_page, text='输入输出')

        self.control_panel = self.create_control_panel(self.notebook)
        self.notebook.add(self.control_panel, text='运行控制')

        self.status_label = ttk.Label(parent, text='', font=('Arial', 12))
        self.status_label.pack(side='bottom', fill='x', padx=20, pady=10)

        return self.notebook

    def create_parameter_page(self, parent):
        frame = ttk.Frame(parent)

        # 添加网格参数控件
        self.dx_var = tk.DoubleVar(value=10.0)
        self.dy_var = tk.DoubleVar(value=10.0)
        ttk.Label(frame, text='网格分辨率(m)').grid(row=0, column=0)
        ttk.Entry(frame, textvariable=self.dx_var).grid(row=0, column=1)
        ttk.Entry(frame, textvariable=self.dy_var).grid(row=1, column=1)

        # 添加时间参数控件
        self.sim_time_var = tk.DoubleVar(value=86400)
        self.dt_max_var = tk.DoubleVar(value=3600)
        self.cfl_factor_var = tk.DoubleVar(value=0.9)
        self.adaptive_dt_var = tk.BooleanVar(value=True)
        frame_adaptive = ttk.Frame(frame)
        ttk.Checkbutton(frame_adaptive, text='启用自适应时间步长', variable=self.adaptive_dt_var).grid(row=2, column=0)
        ttk.Label(frame_adaptive, text='CFL系数:').grid(row=2, column=1)
        ttk.Entry(frame_adaptive, textvariable=self.cfl_factor_var, width=5).grid(row=2, column=2)

        # 添加曼宁系数控件
        self.mannings_n_var = tk.DoubleVar(value=0.01)
        ttk.Label(frame, text='曼宁系数(n):').grid(row=3, column=0)
        ttk.Entry(frame, textvariable=self.mannings_n_var, width=5).grid(row=3, column=1)

        return frame

    def create_io_page(self, parent):
        frame = ttk.Frame(parent)

        # 文件选择控件
        self.dem_path_var = tk.StringVar()
        self.vol_curve_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value='./output')

        ttk.Label(frame, text='输入文件').grid(row=0, column=0)
        ttk.Entry(frame, textvariable=self.dem_path_var).grid(row=0, column=1)
        ttk.Button(frame, text='...', command=lambda: self.browse_file('dem')).grid(row=0, column=2)

        ttk.Label(frame, text='库容曲线').grid(row=1, column=0)
        ttk.Entry(frame, textvariable=self.vol_curve_path_var).grid(row=1, column=1)
        ttk.Button(frame, text='...', command=lambda: self.browse_file('vol')).grid(row=1, column=2)

        ttk.Label(frame, text='输出目录').grid(row=2, column=0)
        ttk.Entry(frame, textvariable=self.output_dir_var).grid(row=2, column=1)
        ttk.Button(frame, text='...', command=lambda: self.browse_folder()).grid(row=2, column=2)

        return frame

    def create_control_panel(self, parent):
        frame = ttk.Frame(parent)

        self.run_button = ttk.Button(frame, text='开始模拟', command=self.start_simulation)
        self.run_button.pack(side='left', padx=10, pady=10)

        self.pause_resume = ttk.Button(frame, text='暂停', state='disabled', command=lambda: self.pause_resume())
        self.pause_resume.pack(side='left', padx=10, pady=10)

        self.terminate_button = ttk.Button(frame, text='终止', state='disabled', command=lambda: self.terminate())
        self.terminate_button.pack(side='left', padx=10, pady=10)

        self.progress = ttk.Progressbar(frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(side='right', fill='x', padx=10, pady=10)

        return frame

    def browse_file(self, type_):
        filename = filedialog.askopenfilename()
        if type_ == 'dem':
            self.dem_path_var.set(filename)
        elif type_ == 'vol':
            self.vol_curve_path_var.set(filename)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        self.output_dir_var.set(folder)

    def save_params(self):
        with open('config.json', 'w') as f:
            json.dump(self.params, f, indent=4)
        messagebox.showinfo("成功", "参数配置已保存！")

    def load_params(self):
        try:
            with open('config.json', 'r') as f:
                loaded_params = json.load(f)
                # 类型转换
                loaded_params['dx'] = float(loaded_params['dx'])
                loaded_params['dy'] = float(loaded_params['dy'])
                loaded_params['dt_max'] = float(loaded_params['dt_max'])
                loaded_params['simulation_time'] = float(loaded_params['simulation_time'])
                loaded_params['cfl_factor'] = float(loaded_params['cfl_factor'])
                loaded_params['gpu_threads'] = int(loaded_params['gpu_threads'])
                loaded_params['mannings_n'] = float(loaded_params['mannings_n'])

                # 更新参数
                self.params.update(loaded_params)
                self.update_gui_from_params()
        except Exception as e:
            messagebox.showerror("错误", f"加载配置失败: {str(e)}")

    def update_gui_from_params(self):
        self.dem_path_var.set(self.params['dem_path'])
        self.vol_curve_path_var.set(self.params['vol_curve_path'])
        self.output_dir_var.set(self.params['output_dir'])

        self.dx_var.set(self.params['dx'])
        self.dy_var.set(self.params['dy'])

        self.sim_time_var.set(self.params['simulation_time'])
        self.dt_max_var.set(self.params['dt_max'])
        self.cfl_factor_var.set(self.params['cfl_factor'])
        self.adaptive_dt_var.set(self.params['adaptive_dt'])
        self.gpu_threads_var.set(self.params['gpu_threads'])
        self.mannings_n_var.set(self.params['mannings_n'])

    def start_simulation(self):
        # 参数验证
        if not os.path.exists(self.dem_path_var.get()):
            messagebox.showerror("错误", "DEM文件不存在！")
            return

        if not os.path.exists(self.vol_curve_path_var.get()):
            messagebox.showerror("错误", "库容曲线文件不存在！")
            return

        # 创建输出目录
        os.makedirs(self.output_dir_var.get(), exist_ok=True)

        # 初始化模拟器
        self.simulator = ShallowWaterSimulator(
            dem_path=self.dem_path_var.get(),
            vol_curve_path=self.vol_curve_path_var.get(),
            output_dir=self.output_dir_var.get(),
            dx=self.dx_var.get(),
            dy=self.dy_var.get(),
            dt_max=self.dt_max_var.get(),
            cfl_factor=self.cfl_factor_var.get(),
            simulation_time=self.sim_time_var.get(),
            visualization_interval=3600,
            adaptive_dt=self.adaptive_dt_var.get(),
            gpu_threads=self.gpu_threads_var.get(),
            manning_n=self.mannings_n_var.get()
        )

        # 更新GUI状态
        self.run_button.config(state='disabled')
        self.pause_resume.config(state='normal')
        self.terminate_button.config(state='normal')
        self.progress['value'] = 0

        # 启动模拟主循环
        self.simulation_thread = threading.Thread(target=self.simulator.execute_simulation)
        self.simulation_thread.start()

    def pause_resume(self):
        if self.simulator.is_paused:
            self.simulator.resume()
            self.pause_resume.config(text='暂停')
        else:
            self.simulator.pause()
            self.pause_resume.config(text='继续')

    def terminate(self):
        if self.simulation_thread is not None and self.simulation_thread.is_alive():
            self.simulator.terminate()
            self.simulation_thread.join()
        self.run_button.config(state='normal')
        self.pause_resume.config(state='disabled')
        self.terminate_button.config(state='disabled')
        self.progress['value'] = 0
        messagebox.showinfo("提示", "模拟已终止！")

    def on_close(self):
        self.terminate()
        self.root.destroy()


# 模拟器核心类
class ShallowWaterSimulator:
    def __init__(self, **kwargs):
        super().__init__()
        self.dem_path = kwargs['dem_path']
        self.vol_curve_path = kwargs['vol_curve_path']
        self.output_dir = kwargs['output_dir']

        self.dx = kwargs['dx']
        self.dy = kwargs['dy']

        self.dt_max = kwargs['dt_max']
        self.cfl_factor = kwargs['cfl_factor']
        self.simulation_time = kwargs['simulation_time']
        self.visualization_interval = 3600
        self.adaptive_dt = kwargs['adaptive_dt']
        self.gpu_threads = kwargs['gpu_threads']
        self.mannings_n = kwargs['mannings_n']

        # 加载数据
        self.dem = rio.open(self.dem_path).read(1)
        self.dem_cp = cp.asarray(self.dem, dtype=np.float32)

        self.vol_curve = np.loadtxt(self.vol_curve_path, delimiter=',').astype(np.float64)
        self.time_vol = self.vol_curve[:, 0]
        self.level_vol = self.vol_curve[:, 1]

        # 初始化GPU数组
        self.grid = Grid(dx=self.dx, dy=self.dy, dem_shape=self.dem_cp.shape)
        self.h = cp.zeros_like(self.dem_cp, dtype=np.float32)
        self.u = cp.zeros_like(self.dem_cp, dtype=np.float32)
        self.v = cp.zeros_like(self.dem_cp, dtype=np.float32)

        # CUDA配置
        cp.cuda.set_device(self.gpu_threads)
        self.streams = [cp.cuda.Stream() for _ in range(4)]

        # 内存池管理
        self.memory_pool = cp.MemoryPool(max_size=2 * 1024 ** 3)
        self.h = cp.zeros_like(self.dem_cp, allocator=self.memory_pool)
        self.u = cp.zeros_like(self.dem_cp, allocator=self.memory_pool)
        self.v = cp.zeros_like(self.dem_cp, allocator=self.memory_pool)

        # 物理参数
        self.gravity = 9.81
        self.current_time = 0.0
        self.last_visualization_time = 0.0
        self.is_paused = False
        self.progress_callback = None

    @jit(nopython=True)
    def compute_speed_max(self, h, u, v, dx, dy):
        max_u = cp.max(abs(u))
        max_v = cp.max(abs(v))
        dt_x = dx / max_u if max_u > 1e-9 else float('inf')
        dt_y = dy / max_v if max_v > 1e-9 else float('inf')
        return min(dt_x, dt_y) * self.cfl_factor

    def compute_fluxes(self):
        with cp.cuda.stream(self.streams[1]):
            q = self.h * self.u
            r = self.h * self.v

            # 曼宁阻力计算
            speed_sq = u ** 2 + v ** 2
            speed = cp.sqrt(speed_sq)
            friction = (self.mannings_n * speed ** 1.5) / cp.sqrt(self.h)

            du_dt = -friction / self.h
            dv_dt = -friction / self.h

            # 计算通量
            flux_qx = (q ** 2 / (2 * self.h) + self.h * u ** 2) if self.h != 0 else 0
            flux_qy = (q * r) / self.h + self.h * u * v
            flux_rx = (r ** 2 / (2 * self.h) + self.h * v ** 2) if self.h != 0 else 0
            flux_ry = (q * r) / self.h + self.h * u * v

            return flux_qx, flux_qy, flux_rx, flux_ry

    def update_h(self, flux_qx, flux_qy, flux_rx, flux_ry):
        with cp.cuda.stream(self.streams[2]):
            # 更新速度场
            dh_dx = cp.gradient(self.h, dx=1, axis=0)[0]
            dh_dy = cp.gradient(self.h, dx=1, axis=1)[0]

            # 添加压力梯度项
            pressure_gradient_x = cp.gradient(cp.gradient(self.h, dx=1, axis=0)[0], dx=1, axis=0)[0] * self.gravity
            pressure_gradient_y = cp.gradient(cp.gradient(self.h, dx=1, axis=1)[0], dx=1, axis=0)[0] * self.gravity

            self.u += (-dh_dx + pressure_gradient_x) / self.gravity + du_dt
            self.v += (-dh_dy + pressure_gradient_y) / self.gravity + dv_dt

            # 更新水位
            h_new = self.h.copy()

            h_new[1:] -= (flux_qx[1:] - flux_qx[:-1]) / (2 * self.dx)
            h_new[:, 1:] -= (flux_ry[1:] - flux_ry[:-1]) / (2 * self.dy)

            self.h = h_new

    def execute_simulation(self):
        self.is_paused = False
        while self.current_time < self.simulation_time and not self.is_paused:
            self.update_boundary_conditions()
            dt = self.compute_adaptive_dt()

            if dt <= 0:
                break

            self.execute_time_step(dt)
            self.current_time += dt

            # 更新进度条
            if self.progress_callback is not None:
                self.progress_callback(min(100.0, (self.current_time / self.simulation_time) * 100))

            # 处理输出
            if (
                    self.current_time - self.last_visualization_time) >= self.visualization_interval or self.current_time == self.simulation_time:
                self.save_visualization(self.current_time)
                self.last_visualization_time = self.current_time

        # 最终保存
        self.save_final_results()

    def update_boundary_conditions(self):
        current_time = self.current_time
        idx = np.argmin(np.abs(self.time_vol - current_time))
        reservoir_level = self.level_vol[idx]
        self.h[:, 0] = reservoir_level

    def compute_adaptive_dt(self):
        dt = self.compute_speed_max(self.h, self.u, self.v, self.dx, self.dy)
        return min(dt, self.dt_max)

    def execute_time_step(self, dt):
        with cp.cuda.stream(self.streams[3]):
            # 计算通量
            flux_qx, flux_qy, flux_rx, flux_ry = self.compute_fluxes()

            # 更新水位
            self.update_h(flux_qx, flux_qy, flux_rx, flux_ry)

            # 更新压力梯度
            dh_dx = cp.gradient(self.h, dx=1, axis=0)[0]
            self.pressure_gradient = cp.gradient(dh_dx, dx=1, axis=0)[0] * self.gravity

            # 更新速度场
            self.u += (-dh_dx + self.pressure_gradient) / self.gravity
            self.v += (-cp.gradient(self.h, dx=1, axis=1)[0] + self.pressure_gradient) / self.gravity

    def save_visualization(self, time):
        plt.imshow(self.h.cpu(), cmap='viridis', vmin=0, vmax=10)
        plt.colorbar(label='Water Depth (m)')
        plt.title(f'Simulation Time: {datetime.fromtimestamp(time).strftime("%H:%M:%S")}')
        plt.savefig(os.path.join(self.output_dir, f't_{int(time // 60):03d}.png'))
        plt.close()

    def save_final_results(self):
        with rio.open(
                os.path.join(self.output_dir, 'final_h.tif'),
                'w',
                driver='GTiff',
                height=self.dem_cp.shape[0],
                width=self.dem_cp.shape[1],
                count=1,
                dtype=np.float32,
                crs='EPSG:4326'
        ) as dst:
            dst.write(self.h.cpu().reshape(-1, 1), 1)

        np.savetxt(
            os.path.join(self.output_dir, 'final_h.csv'),
            self.h.cpu().flatten(),
            delimiter=',',
            header='Easting, Northing, Water Depth (m)'
        )


# 辅助类
class Grid:
    def __init__(self, dx, dy, dem_shape):
        self.dx = dx
        self.dy = dy
        self.nx = dem_shape[1]
        self.ny = dem_shape[0]
        self.x = np.linspace(0, self.nx * dx, self.nx)
        self.y = np.linspace(0, self.ny * dy, self.ny)


if __name__ == '__main__':
    app = FloodSimulatorApp()
    app.root.mainloop()
