import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class Particle:
    """离散元粒子类，包含物理属性和运动状态"""

    def __init__(self, id, radius, mass, position):
        self.id = id  # 粒子唯一标识
        self.radius = radius  # 粒子半径
        self.mass = mass  # 质量
        self.position = np.array(position, dtype=float)  # 位置坐标
        self.velocity = np.zeros(2)  # 速度向量
        self.normal_force = 0.0  # 法向力
        self.slip_state = False  # 滑动状态标记
        self.force = np.zeros(2)  # 合力向量
        self.torque = 0.0  # 扭矩
        self.angular_vel = 0.0  # 角速度
        self.angle = 0.0  # 角度


class DEMSimulator:
    """离散元模拟主类，管理粒子系统、物理模型和数值计算"""

    def __init__(self):
        self.particles = []  # 粒子列表
        self.time = 0.0  # 当前时间
        self.dt = 1e-5  # 初始时间步长
        self.gravity = 9.81  # 重力加速度
        self.kn = 5e5  # 法向刚度
        self.kt = 1e5  # 切向刚度
        self.mu_s = 0.6  # 静态摩擦系数
        self.mu_d = 0.3  # 动态摩擦系数
        self.cohesion = 100  # 粘结强度
        self.damping = 0.1  # 阻尼系数

        # 可视化参数
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(-2, 4)
        self.ax.set_ylim(-2, 4)
        self.ax.set_aspect('equal')
        self.ax.set_title("Slope Instability with Cohesion and Friction")

        self._draw_slope()  # 绘制边坡几何
        self.particle_artists = []  # 粒子可视化对象

        # 数据记录字典
        self.history = {
            'time': [],
            'displacements': {},  # 粒子ID -> 位移历史
            'velocities': {}  # 粒子ID -> 速度历史
        }
        self.initial_positions = {}  # 粒子ID -> 初始位置

    def _draw_slope(self):
        """绘制边坡几何形状"""
        self.ax.plot([-2, 0, 4], [0, 0, 0], 'k--', lw=3)  # 基底
        self.ax.plot([-2, 0, 4], [0, 2, 0], 'k:', lw=2)  # 边坡表面
        self.ax.text(1, 2.5, "Slope Surface", ha='center', color='darkgreen', fontsize=12)
        self.ax.text(-1.5, -0.5, "Base", ha='center', color='darkblue', fontsize=10)

    def add_particle(self, particle):
        """添加粒子并初始化可视化对象"""
        self.particles.append(particle)
        self.initial_positions[particle.id] = particle.position.copy()

        # 创建圆形可视化对象
        artist = Circle(
            (particle.position[0], particle.position[1]),
            particle.radius,
            facecolor='blue', alpha=0.7, edgecolor='black'
        )
        self.ax.add_patch(artist)
        self.particle_artists.append(artist)

        # 初始化历史数据记录
        self.history['displacements'][particle.id] = []
        self.history['velocities'][particle.id] = []

    def add_particles_random(self, num, base_pos, radius_range, mass):
        """随机生成指定数量的粒子"""
        for i in range(num):
            radius = np.random.uniform(radius_range[0], radius_range[1])
            x = base_pos[0] + np.random.uniform(-1, 1) * radius
            y = base_pos[1] + np.random.uniform(0, 1.5) * radius
            self.add_particle(Particle(i + 1, radius, mass, [x, y]))

    def compute_forces(self):
        """计算所有粒子的法向力和切向力"""
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                p1 = self.particles[i]
                p2 = self.particles[j]
                delta = p2.position - p1.position
                dist = np.linalg.norm(delta)

                if dist < (p1.radius + p2.radius - 1e-8):
                    overlap = p1.radius + p2.radius - dist
                    n = delta / dist  # 单位法向量
                    fn = self.kn * overlap  # 法向力

                    # 更新滑动状态
                    p1.slip_state = False
                    p2.slip_state = False
                    if abs(p1.normal_force) > 1e-8 or abs(p2.normal_force) > 1e-8:
                        p1.slip_state = True
                        p2.slip_state = True

                    # 切向力计算（库仑摩擦模型）
                    vt = p2.velocity - p1.velocity
                    vt -= np.dot(vt, n) * n  # 去除法向分量
                    if np.linalg.norm(vt) == 0:
                        ft = np.zeros(2)
                    else:
                        mu = self.mu_d if p1.slip_state or p2.slip_state else self.mu_s
                        max_ft = mu * fn
                        current_ft_mag = self.kt * np.linalg.norm(vt) * self.dt
                        ft = -min(current_ft_mag, max_ft) * vt / np.linalg.norm(vt)

                    p1.force += ft
                    p2.force -= ft

                    # 计算扭矩
                    contact_point_p1 = p1.position + n * (-p1.radius)
                    r_p1 = contact_point_p1 - p1.position
                    torque_p1 = np.cross(r_p1, ft)
                    p1.torque += torque_p1

                    contact_point_p2 = p2.position + n * (-p2.radius)
                    r_p2 = contact_point_p2 - p2.position
                    torque_p2 = np.cross(r_p2, -ft)
                    p2.torque += torque_p2

    def integrate_motion(self):
        """积分运动方程更新粒子位置和速度"""
        for p in self.particles:
            acceleration = p.force / p.mass

            # 平移运动更新
            p.velocity += acceleration * self.dt
            p.position += p.velocity * self.dt

            # 旋转运动更新
            if p.mass == 0:
                angular_acc = 0.0
            else:
                angular_acc = p.torque / (0.4 * p.mass * p.radius ** 2)
            p.angular_vel += angular_acc * self.dt
            p.angle += p.angular_vel * self.dt

            # 重置法向力
            p.normal_force = 0.0

    def adaptive_time_step(self):
        """自适应调整时间步长"""
        if not self.particles:
            return self.dt

        max_acceleration = 0.0
        for p in self.particles:
            acceleration = p.force / p.mass
            current_acc = np.linalg.norm(acceleration)
            max_acceleration = max(max_acceleration, current_acc)

        safety_factor = 0.5
        dt_max = 1e-4
        dt = safety_factor * (2 * self.dt) / max_acceleration if max_acceleration > 1e-10 else dt_max
        dt = min(dt, dt_max)
        dt = max(dt, 1e-6)
        return dt

    def update_display(self):
        """更新可视化窗口"""
        max_speed = 0.0
        for p in self.particles:
            speed = np.linalg.norm(p.velocity)
            max_speed = max(max_speed, speed)

        for artist, p in zip(self.particle_artists, self.particles):
            artist.center = p.position
            speed = np.linalg.norm(p.velocity)
            color = plt.cm.viridis(speed / (max_speed + 1e-6))
            artist.set_facecolor(color)

            if p.slip_state:
                artist.set_edgecolor('red')
                artist.set_alpha(0.9)

        self.fig.canvas.draw()
        plt.pause(0.001)

    def run(self, total_time):
        """主模拟循环"""
        self.history['time'] = []
        self.history['displacements'] = {p.id: [] for p in self.particles}
        self.history['velocities'] = {p.id: [] for p in self.particles}

        with open('simulation_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'ParticleID', 'X', 'Y', 'Vx', 'Vy'])

            while self.time < total_time:
                self.compute_forces()
                self.dt = self.adaptive_time_step()
                self.integrate_motion()

                # 记录数据
                self.history['time'].append(self.time)
                for p in self.particles:
                    displacement = np.linalg.norm(p.position - self.initial_positions[p.id])
                    self.history['displacements'][p.id].append(displacement)
                    self.history['velocities'][p.id].append(p.velocity.copy())
                    writer.writerow([self.time, p.id, p.position[0], p.position[1], p.velocity[0], p.velocity[1]])

                # 更新可视化
                self.update_display()
                self.time += self.dt

        self.plot_results()
        plt.ioff()
        plt.show()

    def plot_results(self):
        """绘制最终结果分析图"""
        plt.figure(figsize=(15, 10))

        # 位移场分析
        displacement = np.zeros(len(self.particles))
        for i, p in enumerate(self.particles):
            displacement[i] = self.history['displacements'][p.id][-1]

        plt.subplot(2, 2, 1)
        plt.scatter([p.position[0] for p in self.particles], [p.position[1] for p in self.particles], c=displacement,
                    cmap='viridis')
        plt.colorbar(label='Displacement (m)')
        plt.title('Final Displacement Field')

        # 速度场分析
        velocity = np.zeros(len(self.particles))
        for i, p in enumerate(self.particles):
            velocity[i] = np.linalg.norm(self.history['velocities'][p.id][-1])

        plt.subplot(2, 2, 2)
        plt.scatter([p.position[0] for p in self.particles], [p.position[1] for p in self.particles], c=velocity,
                    cmap='plasma')
        plt.colorbar(label='Speed (m/s)')
        plt.title('Final Velocity Field')

        # 滑动区域标记
        slip_particles = [p.id for p in self.particles if p.slip_state]
        plt.subplot(2, 2, 3)
        for i, p in enumerate(self.particles):
            if p.id in slip_particles:
                plt.plot(p.position[0], p.position[1], 'ro', markersize=8)
            else:
                plt.plot(p.position[0], p.position[1], 'bo', markersize=4)
        plt.title('Slip Zones')

        # 时间-位移曲线
        for p in self.particles:
            plt.subplot(2, 2, 4)
            plt.plot(self.history['time'], self.history['displacements'][p.id], label=f'P{p.id}')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.title('Displacement vs Time')


if __name__ == "__main__":
    # 模拟参数配置
    sim = DEMSimulator()
    sim.add_particles_random(150, [0.0, 0.0], [0.05, 0.15], 1.0)

    # 物理参数设置
    sim.kn = 5e5  # 法向刚度
    sim.kt = 1e5  # 切向刚度
    sim.cohesion = 100  # 粘结强度
    sim.mu_s = 0.6  # 静态摩擦系数
    sim.mu_d = 0.3  # 动态摩擦系数

    # 运行模拟
    sim.run(total_time=2.0)