import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import inv, LinearOperator, cg

import matplotlib.animation as animation
from IPython.display import HTML
from IPython.display import display


class Layer:
    """Represents a layer in the simulation with a certain thickness."""

    def __init__(self, width, initial_thickness, dz):
        """Initialize the layer with a certain width, initial thickness, and spatial resolution."""
        self.width = width
        self.thickness = np.full(int(width / dz), float(initial_thickness))

    def update_thickness(self, etch_rate, dt):
        """Update the thickness of the layer based on the etch rate and time step."""
        self.thickness -= etch_rate * dt

    def get_thickness_at(self, position):
        """Return the thickness of the layer at the given position."""
        index = int(position / self.dz)
        return self.thickness[index]

    def get_thickness(self):
        """Return the thickness of the layer."""
        return self.thickness

    def show(self):
        """Display the thickness of the layer."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.thickness)
        plt.title("Thickness of the Layer")
        plt.xlabel("Position")
        plt.ylabel("Thickness")
        plt.grid(True)
        plt.show()


class Mesh:

    def __init__(self, width, height, dx=1, dy=1):
        self.width = width
        self.height = height
        self.dx = dx
        self.dy = dy
        self.x_grid, self.y_grid = np.meshgrid(
            np.arange(0, width, dx), np.arange(0, height, dy)
        )

    def refine_region(self, x_start, x_end, y_start, y_end, refinement_factor=2):
        """

        :param x_start:
        :param x_end:
        :param y_start:
        :param y_end:
        :param refinement_factor:
        :return:
        """
        refined_dx = self.dx / refinement_factor
        refined_dy = self.dy / refinement_factor

        x_refined = np.arange(x_start, x_end, refined_dx)
        y_refined = np.arange(y_start, y_end, refined_dy)

        refined_x_grid, refined_y_grid = np.meshgrid(x_refined, y_refined)

        x_indices = np.where(
            (self.x_grid[0, :] >= x_start) & (self.x_grid[0, :] < x_end)
        )[0]
        y_indices = np.where(
            (self.y_grid[:, 0] >= y_start) & (self.y_grid[:, 0] < y_end)
        )[0]

        for i, x_idx in enumerate(x_indices):
            for j, y_idx in enumerate(y_indices):
                self.x_grid[y_idx, x_idx] = refined_x_grid[j, i]
                self.y_grid[y_idx, x_idx] = refined_y_grid[j, i]

    def display_mesh(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.x_grid, self.y_grid, s=10, c="blue")
        plt.title("Mesh Grid")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()


class Contact:
    def __init__(self, position, potential=1.0):
        self.position = position
        self.potential = potential


class Electrolyte:
    def __init__(self, base_conductivity, agitation_factor=1.0):
        self.base_conductivity = base_conductivity
        self.agitation_factor = agitation_factor

    @property
    def adjusted_conductivity(self):
        return self.base_conductivity * self.agitation_factor


class SimulationDomain:
    def __init__(self, width, height, dz, J_1D):
        self.width = width
        self.height = height
        self.dz = dz
        self.nz = int(self.width / self.dz)
        self.electric_field = None
        self.J_1D = J_1D

    def compute_electric_field(self, nz):
        x_original = np.linspace(0, self.width, self.J_1D.size)
        x_new = np.linspace(0, self.width, nz)
        interpolator = interp1d(
            x_original, self.J_1D, kind="linear", fill_value="extrapolate"
        )
        self.electric_field = interpolator(x_new)

    def get_electric_field_at(self, position):
        index = int(position / self.dz)
        return self.electric_field[index]


class EtchingSimulation:
    def __init__(self, domain, substrate, k, J_1D):
        self.domain = SimulationDomain(domain.width, domain.height, domain.dz, J_1D)
        self.substrate = substrate
        self.k = k

    def run(self, time_steps, dt):
        history = []

        for _ in range(time_steps):
            self.domain.compute_electric_field(len(self.substrate.thickness))

            etch_rate = self.k * self.domain.electric_field

            self.substrate.update_thickness(etch_rate, dt)

            history.append(self.substrate.thickness.copy())

        return history


class Visualization:
    def __init__(self, initial_data):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.cax = self.ax.imshow(initial_data, cmap="inferno", animated=True)
        self.fig.colorbar(self.cax, label="Electric Field Magnitude")

    def update_frame(self, data):
        data_2d = np.hstack([data.reshape(-1, 1)] * 10)
        self.cax.set_array(data_2d)
        return (self.cax,)  # comma is important

    def animate(self, data_sequence):
        ani = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            frames=data_sequence,
            blit=True,
            interval=200,
            repeat=True,
        )
        return ani

    @staticmethod
    def show_html(ani):
        display(HTML(ani.to_html5_video()))

    @staticmethod
    def show_plot(ani):
        plt.show()

    @staticmethod
    def save_gif(ani, filename):
        ani.save(filename, writer="imagemagick", fps=1)

    @staticmethod
    def save_mp4(ani, filename):
        ani.save(filename, writer="ffmpeg", fps=1)

    @staticmethod
    def save_html(ani, filename):
        ani.save(filename, writer="html", fps=1)


class ElectrostaticSolver:
    def __init__(self, plate_config, mesh, resistance=1.0, accuracy=1e-4):
        self.plate_config = plate_config
        self.mesh = mesh
        self.resistance = resistance
        self.accuracy = accuracy

    def _generate_laplacian(self):
        height, width = self.mesh.y_grid.shape
        size = height * width
        dx, dy = self.mesh.dx, self.mesh.dy
        main_diag = np.ones(size) * (-2 / dx**2 - 2 / dy**2)
        off_diag_x = np.ones(size - 1) * 1 / dx**2
        off_diag_y = np.ones(size - width) * 1 / dy**2
        diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
        laplacian = diags(diagonals, [0, -1, 1, -width, width]).tocsr()
        return laplacian

    @staticmethod
    def generate_j_1d(self, width, points):
        """Generate a default J_1D using a sinusoidal distribution."""
        x = np.linspace(0, width, points)
        return np.sin(np.pi * x / width)

    @staticmethod
    def jacobi(A):
        """Return the Jacobi preconditioner for matrix A."""
        diagonal = np.diagonal(A.toarray())
        return diags(1.0 / diagonal)

    def solve(self):
        laplacian = self._generate_laplacian()
        plate_flat = self.plate_config.flatten()
        b = -plate_flat
        solution = np.zeros_like(plate_flat)
        M_inv = self.jacobi(laplacian)
        M = LinearOperator((len(plate_flat), len(plate_flat)), matvec=M_inv.dot)
        solution, _ = cg(laplacian, b, x0=solution, tol=self.accuracy, M=M)
        return solution.reshape(self.mesh.y_grid.shape)

    def calculate_field(self):
        potential_distribution = self.solve()
        gradient = np.gradient(potential_distribution)
        E_y, E_x = -gradient[0], -gradient[1]
        return E_x, E_y
