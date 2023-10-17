import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from fipy import (
    CellVariable,
    Grid2D,
    Viewer,
    ImplicitSourceTerm,
    DiffusionTerm,
    LinearLUSolver,
)
import cv2
import io


def configure_plate(width, height, contacts):
    plate = np.zeros((height, width))
    for x, y, potential in contacts:
        plate[y + height // 2, x + width // 2] = potential
    return plate


def get_parameters_container():
    cols = st.columns(3)
    width = cols[0].slider("Width", 50, 200, 100)
    height = cols[1].slider("Height", 50, 200, 100)
    accuracy = cols[2].slider("Accuracy", 1e-4, 1e-2, 1e-3, step=1e-4)
    return width, height, accuracy


def get_contacts_container(height, width):
    st.header("Contacts")
    cols = st.columns(3)
    num_contacts = cols[0].number_input("Number of contacts", 1, 10, 1)
    contacts = []
    for i in range(num_contacts):
        st.subheader(f"Contact {i + 1}")
        cols = st.columns(3)
        x = cols[0].slider(f"X coordinate {i + 1}", -width // 2, (width // 2) - 1, 0)
        y = cols[1].slider(f"Y coordinate {i + 1}", -height // 2, (height // 2) - 1, 0)
        potential = cols[2].slider(f"Potential {i + 1}", 0.0, 2.0, 1.0)
        contacts.append((x, y, potential))
    return contacts


def solve_poisson(plate, resistances):
    """
    Solve the Poisson equation for the given plate and resistances by using FiPy.
    The input plate and resistances are 2D numpy arrays.
    Method:
    1. Create a 2D grid with the same size as the plate
    2. Create a CellVariable for the solution phi and the resistances D on the grid
    3. Constrain the solution phi to the values of the plate at the boundaries of the grid
    4. Solve the equation for phi using the ImplicitSourceTerm and DiffusionTerm
    5. Return the solution phi as a 2D numpy array
    Math for the Poisson equation:
    - The Laplace operator is given by the DiffusionTerm
    - The source term is given by the ImplicitSourceTerm
    - The equation to solve is: div(D * grad(phi)) + S = 0
    - D is the resistivity, S is the source term

    :param plate:
    :param resistances:
    :return:
    """
    # Create a 2D grid with the same size as the plate
    grid = Grid2D(
        dx=1.0, dy=1.0, nx=plate.shape[1], ny=plate.shape[0]
    )  # dx and dy are the grid spacings in x- and y-direction
    # Create a CellVariable for the solution phi and the resistances D on the grid
    phi = CellVariable(name="Electric Potential", mesh=grid, value=0.0)
    D = CellVariable(name="Resistances", mesh=grid, value=resistances)
    # Constrain the solution phi to the values of the plate at the boundaries of the grid
    phi.constrain(
        plate, where=grid.facesTop | grid.facesBottom | grid.facesLeft | grid.facesRight
    )
    # Solve the equation for phi using the ImplicitSourceTerm and DiffusionTerm
    eq = DiffusionTerm(coeff=D) - ImplicitSourceTerm(
        coeff=1.0
    )  # DiffusionTerm is the Laplace operator
    eq.solve(var=phi, solver=LinearLUSolver())
    # Return the solution phi as a 2D numpy array
    return phi.value


def calculate_field(phi):
    """
    Calculate the electric field from the electric potential phi.
    :param phi:
    :return:
    """
    # Calculate the electric field from the electric potential phi
    # The electric field is given by the negative gradient of the electric potential
    # The gradient is calculated by the grad() method of the CellVariable
    # The negative sign is added because the gradient points in the direction of the steepest descent
    E = -phi.grad


def plot_potential(plate, contacts, width, height):
    fig, ax = plt.subplots(figsize=(8, 7))
    y, x = np.mgrid[-height // 2 : height // 2, -width // 2 : width // 2]
    contour = ax.contourf(x, y, plate, cmap="hot")
    for contact in contacts:
        ax.plot(contact[0] + width // 2, contact[1] + height // 2, "wo")
    ax.set_xlim([-width // 2, width // 2])
    ax.set_ylim([-height // 2, height // 2])
    fig.colorbar(contour, ax=ax, label="Potential")
    ax.set_title("Electric Potential")
    st.pyplot(fig)


def plot_electric_field(plate, contacts, width, height):
    E_x, E_y = calculate_field(plate)
    fig, ax = plt.subplots(figsize=(8, 7))
    y, x = np.mgrid[-height // 2 : height // 2, -width // 2 : width // 2]
    ax.quiver(x, y, E_x, E_y, scale=5)
    for contact in contacts:
        ax.plot(contact[0] + width // 2, contact[1] + height // 2, "wo")
    ax.set_xlim([-width // 2, width // 2])
    ax.set_ylim([-height // 2, height // 2])
    ax.set_title("Electric Field")
    st.pyplot(fig)


def main():
    st.title("Electric Potential and Field in a Conductive Plate")
    width, height, accuracy = get_parameters_container()
    contacts = get_contacts_container(height, width)
    plate = configure_plate(width, height, contacts)
    plot_potential(plate, contacts, width, height)
    plot_electric_field(plate, contacts, width, height)


main()
