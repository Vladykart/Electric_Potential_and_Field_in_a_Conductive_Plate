import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import inv, LinearOperator, cg


def set_page_config():
    st.set_page_config(
        page_title="Electric Potential and Field in a Conductive Plate",
        page_icon="🔌",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def title_and_header():
    st.title("Electric Potential and Field in a Conductive Plate")
    st.sidebar.header("Parameters")


def get_parameters():
    width = st.sidebar.slider("Width", 50, 200, 100)
    height = st.sidebar.slider("Height", 50, 200, 100)
    accuracy = st.sidebar.slider("Accuracy", 1e-4, 1e-2, 1e-3, step=1e-4)  # Lowered accuracy
    return width, height, accuracy


def configure_plate(width, height, contacts):
    plate = np.zeros((height, width))
    for x, y, potential in contacts:
        plate[y+height//2, x+width//2] = potential  # add offsets to x and y
    return plate


def get_contacts(height, width):
    st.sidebar.header("Contacts")
    num_contacts = st.sidebar.number_input("Number of contacts", 1, 10, 1)
    contacts = []
    for i in range(num_contacts):
        st.sidebar.subheader(f"Contact {i + 1}")
        x = st.sidebar.slider(f"X coordinate {i + 1}", -width//2, (width//2)-1, 0)
        y = st.sidebar.slider(f"Y coordinate {i + 1}", -height//2, (height//2)-1, 0)
        potential = st.sidebar.slider(f"Potential {i + 1}", 0.0, 2.0, 1.0)
        contacts.append((x, y, potential))
    return contacts


def jacobi(A):
    d = A.diagonal()
    D = diags(d)
    M_inv = inv(D)
    return M_inv


def solve_laplace(plate, accuracy):
    height, width = plate.shape
    size = height * width
    main_diag = np.ones(size) * -4
    off_diags = np.ones(size - 1)
    diagonals = [main_diag, off_diags, off_diags, off_diags, off_diags]
    A = diags(diagonals, [0, -1, 1, -width, width]).tocsr()

    # Prepare the RHS vector
    plate_flat = plate.flatten()
    b = -plate_flat

    # Enforce boundary conditions
    b[:width] = 0
    b[-width:] = 0
    b[::width] = 0
    b[width - 1 :: width] = 0

    # Initial guess for the solution
    solution = np.zeros_like(b)

    # Define preconditioner
    M_inv = jacobi(A)
    M = LinearOperator((size, size), matvec=M_inv.dot)

    while True:
        # Solve the system of linear equations
        new_solution, _ = cg(A, b + A.dot(solution), x0=solution, tol=accuracy, M=M)
        diff = np.linalg.norm(new_solution - solution)
        if diff < accuracy:
            break
        solution = new_solution

    return solution.reshape((height, width))


def calculate_field(plate):
    gradient = np.gradient(plate)
    E_y, E_x = -gradient[0], -gradient[1]
    return E_x, E_y


def plot_results(plate, x, y, E_x, E_y, contacts, width, height):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    contour = ax[0].contourf(x, y, plate, cmap="hot")
    for contact in contacts:
        ax[0].plot(contact[0] + width // 2, contact[1] + height // 2, "wo")
    ax[0].set_xlim([-width // 2, width // 2])
    ax[0].set_ylim([-height // 2, height // 2])
    fig.colorbar(contour, ax=ax[0], label="Potential")
    ax[0].set_title("Electric Potential")

    ax[1].quiver(x, y, E_x, E_y, scale=5)
    for contact in contacts:
        ax[1].plot(contact[0] + width // 2, contact[1] + height // 2, "wo")
    ax[1].set_xlim([-width // 2, width // 2])
    ax[1].set_ylim([-height // 2, height // 2])
    ax[1].set_title("Electric Field")

    st.pyplot(fig)


def main():
    set_page_config()
    title_and_header()
    width, height, accuracy = get_parameters()
    contacts = get_contacts(height, width)
    plate = configure_plate(width, height, contacts)
    st.info("Solving Laplace Equation ...")
    plate = solve_laplace(plate, accuracy)
    E_x, E_y = calculate_field(plate)
    y, x = np.mgrid[-height // 2 : height // 2, -width // 2 : width // 2]
    plot_results(plate, x, y, E_x, E_y, contacts, width, height)  # pass width and height as arguments


if __name__ == "__main__":
    main()
