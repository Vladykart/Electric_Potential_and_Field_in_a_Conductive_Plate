import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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



def configure_plate(width, height, contacts):
    plate = np.zeros((height, width))
    for x, y, potential in contacts:
        plate[y + height // 2, x + width // 2] = potential  # add offsets to x and y
    return plate


def get_contacts(height, width):
    st.sidebar.header("Contacts")
    num_contacts = st.sidebar.number_input("Number of contacts", 1, 10, 1)
    contacts = []
    for i in range(num_contacts):
        st.sidebar.subheader(f"Contact {i + 1}")
        x = st.sidebar.slider(f"X coordinate {i + 1}", -width // 2, (width // 2) - 1, 0)
        y = st.sidebar.slider(
            f"Y coordinate {i + 1}", -height // 2, (height // 2) - 1, 0
        )
        potential = st.sidebar.slider(f"Potential {i + 1}", 0.0, 2.0, 1.0)
        contacts.append((x, y, potential))
    return contacts


def jacobi(A):
    d = A.diagonal()
    D = diags(d).tocsc()  # convert D to CSC format
    M_inv = inv(D)
    return M_inv



def solve_poisson(plate, resistance, accuracy):
    height, width = plate.shape
    size = height * width
    conductivity = 1 / resistance  # calculate the conductivity
    main_diag = np.ones(size) * -4 * conductivity
    off_diags = np.ones(size - 1) * conductivity
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


def plot_results(plate, x, y, E_x, E_y, contacts, width, height, title=''):
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))

    # Left panel: Electric Potential Contour
    contour = ax[0].contourf(x, y, plate, cmap="hot", norm=mpl.colors.LogNorm())  # Use LogNorm for normalization
    for contact in contacts:
        ax[0].plot(contact[0] + width // 2, contact[1] + height // 2, "wo")
    ax[0].set_xlim([-width // 2, width // 2])
    ax[0].set_ylim([-height // 2, height // 2])
    ax[0].set_title(f"Electric Potential ({title})")

    # Right panel: Electric Field Quiver Plot
    ax[1].quiver(x, y, E_x, E_y, scale=5)
    for contact in contacts:
        ax[1].plot(contact[0] + width // 2, contact[1] + height // 2, "wo")
    ax[1].set_xlim([-width // 2, width // 2])
    ax[1].set_ylim([-height // 2, height // 2])
    ax[1].set_title(f"Electric Field ({title})")

    # Add colorbar to the left panel for electric potential
    cbar = fig.colorbar(contour, ax=ax[0], label="Potential (log scale)")

    # Display the figure using Streamlit
    st.pyplot(fig)


def get_parameters_container():
    cols = st.columns(3)

    width = cols[0].slider(
        "Width", 50, 200, 100, help="Width of the conductive plate in arbitrary units"
    )
    height = cols[1].slider(
        "Height", 50, 200, 100, help="Height of the conductive plate in arbitrary units"
    )
    accuracy = cols[2].slider(
        "Accuracy", 1e-4, 1e-2, 1e-3, step=1e-4, help="Accuracy for the solver in CG /n 1e-4, 1e-2, 1e-3, step=1e-4"
    )
    resistance = cols[0].slider(
        "Resistance", 0.1, 10.0, 1.0
    )  # New resistance parameter
    return width, height, accuracy, resistance


def get_contacts_container(height, width):
    st.header("Contacts")
    cols = st.columns(3)
    num_contacts = cols[0].number_input("Number of contacts", 1, 10, 1)
    empty_col = cols[1].empty()
    empty_col = cols[2].empty()

    contacts = []
    for i in range(num_contacts):
        st.subheader(f"Contact {i + 1}")

        cols = st.columns(3)
        x = cols[0].slider(
            f"X coordinate {i + 1}",
            -width // 2,
            (width // 2) - 1,
            0,
            help="X-coordinate of the contact",
        )
        y = cols[1].slider(
            f"Y coordinate {i + 1}",
            -height // 2,
            (height // 2) - 1,
            0,
            help="Y-coordinate of the contact",
        )
        potential = cols[2].slider(
            f"Potential {i + 1}", 0.0, 2.0, 1.0, help="Potential at the contact"
        )

        # Add error handling for contact positions and potential
        if not -width // 2 <= x < width // 2 or not -height // 2 <= y < height // 2:
            st.error("Error: Contact position must be within the plate area.")
            return
        if not 0.0 <= potential <= 2.0:
            st.error("Error: Potential must be between 0.0 and 2.0.")
            return

        contacts.append((x, y, potential))

    return contacts


def main():
    st.set_page_config(
        page_title="Electric Potential and Field in a Conductive Plate",
        page_icon="🔌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Electric Potential and Field in a Conductive Plate")
    width, height, accuracy, resistance = get_parameters_container()
    contacts = get_contacts_container(height, width)
    plate = configure_plate(width, height, contacts)

    if st.button("Generate"):
        st.info("Solving Equation ...")
        resistances = np.linspace(1, 0.1, num=10)  # or whatever range you're interested in
        for resistance in resistances:
            conductivity = 1 / resistance
            plate = solve_poisson(plate, conductivity, accuracy)
            E_x, E_y = calculate_field(plate)
            y, x = np.mgrid[-height // 2: height // 2, -width // 2: width // 2]
            plot_results(plate, x, y, E_x, E_y, contacts, width, height, title=f'Resistance = {resistance}')


if __name__ == "__main__":
    main()
