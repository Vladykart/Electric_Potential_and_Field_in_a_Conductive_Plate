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


def image_to_resistance_mask(uploaded_file, width, height):
    image = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE) / 255.0  # normalize between 0 and 1
    resized_img = cv2.resize(
        img, (height, width)
    )  # make sure it has the same size as the plate
    return resized_img


def solve_poisson(plate, resistances):
    ny, nx = plate.shape
    dx = 1.0  # grid spacing
    mesh = Grid2D(dx=dx, dy=dx, nx=nx, ny=ny)
    phi = CellVariable(name="solution variable", mesh=mesh, value=0.0)
    X, Y = mesh.faceCenters
    phi.constrain(
        plate.T.flatten(),
        mesh.facesLeft | mesh.facesRight | mesh.facesTop | mesh.facesBottom,
    )
    D = CellVariable(name="resistivity", mesh=mesh, value=resistances.T.flatten())
    eq = ImplicitSourceTerm(coeff=1.0) + DiffusionTerm(coeff=D)
    eq.solve(var=phi, solver=LinearLUSolver(), dt=0.01)
    return phi.value.reshape((ny, nx))


def plot_results(plate, contacts, width, height):
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


def generate_noise_image(height, width):
    # Generate a random noise image
    noise_img = np.random.normal(size=(height, width), loc=0.5, scale=0.1)
    noise_img = np.clip(noise_img, 0, 1)  # Ensure values are between 0 and 1
    return noise_img


def adapt_mask(data, mask):
    # Get the dimensions of the data
    data_dim = data.shape

    # Adapt the mask dimensions to match data
    new_mask = np.resize(mask, data_dim)

    return new_mask


def main():
    st.title("Electric Potential and Field in a Conductive Plate")
    width, height, accuracy = get_parameters_container()
    contacts = get_contacts_container(height, width)
    plate = configure_plate(width, height, contacts)
    uploaded_file = st.file_uploader(
        "Choose an image for the resistance mask", type=["jpg", "png"]
    )
    use_noise = st.checkbox("Use noise", value=False)
    resistance_mask = None

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        resistance_mask = 1 - image_to_resistance_mask(uploaded_file, width, height)  # invert the greyscale
        st.write("Image processed for resistance mask.")

    elif use_noise:
        st.info("No image uploaded, generating noise image.")
        noise_image = generate_noise_image(height, width)
        st.image(noise_image, caption="Generated noise image", use_column_width=True)
        resistance_mask = 1 - noise_image  # invert the greyscale
        adaptive_mask = adapt_mask(plate, resistance_mask)
        st.write("Image processed for resistance mask.")