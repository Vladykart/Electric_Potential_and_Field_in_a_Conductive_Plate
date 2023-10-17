from electrochemical_etching_process_simulation.EEP import *
import numpy as np
import unittest
import logging

logging.basicConfig(level=logging.DEBUG)

class TestLayer(unittest.TestCase):
    logging.info("Testing Layer class")

    def test_update_thickness(self):
        logging.info("Testing update_thickness method")
        layer = Layer(10, 5, 1)
        layer.update_thickness(1, 1)
        self.assertEqual(layer.thickness.tolist(), [4] * 10)
        logging.info("Passed update_thickness method test")
        logging.info('Result: ' + str(layer.thickness.tolist()))


class TestMesh(unittest.TestCase):
    logging.info("Testing Mesh class")
    def test_refine_region(self):
        logging.info("Testing refine_region method")
        mesh = Mesh(10, 10, 1, 1)
        logging.info("Before refinement: " + str(mesh.x_grid[0, :5].tolist()))
        mesh.refine_region(0, 5, 0, 5, 2)
        logging.info("After refinement: " + str(mesh.x_grid[0, :5].tolist()))
        self.assertEqual(mesh.x_grid[0, :5].tolist(), [0, 0, 1, 1, 2])
        logging.info("Passed refine_region method test")
        logging.info("Before refinement: " + str(mesh.y_grid[:5, 0].tolist()))
        logging.info("After refinement: " + str(mesh.y_grid[:5, 0].tolist()))
        self.assertEqual(mesh.y_grid[:5, 0].tolist(), [0, 0, 1, 1, 2])
        logging.info("Passed refine_region method test")

        logging.info("Result: " + str(mesh.x_grid[0, :5].tolist()))



class TestElectrolyte(unittest.TestCase):
    def test_adjusted_conductivity(self):
        electrolyte = Electrolyte(1, 2)
        self.assertEqual(electrolyte.adjusted_conductivity, 2)


class TestSimulationDomain(unittest.TestCase):
    def test_compute_electric_field(self):
        J_1D = np.ones(int(5 / 0.01))  # Define J_1D here
        domain = SimulationDomain(5, 4, 0.01, J_1D)  # Pass J_1D to SimulationDomain
        domain.compute_electric_field(len(J_1D))
        self.assertEqual(domain.electric_field.tolist(), [1.0] * 500)


class TestEtchingSimulation(unittest.TestCase):
    def test_run(self):
        J_1D = np.ones(int(5 / 0.01))  # Define J_1D here
        domain = SimulationDomain(5, 4, 0.01, J_1D)  # Pass J_1D to SimulationDomain
        substrate = Layer(5, 5, 0.01)
        simulation = EtchingSimulation(
            domain, substrate, 0.0001, J_1D
        )  # Pass J_1D to EtchingSimulation
        history = simulation.run(time_steps=1, dt=1)
        self.assertEqual(history[0].tolist(), [4.9999] * 500)


if __name__ == "__main__":
    unittest.main()
