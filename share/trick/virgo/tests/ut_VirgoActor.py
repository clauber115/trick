import os, sys, shutil, inspect
import unittest
import pdb

# Add path to virgo module
thisFileDir = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)))
virgo_dir=os.path.abspath(os.path.join(thisFileDir, '../'))
sys.path.append(virgo_dir)
from Virgo import *
meshes_dir=os.path.join(virgo_dir, 'meshes')
tests_dir=os.path.join(virgo_dir, 'tests')

def suite():
    """Create test suite from test cases here and return"""
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(VirgoDataPlaybackActorTestCase))
    return (suites)

class VisualizableTestCase(unittest.TestCase):
    """
    A base class for VTK unit tests that supports optional visualization for debugging.
    """

    def get_origin_axes(self):
      origin_axes = vtk.vtkAxesActor()
      origin_axes.SetTotalLength(5, 5, 5)  # Size of axes (x, y, z lengths)
      origin_axes.SetShaftTypeToCylinder()  # Cylindrical shafts for visibility
      origin_axes.SetAxisLabels(True)  # Show x, y, z labels
      return(origin_axes)
    
    def visualize_scene(self, actors, axis_length=5):
        """
        Optionally visualize a list of actors in a render window.
        :param actors: A single vtkActor or a list of vtkActors to visualize.

        Set the environment variable VIRGO_VISUALIZE_TESTS=1 to enable
        visualization during test runs for debugging.  When visualization is
        enabled, a render window will pop up and block until closed.
        """
        self.visualize = os.environ.get('VIRGO_VISUALIZE_TESTS', '0') == '1'
        if not self.visualize:
            return
        print(f"Visualizing. Exit window to continue.")
        
        if not isinstance(actors, list):
            actors = [actors]
        
        # Set up the scene
        renderer = vtk.vtkRenderer()
        for actor in actors:
            renderer.AddActor(actor)
        renderer.AddActor2D(self.get_origin_axes())
        
        # Create render window and interactor
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)  # Optional: Set window size
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.SetRenderWindow(render_window)
        
        # Render and start interaction (blocks until window is closed)
        render_window.Render()
        interactor.Start()

    def save_scene_to_image(self, actors, filename="scene.png"):
        """
        Save the scene with the given actors to an image file without displaying a window.
        :param actors: A single vtkActor or a list of vtkActors to render.
        :param filename: Name of the output image file (e.g., 'scene.png').
        """
        self.save_images = os.environ.get('VIRGO_WRITE_TEST_IMAGES', '0') == '1'
        if not self.save_images:
            return
        
        if not isinstance(actors, list):
            actors = [actors]
        
        # Set up the scene
        renderer = vtk.vtkRenderer()
        for actor in actors:
            renderer.AddActor(actor)
        renderer.AddActor2D(self.get_origin_axes())
        
        # Create render window with off-screen rendering
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)
        render_window.SetOffScreenRendering(1)  # Enable off-screen rendering
        
        # Render the scene
        render_window.Render()
        
        # Capture the image
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.SetInputBufferTypeToRGBA()  # Use RGBA for better quality
        window_to_image.ReadFrontBufferOff()  # Read from back buffer
        window_to_image.Update()
        
        # Write to PNG file
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(os.path.join(tests_dir, filename))
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()

    def vis(self):
        """
        Easy-to-call function that will save scenes to image or show them in an
        interactive window depending on what the user has requested. Does nothing
        if neither of those features is enabled by the user
        """
        self.save_scene_to_image(self.instance, filename=f".{self.__class__.__name__}_{self._testMethodName}.png")
        self.visualize_scene(self.instance)

class VirgoDataPlaybackActorTestCase(VisualizableTestCase):

    def setUp(self):
        VisualizableTestCase().setUp()
        # Nominal no-error when parsing the trick-sims config file scenario
        self.instance = VirgoDataPlaybackActor(mesh=os.path.join(meshes_dir, 'teapot.obj'))


    def test_init_nominal(self):
        #import pdb; pdb.set_trace()
        self.assertEqual(self.instance.name, 'No Name')
        self.assertEqual(self.instance.init_pyr, None)

        self.assertIsNone(self.instance.get_current_position())
        #self.assertAlmostEqual(self.instance.get_current_position(), [0.0, 0.0, 0.0])
        self.vis()