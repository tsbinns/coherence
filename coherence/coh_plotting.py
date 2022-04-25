"""Classes for plotting results.

CLASSES
-------
Plotting : Abstract Base Class
-   Abstract class for plotting results.

LinePlot : subclass of Plotting
-   Class for plotting results on line plots.

BoxPlot : subclass of Plotting
-   Class for plotting results on box plots.

SurfacePlot : subclass of Plotting
-   Class for plotting results on cortical surfaces.
"""

from abc import ABC

class Plotting(ABC):