"""
Fit circle to set of points. 
Find radius of curvature. 
Find planitude and alatude with respect to another point.
"""
import sys
import numpy as np
from scipy.optimize import minimize
import regions as rg
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve_fft
from matplotlib import pyplot as plt
import matplotlib.patches
import seaborn as sns
sns.set_style('white')

def read_arc_data_ds9(filename):
    """
    Return the sky coordinates of a star (single point of type
    'circle') and arc (multiple points of type: 'x'), which are read
    from the DS9 region file `filename`
    """
    regions = rg.read_ds9(filename)

    try:
        star, = [x for x in regions if x.visual['point'] == 'circle']
    except IndexError:
        sys.exit("One and only one 'circle' region is required")
    points = [x for x in regions if x.visual['point'] == 'x']
    return star, points


def get_arc_xy(region_filename, fits_filename):
    """
    Return pixel coordinates for arc points and star point, which
    are read as sky coordinates from `region_filename` in DS9 format
    (the arc points must have the 'x' shape and the star point must
    have the 'circle' shape).  The WCS transformation is read from the
    header of `fits_filename`.

    Returns `xs`, `ys`, `x`, `y`
    """
    # Find the arc and star sky coordinates 
    star, points = read_arc_data_ds9(region_filename)
    # Find WCS transformation from FITS image header
    hdu, = fits.open(fits_filename)
    w = WCS(hdu.header)
    # Convert to pixel coordinates
    xs, ys = SkyCoord(star.center).to_pixel(w)
    x, y = SkyCoord([point.center for point in points]).to_pixel(w)
    # Return xs, ys as scalar floats and x, y as 1-d arrays of floats
    return xs[0], ys[0], x, y
    

def mean_radius(x, y, xc, yc):
    """
    x, y are vectors of data points
    xc, yc is the center of curvature

    Returns mean distance of (x, y) from (xc, yc)

    Implements equation (E1) of TYH18
    """
    return np.mean(np.hypot(x - xc, y - yc))

def square_deviation(x, y, xc, yc):
    """
    Total square deviation of points from mean distance

    Implements equation (E2) of TYH18
    """
    rm = mean_radius(x, y, xc, yc)
    return np.sum((np.hypot(x - xc, y - yc) - rm)**2)

def objective_f(center, xdata, ydata):
    """Function to minimize"""
    return square_deviation(xdata, ydata, center[0], center[1])
    
def fit_circle_to_xy(x, y):
    # guess the starting values
    soln0 = np.array((np.mean(x), np.mean(y)))
    # Do the fitting
    soln = minimize(objective_f, soln0, args=(x, y))
    return soln


def axis_unit_vector(r0, rc):
    """
    Find the axis unit vector from the center of curvature at `rc` to
    the star at `r0`, where `rc` and `r0` are both length-2 xy vectors
    (can be any sequence, auto-converted to numpy array)

    Implements equation (E3) of TYH18
    """
    assert len(r0) == len(rc) == 2
    uvec = np.array(r0) - np.array(rc)
    uvec /= np.hypot(*uvec)
    return uvec


def apex_distance(r0, rc, Rc, uvec):
    """
    Implements equation (E4) of TYH18
    """
    R0 = rc + Rc*uvec - r0
    return np.hypot(*R0)


def find_theta(x, y, x0, y0, uvec):
    """
    Find angle in degrees of all points (`x`, `y`) from the axis
    `uvec`, measured around the point (`x0`, `y0`)

    According to need, (x0, y0) can either be the star or the center
    of curvature,
    """
    xy = np.stack((x - x0, y - y0), axis=-1)
    R_cos_theta = np.dot(xy, uvec)
    R_sin_theta = np.cross(xy, uvec)
    theta = np.arctan2(R_sin_theta, R_cos_theta)
    return np.degrees(theta)


class FittedCircle(object):
    def __init__(self, x, y, xs, ys, mask=None, verbose=False):
        self.x = x
        self.y = y
        self.xs = xs
        self.ys = ys
        self.verbose = verbose
        if mask is None:
            # Use all the x, y points
            self.mask = np.ones_like(x).astype(bool)
        else:
            # Restrict to certain x, y points
            self.mask = mask
        self.results = fit_circle_to_xy(self.x[self.mask], self.y[self.mask])
        self.r0 = self.xs, self.ys
        self.rc = self.results.x
        self.Rc = mean_radius(self.x[self.mask], self.y[self.mask], *self.results.x)
        self.xihat = axis_unit_vector(self.r0, self.rc)
        self.R0 = apex_distance(self.r0, self.rc, self.Rc, self.xihat)
        # Calculate the theta values for all points, regardless of the mask
        self.theta = find_theta(self.x, self.y, self.xs, self.ys, self.xihat)
        self.theta_c = find_theta(self.x, self.y, self.rc[0], self.rc[1], self.xihat)
        # Find the R(theta)
        self.R = np.hypot(self.x - self.xs, self.y - self.ys)
        # Find sort order of theta increasing
        order = np.argsort(self.theta)
        self.R90 = np.interp([-90.0, 90.0], self.theta[order], self.R[order])
        if self.verbose:
            print(self.results.message)
            print("  Apex distance:", self.R0)
            print("  Radius of curvature:", self.Rc)
            print("  Perpendicular radius (+/-):", self.R90)
        


def plot_solution(region_filename, fits_filename, plotfile, verbose=True):
    # Find WCS transformation from FITS image header
    hdu, = fits.open(fits_filename)
    w = WCS(hdu.header)
    # Pot the image data from the FITS file
    fig, ax = plt.subplots(subplot_kw=dict(projection=w))
    ax.imshow(hdu.data, origin='lower', vmin=2.8, vmax=3.5, cmap='viridis')

    xs, ys, x, y = get_arc_xy(region_filename, fits_filename)
    cc = [FittedCircle(x, y, xs, ys, verbose=True)]

    # Size of viewport
    size = 150
    x1, x2 = xs - size, xs + size
    y1, y2 = ys - size, ys + size
    
    # Contour of a smoothed version
    ax.contour(
        convolve_fft(hdu.data, Gaussian2DKernel(stddev=2)),
        levels=np.linspace(2.8, 3.5, 15),
        linewidths=0.5)


    for iter in range(5):
        m = np.abs(cc[-1].theta_c) <= 30.0
        cc.append(FittedCircle(x, y, xs, ys, mask=m, verbose=True))

    ax.scatter(x[m], y[m], s=30, color='r', zorder=2)
    ax.scatter(x[~m], y[~m], s=15, color='w', zorder=2)

    colors = sns.color_palette("Oranges_r", n_colors=len(cc))
    for c, color in zip(cc, colors):
        ax.add_patch(
            matplotlib.patches.Circle(c.rc, radius=c.Rc, ec=color, fc='none'))
        ax.plot(
            [c.rc[0], c.rc[0] + 1.2*c.Rc*c.xihat[0]],
            [c.rc[1], c.rc[1] + 1.2*c.Rc*c.xihat[1]],
            ls="--", color=color,
        )
        ax.scatter(c.rc[0], c.rc[1], s=30, color=color)

    ax.scatter(xs, ys, s=30, color='k', zorder=2)

    ra, dec = ax.coords
    ra.set_major_formatter('hh:mm:ss.ss')
    dec.set_major_formatter('dd:mm:ss.s')
    ra.set_axislabel('RA (J2000)')
    dec.set_axislabel('Dec (J2000)')
    
    ax.set(
        xlim=[x1, x2],
        ylim=[y1, y2],
    )
    fig.savefig(plotfile)
    return plotfile
    

TESTDATA = np.array([1, 2, 3, 4]), np.array([1, 2, 2, 1])
TESTCENTER = np.array([2.5, 0.5])

TEST_REGION_FILE = "data/new-w000-400-ridge.reg"
TEST_FITS_FILE = "data/w000-400-Bally_09-extract.fits"
TEST_PLOT_FILE = 'plot-w000-400-ridge.pdf'

if __name__ == "__main__":

    try:
        arc = str(sys.argv[1])
        TEST_REGION_FILE = TEST_REGION_FILE.replace("ridge", arc)
        TEST_PLOT_FILE = TEST_PLOT_FILE.replace("ridge", arc)
    except:
        pass
    
    # Test with simple points
    print("### Simple Test")
    results = fit_circle_to_xy(*TESTDATA)
    assert np.allclose(results.x, TESTCENTER)
    print(results)

    # Test with real image and region file
    print("### Image Test")
    print("Figure file:",
          plot_solution(TEST_REGION_FILE, TEST_FITS_FILE, TEST_PLOT_FILE))
