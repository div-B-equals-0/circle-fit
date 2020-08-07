"""
Fit circle to set of points.
Find radius of curvature.
Find planitude and alatude with respect to another point.
"""
import sys
import json
import os
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

# Module-level variables 
PT_STAR = "circle"
PT_ARC = "x"


def is_pt_region(x):
    """
    True if region is a point-region (rather than a circle, line, box, etc)
    """
    return isinstance(x, rg.shapes.point.PointSkyRegion)


def read_arc_data_ds9(filename):
    """
    Return the sky coordinates of a star (single point of type
    `pt_star`) and arc (multiple points of type: `pt_arc`), which are
    read from the DS9 region file `filename`
    """
    regions = rg.read_ds9(filename)
    # Eliminate regions that are not points since they will cause errors
    regions = list(filter(is_pt_region, regions))
   
    points = [x for x in regions if x.visual['point'] == PT_ARC]
    stars = [x for x in regions if x.visual['point'] == PT_STAR]
    assert len(stars) > 0, f"At least one '{PT_STAR}' region is required"
    star = stars[0]
    if len(stars) > 1:
        print(f"WARNING: multiple '{PT_STAR}' regions found in {filename} - using first one")
    return star, points


def resample_with_partial_replacement(a, fraction):
    """
    Like `numpy.random.choice`, but intermediate between
    `replace=True` and `replace=False`, according to the value of
    `fraction`.

    If `fraction=1.0`, then it is equivalent to
    `numpy.random.choice(a, size=len(a), replace=True)`.  If
    `fraction=0.0`, then it is equivalent to `numpy.random.choice(a,
    size=len(a), replace=False)`, which just gives you back `a` but
    with the elements in a different order.  For intermediate values,
    return a fraction `1 - fraction` of the elements re-sampled
    WITHOUT replacement, and the remainder re-sampled WITH replacement.
    """
    assert 0.0 <= fraction <= 1.0
    n = len(a)
    k = int(fraction*n)
    withouts = np.random.choice(a, n-k, replace=False)
    remainder = list(set(a) - set(withouts))
    # remainder = [_ for _ in a if not _ in withouts]
    assert len(remainder) == k
    if k > 0:
        withs = np.random.choice(remainder, k, replace=True)
    else:
        withs = []
    return np.concatenate((withouts, withs))


def get_primary_hdu(filename):
    hdulist = fits.open(filename)
    try:
        hdu = hdulist["sci"]
    except:
        hdu = hdulist[0]
    return hdu


def get_arc_xy(region_filename, fits_filename, wcs=None,
               resample=False, resample_fraction=0.5):
    """
    Return pixel coordinates for arc points and star point, which are
    read as sky coordinates from `region_filename` in DS9 format (the
    arc points must have the 'x' shape and the star point must have
    the 'circle' shape).  The WCS transformation is read from the
    header of `fits_filename`, or can be provide directly as `wcs` (in
    which case, `fits_filename` is ignored).

    Returns `xs`, `ys`, `x`, `y`

    If optional argument `resample` is True (default: False), then
    resample the arc points with replacement to give a list of the
    same length (but with repetitions).  This can be used for
    bootstrapping
    """
    # Find the arc and star sky coordinates
    star, points = read_arc_data_ds9(region_filename)
    assert(len(points) > 0), "No points found in arc"
    if resample:
        # Resampling is for bootstrap estimation of uncertainties.
        # Repeat the process 50 times with resample=True to get a good
        # idea of the spread
        points = resample_with_partial_replacement(
            points, fraction=resample_fraction)

    # Find WCS transformation from FITS image header
    if wcs is None:
        hdu = get_primary_hdu(fits_filename)
        wcs = WCS(hdu.header)
    # Convert to pixel coordinates
    xs, ys = SkyCoord(star.center).to_pixel(wcs)
    x, y = SkyCoord([point.center for point in points]).to_pixel(wcs)
    # Return xs, ys as scalar floats and x, y as 1-d arrays of floats
    try:
        # Original version worked in 2018 version of SkyCoord
        return xs[0], ys[0], x, y
    except IndexError:
        # 2020 version of SkyCoord now returns 0-dimensional arrays for scalar case
        return float(xs), float(ys), x, y

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


def fit_circle_to_xy(x, y, soln0=None):
    # guess the starting values if not provided
    if soln0 is None:
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
    assert np.hypot(*uvec) != 0.0, (r0, rc)
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
    Find angle in degrees of all points (`x`, `y`) from the axis unit
    vector `uvec`, measured around the point (`x0`, `y0`)

    According to need, (x0, y0) can either be the star or the center
    of curvature,
    """

    assert len(x) == len(y)
    assert len(uvec) == 2
    assert np.isclose(np.hypot(*uvec), 1.0), (x, y, x0, y0, uvec)
    # Rvec is array of radius vectors from (x0, y0) to each (x, y)
    Rvec = np.stack((x - x0, y - y0), axis=-1)
    R_cos_theta = np.dot(Rvec, uvec)
    R_sin_theta = np.cross(Rvec, uvec)
    theta = np.arctan2(R_sin_theta, R_cos_theta)
    return np.degrees(theta)


class FittedCircle(object):
    """
    A single circle fitted to a set of points, with an optional mask
    that specifies which points to include in the fit
    """
    def __init__(self, x, y, xs, ys, mask=None, verbose=False):
        self.x = x
        self.y = y
        self.xs = xs
        self.ys = ys
        self.r0 = self.xs, self.ys
        self.verbose = verbose
        if mask is None:
            # Use all the x, y points
            self.mask = np.ones_like(x).astype(bool)
        else:
            # Restrict to certain x, y points
            self.mask = mask
        # Initial guess for rc is source position r0
        self.results = fit_circle_to_xy(self.x[self.mask], self.y[self.mask],
                                        soln0=self.r0)
        self.rc = self.results.x
        self.Rc = mean_radius(self.x[self.mask], self.y[self.mask], *self.results.x)
        # pdb.set_trace()
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
        self.Pi = self.Rc/self.R0
        self.Lambda_m, self.Lambda_p = self.R90/self.R0
        self.Lambda = 0.5*(self.Lambda_p + self.Lambda_m)
        self.dLambda = 0.5*(self.Lambda_p - self.Lambda_m)
        # Angle of axis (but remember, this is in pixel coordinates)
        self.angle = np.rad2deg(np.arctan2(*self.xihat))
        if self.verbose:
            print(self.results.message)
            print("  Apex distance:", self.R0)
            print("  Radius of curvature:", self.Rc)
            print("  Perpendicular radius (+/-):", self.R90)

    def __str__(self):
        return f"CircleFit(Pi = {self.Pi:.3f}, Lambda = {self.Lambda:.3f}, dLambda = {self.dLambda:.3f})"
        

class IteratedFit(object):
    """
    A sequence of `FittedCircle`s, where each after the first has its
    mask set to those points lying within `delta_theta` of the axis of
    the previous fit
    """
    def __init__(self, x, y, xs, ys, delta_theta=75.0, maxiter=3, verbose=False):
        # First circle is fitted to all the points
        self.circles = [FittedCircle(x, y, xs, ys, verbose=verbose)]
        self.masks = [np.ones_like(x).astype(bool)]
        # Iterate to improve the fit
        for it in range(maxiter):
            # The mask m selects for the fit only those points that
            # lie within delta_theta of previous axis
            m = np.abs(self.circles[-1].theta) <= delta_theta
            self.circles.append(FittedCircle(x, y, xs, ys, mask=m, verbose=verbose))
            self.masks.append(m)


class ShapeDistributions(object):
    """Container for shapes of bootstrap-resampled fits"""
    def __init__(self, bootstraps):
        self.Lambda = np.array([b.Lambda for b in bootstraps])
        self.dLambda = np.array([b.dLambda for b in bootstraps])
        self.Pi = np.array([b.Pi for b in bootstraps])
        # Note that R0 is in pixels
        # TODO: use WCS to convert to arcsec
        self.R0 = np.array([b.R0 for b in bootstraps])
        # TODO: also find position angle of axis in world coordinates
        self.angle = np.array([b.angle for b in bootstraps])
        # Stack of all the arrays that can be used
        self.data = [self.Pi, self.Lambda, self.dLambda, self.R0, self.angle]
        self.corr = np.corrcoef(self.data)


class FitWithErrors(object):
    """
    An iterated circle fit with errors calculated by bootstrap resampling
    """
    def __init__(self, region_filename, fits_filename,
                 delta_theta=75, nbootstrap=50, fraction=0.5,
                 verbose=False,
    ):
        wcs = WCS(fits.open(fits_filename)[0].header)
        xs, ys, x, y = get_arc_xy(region_filename, None, wcs=wcs,
                                  resample=False)
        if verbose:
            print(f"#### Full dataset")
            print(f"Star: {xs:.1f} {ys:.1f}")
            print(x)
            print(y)
           
        fit = IteratedFit(x, y, xs, ys, delta_theta, verbose=verbose)
        self.shape = fit.circles[-1]
        # bootstraps is a list of FittedCircle() instances
        self.bootstraps = []
        for _ in range(nbootstrap):
            xs, ys, x, y = get_arc_xy(region_filename, None, wcs=wcs,
                                      resample=True, resample_fraction=fraction)
            if verbose:
                print(f"#### Bootstrap #{_}")
                print(x)
                print(y)

            fit = IteratedFit(x, y, xs, ys, delta_theta, verbose=verbose)
            self.bootstraps.append(fit.circles[-1])
        self.shape_dist = ShapeDistributions(self.bootstraps)


def plot_solution(
        region_filename, fits_filename, plotfile, delta_theta,
        vmin=None, vmax=None, sigma=2.0,
        resample=False, resample_fraction=0.5,
        verbose=False, maxiter=3, remove_sip_kwds_from_header=True,
):
    """
    Iteratively fit circle to bow and plot the result.
    """
    # Find WCS transformation from FITS image header
    hdu = get_primary_hdu(fits_filename)

    # Sometimes the SIP image distortion keywords remain in the FITS
    # header, even though they have already been applied to the image
    # (i.e. a drizzled image).  This confuses astropy.wcs, so we nuke
    # all such keywords before they can cause any mischief
    if remove_sip_kwds_from_header:
        del hdu.header["A_*"]
        del hdu.header["B_*"]

    w = WCS(hdu.header)

    xs, ys, x, y = get_arc_xy(region_filename, None, wcs=w,
                              resample=resample, resample_fraction=resample_fraction)

    # Size of view port
    size = 150
    x1, x2 = xs - size, xs + size
    y1, y2 = ys - size, ys + size

    # Cut out a slice of the image to make everything else quicker
    # Leave a big enough margin to accommodate the smoothing kernel
    margin = 3*sigma + 2
    i1, i2 = int(x1 - margin), int(x2 + margin)
    j1, j2 = int(y1 - margin), int(y2 + margin)
    data_slice = hdu.data[j1:j2, i1:i2]
    wslice = w.slice((slice(j1, j2), slice(i1, i2)))

    # Get the points again, but with the new sliced wcs
    xs, ys, x, y = get_arc_xy(region_filename, None, wcs=wslice,
                              resample=resample, resample_fraction=resample_fraction)
    x1, x2 = xs - size, xs + size
    y1, y2 = ys - size, ys + size
    fit = IteratedFit(x, y, xs, ys, delta_theta, verbose=verbose, maxiter=maxiter)

    # Save the important parameters of last fit to a JSON file
    savecircle = fit.circles[-1]
    fileprefix, _ = os.path.splitext(plotfile)
    savedict = {
        "info": "Last iterated fit from circle_fit.py",
        "region file": str(region_filename),
        "FITS file": str(fits_filename),
        "d theta": delta_theta,
        "Pi": savecircle.Pi,
        "Lambda": savecircle.Lambda,
        "d Lambda": savecircle.dLambda,
        "R0": savecircle.R0,
        "axis angle": savecircle.angle,
    }
    with open(fileprefix + ".json", "w") as f:
        json.dump(savedict, f, indent=4)

    # Find suitable limits if not explicitly specified
    vmedian = np.median(data_slice)
    mpos = data_slice > vmedian  # mask of pixels brighter than median
    vmpd = np.median(data_slice[mpos] - vmedian)  # median positive deviation
    vmnd = np.median(vmedian - data_slice[~mpos])  # median negative deviation
    if vmax is None:
        vmax = vmedian + 5*vmpd
    if vmin is None:
        vmin = vmedian - 5*vmnd

        
    # Plot the image data from the FITS file
    fig, ax = plt.subplots(subplot_kw=dict(projection=wslice))
    ax.imshow(data_slice, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')

    # Contour of a smoothed version of image
    ax.contour(
        convolve_fft(data_slice, Gaussian2DKernel(x_stddev=sigma)),
        levels=np.linspace(vmin, vmax, 15),
        linewidths=0.5)

    m = fit.masks[-1]
    ax.scatter(x[m], y[m], s=10, color='r', zorder=2)
    ax.scatter(x[~m], y[~m], s=10, color='w', zorder=2)

    colors = sns.color_palette("Oranges_r", n_colors=len(fit.circles))
    for c, color in zip(fit.circles, colors):
        ax.add_patch(
            matplotlib.patches.Circle(c.rc, radius=c.Rc, ec=color, fc='none'))
        ax.plot(
            [c.rc[0], c.rc[0] + 1.2*c.Rc*c.xihat[0]],
            [c.rc[1], c.rc[1] + 1.2*c.Rc*c.xihat[1]],
            ls="--", color=color,
        )
        ax.scatter(c.rc[0], c.rc[1], s=30, color=color)
        print(c)
    # Draw the R_90 radii for the final fit
    # Use fact that perpendicular(x, y) = (-y, x)
    ax.plot(
        [xs - c.R90[0]*c.xihat[1], xs + c.R90[1]*c.xihat[1]],
        [ys + c.R90[0]*c.xihat[0], ys - c.R90[1]*c.xihat[0]],
        ls=":", color=color,
        )
    
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
    plt.close(fig)              # Avoid resource leak!
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

    try:
        DELTA_THETA = float(sys.argv[2])
    except:
        DELTA_THETA = 75.0

    try:
        BOOTSTRAP_RESAMPLE_REPLACEMENT_FRACTION = float(sys.argv[3])
    except:
        BOOTSTRAP_RESAMPLE_REPLACEMENT_FRACTION = 0.5

    TEST_PLOT_FILE = TEST_PLOT_FILE.replace(".pdf", f"-{int(DELTA_THETA):02d}.pdf")

    # Test with simple points
    print("### Simple Test")
    results = fit_circle_to_xy(*TESTDATA)
    assert np.allclose(results.x, TESTCENTER)

    # Test with real image and region file
    print("### Image Test")
    print("Figure file:",
          plot_solution(TEST_REGION_FILE, TEST_FITS_FILE, TEST_PLOT_FILE, DELTA_THETA))

    # Test the resampling
    for j in range(5):
        print(f"### Resample Test {j:01d}")
        print("Figure file:",
              plot_solution(
                  TEST_REGION_FILE,
                  TEST_FITS_FILE,
                  TEST_PLOT_FILE.replace(".pdf", f"-resample{j:01d}.pdf"),
                  DELTA_THETA,
                  resample=True,
                  resample_fraction=BOOTSTRAP_RESAMPLE_REPLACEMENT_FRACTION,
              ))
