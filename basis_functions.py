import numpy as np
import matplotlib.pyplot as plt
from typing import List
from numpy.typing import NDArray
from tqdm import tqdm


def lagrange_polynomials_1d(xs):
    """
    Args:
        xs: array, nodes that defines the Lagrange bases

    Return:
        funcs: list of callables, funcs[i](x) evaluates the Lagrange basis l_i(x)
    """
    funcs = []
    for j in range(len(xs)):

        def lj(x, j=j):
            ljx = 1.0
            for m in range(len(xs)):
                if m != j:
                    ljx *= (x - xs[m]) / (xs[j] - xs[m])
            return ljx

        funcs.append(lj)
    return funcs


def lagrange_polynomials_2d(xs1, xs2):
    """
    2D Lagrange basis functions via tensor product

    Args:
        xs: array, nodes that defines the Lagrange bases

    Return:
        funcs: list of callables, funcs[i][j](x, y) evaluates l_i(x) * l_j(y),
               where l_i and l_j are Lagrange bases
    """
    li = lagrange_polynomials_1d(xs1)
    lj = lagrange_polynomials_1d(xs2)

    funcs = []
    for i in range(len(li)):
        funcs.append([])
        for j in range(len(lj)):
            funcs[i].append(lambda x, y, i=i, j=j: li[i](x) * lj[j](y))
    return funcs


def polynomials_fit_1d(p, pts):
    """
    Args:
        p: polynomial degree
        pts: list of x-coordinates
    """
    Nk = len(pts)
    Np = 1 + p
    if Nk != Np:
        print("Nk != Np (%d, %d), can't invert Vk" % (Nk, Np))

    Vk = np.zeros((Nk, Np))
    for j in range(Np):
        Vk[:, j] = pts**j

    Ck = np.linalg.inv(Vk)
    cond_Vk = np.linalg.cond(Vk)

    funcs = []
    for i in range(Nk):

        def phi(x, i=i):
            ret = 0.0
            for j in range(Np):
                ret += Ck[j, i] * x**j
            return ret

        funcs.append(phi)
    return funcs, cond_Vk


def polynomials_fit_1d_ref_elem(p, pts):
    """
    Args:
        p: polynomial order along one dimension
        pts: list of pts
    """
    pts = np.array(pts)
    pt_min = np.min(pts)
    pt_max = np.max(pts)
    h = pt_max - pt_min
    pts_ref = -1.0 + 2.0 * (pts - pt_min) / h

    funcs_ref, cond_Vk = polynomials_fit_1d(p, pts_ref)

    funcs = []

    for fref in funcs_ref:

        def phi(x, fref=fref):
            return fref(
                -1.0 + 2.0 * (x - pt_min) / h,
            )

        funcs.append(phi)

    return funcs, cond_Vk


def polynomials_fit_2d(p, pts):
    """
    Args:
        p: polynomial order along one dimension
        pts: list of pts
    """

    Nk = len(pts)
    Np_1d = 1 + p
    Np = Np_1d**2
    if Nk != Np:
        print("Nk != Np (%d, %d), can't invert Vk" % (Nk, Np))

    Vk = np.zeros((Nk, Np))
    for i, xy in enumerate(pts):
        x = xy[0]
        y = xy[1]
        xpows = [x**j for j in range(Np_1d)]
        ypows = [y**j for j in range(Np_1d)]
        for j in range(Np_1d):
            for k in range(Np_1d):
                idx = j * Np_1d + k
                Vk[i, idx] = xpows[j] * ypows[k]

    Ck = np.linalg.inv(Vk)
    cond_Vk = np.linalg.cond(Vk)

    funcs = []
    for i in range(Nk):

        def phi(x, y, i=i):
            xpows = [x**j for j in range(Np_1d)]
            ypows = [y**j for j in range(Np_1d)]
            ret = 0.0
            for j in range(Np_1d):
                for k in range(Np_1d):
                    idx = j * Np_1d + k
                    ret += Ck[idx, i] * xpows[j] * ypows[k]
            return ret

        funcs.append(phi)

    return funcs, cond_Vk


def polynomials_fit_2d_ref_elem(p, pts):
    """
    Args:
        p: polynomial order along one dimension
        pts: list of pts
    """
    pts = np.array(pts)
    pt_min = pts.min(axis=0)
    pt_max = pts.max(axis=0)
    h = pt_max - pt_min
    pts_ref = -1.0 + 2.0 * (pts - pt_min) / h

    # plt.plot(pts_ref[:, 0], pts_ref[:, 1], "o")
    # plt.grid()
    # plt.show()
    # exit()

    funcs_ref, cond_Vk = polynomials_fit_2d(p, pts_ref)

    funcs = []

    for fref in funcs_ref:

        def phi(x, y, fref=fref):
            return fref(
                -1.0 + 2.0 * (x - pt_min[0]) / h[0],
                -1.0 + 2.0 * (y - pt_min[1]) / h[1],
            )

        funcs.append(phi)

    return funcs, cond_Vk


def demo_poly_fit_1d(p=3):
    start = 40.0
    stop = p + 40.0
    num = p + 1
    xs = np.linspace(start, stop, num)
    funcs_lag = lagrange_polynomials_1d(xs)
    funcs_fit = polynomials_fit_1d(p, xs)

    x = np.linspace(start, stop, 201)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    for i, fun in enumerate(funcs_lag):
        axs[0].plot(x, fun(x), label="basis %d" % i)

    for i, fun in enumerate(funcs_fit):
        axs[1].plot(x, fun(x), label="basis %d" % i)

    for i, (fun1, fun2) in enumerate(zip(funcs_lag, funcs_fit)):
        axs[2].semilogy(x, fun1(x) - fun2(x), label="basis %d" % i)

    axs[0].set_title("Lagrange polynomials")
    axs[1].set_title("Polynomials by fit")
    axs[2].set_title("Error")

    for ax in axs:
        ax.legend()
        ax.grid()

    plt.show()
    return


def demo_poly_fit_2d(p=3, i=0, j=0):
    start = 5.0
    stop = p + 5.0
    num = p + 1

    pts = np.linspace(start, stop, num)
    funcs_lag = lagrange_polynomials_2d(pts, pts)

    pts2 = [(i, j) for i in pts for j in pts]

    funcs_fit, cond = polynomials_fit_2d(p, pts2)
    funcs_fit_ref, cond_ref = polynomials_fit_2d_ref_elem(p, pts2)

    x = np.linspace(start, stop, 201)
    y = np.linspace(start, stop, 201)
    x, y = np.meshgrid(x, y)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    axs[0].contour(x, y, funcs_lag[i][j](x, y), levels=100)
    axs[1].contour(x, y, funcs_fit[(p + 1) * i + j](x, y), levels=100)
    axs[2].contour(x, y, funcs_fit_ref[(p + 1) * i + j](x, y), levels=100)

    axs[0].set_title("Lagrange polynomials (%d, %d)" % (i, j))
    axs[1].set_title("Polynomials by fit, cond(Vk): %.2e" % cond)
    axs[2].set_title("Polynomials by fit and normalization, cond(Vk): %.2e" % cond_ref)

    plt.show()

    return


def test_gd_impl(Np_1d=2):
    pts = [2.0 * i / (Np_1d - 1.0) - 1.0 for i in range(Np_1d)]
    pts = np.array(pts)
    print(pts)
    pts2 = [(i, j) for j in pts for i in pts]
    pts2 = np.array(pts2)

    funcs_fit, cond = polynomials_fit_2d(Np_1d - 1, pts2)
    print(cond)
    exit()

    x, y = 0.39214122, -0.24213123
    for i, f in enumerate(funcs_fit):
        print("%.16f," % f(x, y))

    return


class OneDimShapeFuncDemo:
    def __init__(
        self,
        Np_1d: int,  # number of points for each element
        dof: NDArray[np.float64],  # a dof array of size nelems + 1
    ) -> None:
        self.Np_1d = Np_1d
        self.dof = dof
        self.nnodes = len(dof)
        self.nelems = self.nnodes - 1

        assert self.Np_1d % 2 == 0
        assert self.nnodes >= self.Np_1d

        return

    def get_elem_nodes(self, elem: int):
        begin = elem - (self.Np_1d // 2 - 1)
        end = begin + self.Np_1d
        if begin < 0:
            end -= begin
            begin = 0
        elif end > self.nnodes:
            begin -= end - self.nnodes
            end = self.nnodes

        nodes = np.arange(begin, end)
        return nodes

    def get_elem_samples(self, elem: int, nsamples_per_elem: int):
        x_samples = np.linspace(elem, (elem + 1), nsamples_per_elem)
        return x_samples

    def get_nodes_xloc(self, nodes: NDArray[np.int32]):
        xloc = nodes * 1.0
        return xloc

    def construct_element_func(self, elem: int):
        nodes = self.get_elem_nodes(elem)
        xloc = self.get_nodes_xloc(nodes)
        basis_funcs, cond = polynomials_fit_1d_ref_elem(self.Np_1d - 1, xloc)
        assert len(basis_funcs) == len(nodes)

        def element_func(x):
            ret = 0.0
            for i in range(len(basis_funcs)):
                ret += self.dof[nodes[i]] * basis_funcs[i](x)
            return ret

        return element_func


def on_resize(event):
    # Get the updated figure size
    new_size = event.canvas.figure.get_size_inches()
    print(f"Figure resized: {new_size[0]:.2f} x {new_size[1]:.2f} inches")


def demo_1d_shape_functions(Np_1d_list: List[int]):
    nelems = 10
    nnodes = nelems + 1
    nsamples_per_elem = 50

    ncols = len(Np_1d_list)
    fig, axs = plt.subplots(
        constrained_layout=True, ncols=ncols, nrows=nnodes, figsize=(6.86 * ncols, 8.73)
    )
    fig.canvas.mpl_connect("resize_event", on_resize)

    for j in range(ncols):
        for i in tqdm(range(nnodes)):
            dof = np.zeros(nnodes)
            dof[i] = 1.0
            demo = OneDimShapeFuncDemo(Np_1d=Np_1d_list[j], dof=dof)

            x = []
            y = []
            for elem in range(nelems):
                fe = demo.construct_element_func(elem)
                xe = demo.get_elem_samples(elem, nsamples_per_elem)
                ye = fe(xe)
                x.append(xe)
                y.append(ye)

            x = np.concatenate(x)
            y = np.concatenate(y)

            axs[i, j].plot(x, y, color="blue")
            axs[i, j].grid(axis="x")
            axs[i, j].set_xlim([0, nelems])
            axs[i, j].set_xticks(range(0, nnodes))
            axs[i, j].set_ylabel(
                f"$\\phi_{{{i}}}(x)$",
                rotation=0,
                va="center_baseline",
                ha="left",
            )
            axs[i, j].yaxis.set_label_coords(1.01, 0.5)

        axs[-1, j].text(
            0.5,
            -0.75,
            f"({chr(ord('a') + j)})",
            weight="bold",
            fontsize=12,
            ha="center",
            va="center_baseline",
            transform=axs[-1, j].transAxes,
        )

    fig.savefig(f"basis_functions.jpg", dpi=1200)
    plt.show()
    return


if __name__ == "__main__":
    # demo_poly_fit_1d()
    # demo_poly_fit_2d()
    # test_gd_impl(6)

    demo_1d_shape_functions(Np_1d_list=[2, 4, 6])
