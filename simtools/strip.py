"""Two-layer nanostrip.

Two-layer nanostrip with material parameters based on FeGe to study Bloch
points in confined nanostructures.
"""
import math
import os

import discretisedfield as df
import discretisedfield.tools as dft
import matplotlib.pyplot as plt
import micromagneticmodel as mm
import numpy as np
import oommfc as mc

from . import short_init


class Strip:
    """Class for two-layer nanostrips with exchange, DMI, and demag.

    A number of subregions is defined. There is a distinction between fixed
    and free subregions. If `subdivide_free=False` there are two "free"
    subregions called `freebottom` and `freetop`. Otherwise there is a number
    of free subregions called `free{n}bottom`/`free{n}top` with
    n = (1, 2, ..., subdivisions).

    Initpattern is the most convenient way to specify Bloch point types and
    positions. It expects a string consisting of `i` and `o` (and optional an
    arbitrary other letter to mark spaces) that describes the arrangement in
    x-direction. To conveniently create these patterns use the functions
    provided in `short_init`. It is highly recommented to use initpattern
    instead of: subdivisions, positions, m_values, and subdivide_free.

    Parameters
    ----------
    length : numbers.Real
        Strip length in m.

    width : numbers.Real
        Strip width in m.

    htop : numbers.Real
        Top-layer thickness in m.

    hbottom : numbers.Real
        Bottom-layer thickness in m.

    dirname : str
        Name of the directory to use.

    initpattern : str
        Initpattern to specify Bloch point types and positions. Expects a
        string consisting of `i` and `o` (and optional an other letter to mark
        spaces) that describes the arrangement in x-direction.

    bottom_fixed_cells : int, optional
        Number of cells that are fixed at the bottom sample boundary, defaults
        to 1.

    top_fixed_cells : int, optional
        Number of cells that are fixed at the top sample boundary, defaults
        to 1.

    fixed_size : tuple(float, float)
        Fraction of a subregion to fix in x, y direction. Defaults to fixing
        half of the subregion size in x and y. The fixed subregion is centred
        in the free subregion.

    cell : (3,) array-like, optional
        Finite-difference cell size.

    subdivisions : int
        Number of subdivisions for optionally fixing cells during
        initialisation.

    positions : list(int)
        List of positions where cells should be fixed (counting starts at 1).

    subdivide_free : bool, optional
        Subdivide the free subregions, defaults to `False` where only one
        subregion for top and bottom layer is defined.

    m_values : list(str, discretisedfield.Field.value)
        Initialisation of the magnetic field. The first parameter is used as
        part of the file name and must be a valid Python variable name. The
        Second parameter can be anything accepted by `discretisedfield.Field`
        for initialisation. Fixed/free subregions that are labelled during
        mesh initialisation can be used for fine control.

    bc : str, optional
        Defines boundary conditions like `discretisedfield.Mesh`

    structure : numbers.Real, callable, optional
        Define structure of the strip by specifying a varying norm of
        `discretisedfield.Field`.

    """

    FeGe = {'Ms': 384e3, 'A': 8.87e-12, 'D': 1.58e-3,
            'crystalclass': 'T', 'alpha': 0.28}

    def __init__(self,
                 length,
                 width,
                 htop,
                 hbottom,
                 dirname,
                 initpattern=None,
                 bottom_fixed_cells=1,
                 top_fixed_cells=1,
                 fixed_size=(.5, .5),
                 cell=(2.5e-9, 2.5e-9, 2.5e-9),
                 subdivisions=None,
                 positions=None,
                 subdivide_free=False,
                 m_values=None,
                 bc='',
                 structure=None):
        self.dirname = dirname

        self.htop = htop
        self.hbottom = hbottom

        if (initpattern is None
            and (subdivisions is None or positions is None
                 or m_values is None)):
            raise RuntimeError('Either [`initpattern`] or [`subdivisions`,'
                               ' `positions`, `m_values`] must be defined.')
        elif initpattern is not None:
            config = short_init.pattern_to_config(initpattern)
            subdivisions = config['subdivisions']
            positions = config['positions']
            subdivide_free = config['subdivide_free']
            m_values = config['m_values']

        mesh = self._init_mesh(length=length,
                               width=width,
                               htop=htop,
                               hbottom=hbottom,
                               subdivisions=subdivisions,
                               positions=positions,
                               bottom_fixed_cells=bottom_fixed_cells,
                               top_fixed_cells=top_fixed_cells,
                               fixed_size=fixed_size,
                               cell=cell,
                               subdivide_free=subdivide_free,
                               bc=bc)

        init_m_name, init_m_value = m_values

        self.basename = Strip.get_basename(length=length, width=width,
                                           ht=htop, m0=init_m_name, bc=bc)
        self.system = mm.System(name=self.basename)

        D_dict = {'default': Strip.FeGe['D']}
        subregions = list(mesh.subregions.keys())
        for i, subregion1 in enumerate(subregions):
            if 'bottom' in subregion1:
                for subregion2 in subregions[i:]:
                    if 'bottom' in subregion2:
                        D_dict[f'{subregion1}:{subregion2}'] = -Strip.FeGe['D']
        self.system.energy = (mm.Exchange(A=Strip.FeGe['A'])
                              + mm.Demag()
                              + mm.DMI(D=D_dict,
                                       crystalclass=Strip.FeGe['crystalclass'])
                              )
        self.system.dynamics = (mm.Damping(alpha=Strip.FeGe['alpha'])
                                + mm.Precession(gamma0=mm.consts.gamma0))

        if structure is not None:
            norm = structure
        else:
            norm = Strip.FeGe['Ms']
        self.system.m = df.Field(mesh=mesh, dim=3, value=init_m_value,
                                 norm=norm)

    def min_drive(self, fixed=False, verbose=1, **kwargs):
        """Run minimisation driver for the strip.

        Parameters
        ----------
        fixed : bool, optional
            Fix subregions that contain the substring `fixed` in their names.

        kwargs : dict, optional
            Parameters that are accepted for the initialisation of the
            `MinDriver` of Ubermag.
        """
        md = mc.MinDriver(**kwargs)
        if fixed:
            fixed_sub = [sub for sub in self.system.m.mesh.subregions.keys()
                         if 'fixed' in sub]
        else:
            fixed_sub = None
        md.drive(self.system, dirname=self.dirname, fixed_subregions=fixed_sub,
                 verbose=verbose)

    def time_drive(self, t, n, fixed=False, verbose=1, **kwargs):
        """Run time driver for the strip.

        Parameters
        ----------
        t : numbers.Real
            Time is seconds.

        n : int
            Number of steps to store.

        kwargs : dict, optional
            Parameters that are accepted for the initialisation of the
            `TimeDriver` in Ubermag.
        """
        td = mc.TimeDriver(**kwargs)
        if fixed:
            fixed_sub = [sub for sub in self.system.m.mesh.subregions.keys()
                         if 'fixed' in sub]
        else:
            fixed_sub = None
        td.drive(self.system, t=t, n=n, fixed_subregions=fixed_sub,
                 dirname=self.dirname, verbose=verbose)

    def count_bp_and_save(self, minimisation):
        """Count Bloch points and saves results.

        Counts all Bloch points based on the emergent magnetic field and
        a series of integrals in x-direction.

        Saves system data available in `system.table` as csv. Some additional
        data is stored: `length` and `width` in nm; The total topological
        charge `q` and an absolute version `q_abs` computed as integral over
        div(F); Bloch point numbers (total, HH, TT); Bloch point pattern in
        x-direction in the form `list([<current BP number>,
        <number of cells with current BP number>)])`; number of regions with
        angles between neighbouring cells above 80/90 degrees (as additional
        check for the classification).

        Saves the magnetisation field as `hdf5`.

        Parameters
        ----------
        minimisation : str
            String to specify additional information when saving the file.
            `minimisation` is added to the end of the filename. Used to
            distinguish between files after different types of minimisations.
        """
        end_data = self.system.table.data.tail(1).copy()
        end_data['length'] = round(self.system.m.mesh.region.edges[0] * 1e9)
        end_data['width'] = round(self.system.m.mesh.region.edges[1] * 1e9)
        end_data['htop'] = round(self.htop * 1e9)
        end_data['hbottom'] = round(self.hbottom * 1e9)
        for i, axis in enumerate('xyz'):
            end_data[f'mesh_n_{axis}'] = self.system.m.mesh.n[i]

        F = dft.emergent_magnetic_field(self.system.m.orientation)
        F_div = F.div
        q_volint = df.integral(F_div * df.dV) / (4 * np.pi)
        q_abs_volint = df.integral(abs(F_div) * df.dV) / (4 * np.pi)
        end_data['q'] = q_volint
        end_data['q_abs'] = q_abs_volint

        bp_details = dft.count_bps(self.system.m)
        end_data['bp_number'] = bp_details['bp_number']
        end_data['bp_number_hh'] = bp_details['bp_number_hh']
        end_data['bp_number_tt'] = bp_details['bp_number_tt']
        end_data['bp_arrangement'] = bp_details['bp_pattern_x']

        end_data.to_csv(os.path.join(self.dirname, self.basename,
                        f'{self.basename}.{minimisation}.csv'), index=False)
        self.system.m.write(os.path.join(self.dirname, self.basename,
                                         f'{self.basename}.'
                                         f'{minimisation}.hdf5'))

    def show_m(self, z_top=1e-9, z_bottom=-1e-9, title=None):
        """Plot plane in +-xy and xz."""
        fig, axs = plt.subplots(ncols=3, figsize=(24, 6))
        m = self.system.m.orientation
        m.plane(z=z_top).mpl(ax=axs[0],
                             scalar_kw=dict(colorbar=False,
                                            clim=(-1, 1)))
        axs[0].set_title(f'z = {round(z_top * 1e9)}nm')
        m.plane(z=z_bottom).mpl(ax=axs[1],
                                scalar_kw=dict(colorbar=False,
                                               clim=(-1, 1)))
        axs[1].set_title(f'z = {round(z_bottom * 1e9)}nm')
        m.plane('y').z.mpl(ax=axs[2],
                           scalar_kw=dict(colorbar=False,
                                          colorbar_label='m_z',
                                          clim=(-1, 1)))
        axs[2].set_title(f'y = {round(m.mesh.region.edges[1] * 1e9 / 2)}nm')
        if title is not None:
            fig.suptitle(f'{title}\n', fontsize=24)
        plt.tight_layout()

    def _compute_subregion_edge_points(self, mesh, xlen, xlower, ylower,
                                       zlower, xcells, ycells, zcells):
        """Compute boundaries for fixed subregions.

        Parameters
        ----------
        mesh : discretisedfield.Mesh
            Mesh to operate on.
        """
        xfree = math.floor((xlen - xcells) / 2)
        yfree = math.floor((mesh.n[1] - ycells) / 2)
        n1 = [xlower + xfree, ylower + yfree, zlower]
        n2 = [xlower + xfree + xcells, ylower + yfree + ycells,
              zlower + zcells]

        p1 = list(mesh.index2point(n1))
        p2 = list(mesh.index2point(n2))

        p1[0] -= mesh.dx / 2
        p1[1] -= mesh.dy / 2
        p1[2] -= mesh.dz / 2

        p2[0] += mesh.dx / 2
        p2[1] += mesh.dy / 2
        p2[2] += mesh.dz / 2
        # rounding assumes nm accuracy
        return (np.round(p1, decimals=10), np.round(p2, decimals=10))

    def _init_mesh(self, length, width, htop, hbottom, subdivisions, positions,
                   bottom_fixed_cells: int, top_fixed_cells: int, fixed_size,
                   subdivide_free, cell, bc):
        """Initialise mesh using the parameters given to __init__."""
        if subdivisions < 1:
            raise ValueError('The number of subdivisions must be positive.')
        if any(map(lambda x: x < 1 or x > subdivisions, positions)):
            msg = 'Values of positions must be within [1, subdvisisions]'
            raise ValueError(msg)

        region = df.Region(p1=(0, 0, -hbottom),
                           p2=(length, width, htop))
        mesh = df.Mesh(region=region, cell=cell, bc=bc)
        subregions = {}

        # last subregion can be larger
        sub_n = math.floor(mesh.n[0] / subdivisions)
        sub_length = sub_n * mesh.dx
        lower_x = 0
        for p in range(1, subdivisions + 1):
            if p in positions:
                x_cells = math.floor(sub_n * fixed_size[0])
                y_cells = math.floor(mesh.n[1] * fixed_size[1])
                lower_xn = sub_n * (p - 1)
                # bottom
                p1, p2 = self._compute_subregion_edge_points(
                    mesh, xlen=sub_n, xlower=lower_xn, ylower=0, zlower=0,
                    xcells=x_cells, ycells=y_cells,
                    zcells=bottom_fixed_cells - 1)
                subregions[f'fixed{p}bottom'] = df.Region(p1=p1, p2=p2)
                # top
                p1, p2 = self._compute_subregion_edge_points(
                    mesh, xlen=sub_n, xlower=lower_xn, ylower=0,
                    zlower=mesh.n[2] - top_fixed_cells,
                    xcells=x_cells, ycells=y_cells,
                    zcells=top_fixed_cells - 1)
                subregions[f'fixed{p}top'] = df.Region(p1=p1, p2=p2)

            # free subregions surrounding fixed subregions
            x_pos = p * sub_length
            if subdivide_free:
                if p == subdivisions:
                    x_pos = length
                subregions[f'free{p}bottom'] = df.Region(
                    p1=(lower_x, 0, -hbottom), p2=(x_pos, width, 0))
                subregions[f'free{p}top'] = df.Region(
                    p1=(lower_x, 0, 0), p2=(x_pos, width, htop))
                lower_x = x_pos

        if not subdivide_free:
            subregions['freebottom'] = df.Region(p1=(0, 0, -hbottom),
                                                 p2=(length, width, 0))
            subregions['freetop'] = df.Region(p1=(0, 0, 0),
                                              p2=(length, width, htop))

        mesh.subregions = subregions
        return mesh

    @classmethod
    def get_basename(cls, length, width, ht, m0, bc=''):
        """Basename of a simulation.

        The name has the form
        `mbp_l_<length>_w_<width>_ht_<ht>_[bc_<bc>_]<initial_m>`.
        Information about boundary conditions is only part of the file
        name if `bc != ''`.
        All lengths are defined in nm.
        """
        return (f'mbp_l_{int(round(length * 1e9))}_w_{int(round(width * 1e9))}'
                f'_ht_{int(round(ht * 1e9))}{f"_bc_{bc}" if bc != "" else ""}'
                f'_{m0}')
