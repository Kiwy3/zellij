from zellij.core.metaheuristic import Metaheuristic
from zellij.strategies.utils.chaos_map import Chaos_map

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CGS(Metaheuristic):

    """Chaotic Global search

    CGS is an exploration Metaheuristic using chaos to violently move over the search space.
    It is continuous optimization, so the Searchspace is converted to continuous.
    To do so, it uses a chaotic map, such as Henon or Kent map.

    Attributes
    ----------
    level : int
        Chaotic level corresponds to the number of iteration of the chaotic map
    map : Chaos_map
        Chaotic map used to sample points. See Chaos_map object.
    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous
    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous
    center : float
        List of floats containing the coordinates of the search space center converted to continuous
    radius : float
        List of floats containing the radius for each dimensions of the search space converted to continuous

    Methods
    -------

    run(self,shift=1, n_process=1)
        Runs CGS

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    Chaotic_optimization : CGS is used here to perform an exploration
    CLS : Chaotic Local Search
    CFS : Chaotic Fine Search
    """

    def __init__(self, loss_func, search_space, f_calls, level, chaos_map, create=False, verbose=True):
        """__init__(self, loss_func, search_space, f_calls, level, chaos_map, create=False, verbose=True)

        Initialize CGS class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y
        search_space : Searchspace
            Search space object containing bounds of the search space
        f_calls : int
            Maximum number of loss_func calls
        level : int
            Chaotic level corresponds to the number of iteration of the chaotic map
        map : Chaos_map
            Chaotic map used to sample points. See Chaos_map object.
        create : boolean, optional
            Deprecated, must be removed
        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func, search_space, f_calls, verbose)

        self.level = level

        if create and type(chaos_map) == str:
            self.map = Chaos_map(chaos_map, self.level, self.search_space.n_variables)
        elif type(chaos_map) != str:
            self.map = chaos_map

        ##############
        # ATTRIBUTES #
        ##############

        self.up_bounds = np.array([1 for _ in range(self.search_space.n_variables)])
        self.lo_bounds = np.array([0 for _ in range(self.search_space.n_variables)])

        # Working attributes, saved to avoid useless computations.
        self.up_plus_lo = self.up_bounds + self.lo_bounds
        self.up_m_lo = self.up_bounds - self.lo_bounds

        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.lo_bounds

    def run(self, shift=1, n_process=1):

        """run(self,shift=1, n_process=1)

        Parameters
        ----------
        shift : int, default=1
            Determine the starting point of the chaotic map.
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        self.k = shift

        # For each level of chaos
        shift_map = (self.k - 1) * self.level
        points = np.empty((0, self.search_space.n_variables), dtype=float)

        n_points = self.loss_func.calls
        l = 0

        while l < self.level and n_points < self.f_calls:

            # Randomly select a parameter index of a solution
            d = np.random.randint(self.search_space.n_variables)

            # Select chaotic_variables among the choatic map
            y = self.map.chaos_map[l + shift_map] * self.map.chaos_map[self.k - 1]
            # Apply 3 transformations on the selected chaotic variables
            r_mul_y = np.multiply(self.up_m_lo, y)

            # xx = [np.add(self.center,r_mul_y), np.add(self.center,np.multiply(self.radius,np.multiply(2,y)-1)), np.subtract(self.up_bounds,r_mul_y)]

            # for each transformation of the chaotic variable
            # for x in xx:
            #
            #     x_ = np.subtract(self.up_plus_lo,x)
            #     sym = np.matrix([x,x,x_,x_])
            #     sym[1,d] = x_[d]
            #     sym[3,d] = x[d]
            #     points = np.append(points,sym,axis=0)
            #     n_points += 4

            xx = [self.lo_bounds + r_mul_y, self.up_bounds - r_mul_y]

            # for each transformation of the chaotic variable
            sym = np.array([xx[0], xx[1], xx[0], xx[1]])
            sym[2, d] = xx[1][d]
            sym[3, d] = xx[0][d]

            points = np.append(points, sym, axis=0)
            n_points += 4

            l += 1

        ys = self.loss_func(
            self.search_space.convert_to_continuous(points, True),
            algorithm="CLS",
        )
        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        # best solution found
        best_sol = points[idx]
        best_scores = ys[idx]

        return best_sol, best_scores


class CLS(Metaheuristic):

    """Chaotic Local Search

    CLS is an exploitation Metaheuristic using chaos to wiggle points arround an initial solution.\
     It uses a rotating polygon to distribute those points, a progressive and mooving zoom on the best solution found, to refine it.
    It is continuous optimization, so the Searchspace is converted to continuous.
    To do so, it uses a chaotic map, such as Henon or Kent map.

    Attributes
    ----------
    level : int
        Chaotic level corresponds to the number of iteration of the chaotic map
    map : Chaos_map
        Chaotic map used to sample points. See Chaos_map object.
    polygon : int
        Vertex number of the rotating polygon (has an influence on the number of evaluated points)
    red_rate : float
        Reduction rate of the progressive zoom on the best solution found
    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous
    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous
    center : float
        List of floats containing the coordinates of the search space center converted to continuous
    radius : float
        List of floats containing the radius for each dimensions of the search space converted to continuous

    Methods
    -------

    run(self,X0,Y0,chaos_level=0,shift=1, n_process=1)
        Runs CLS

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    Chaotic_optimization : CLS is used here to perform an exploitation
    CGS : Chaotic Global Search
    CFS : Chaotic Fine Search
    """

    def __init__(self, loss_func, search_space, f_calls, level, polygon, chaos_map, red_rate=0.5, verbose=True):

        """__init__(self, loss_func, search_space, f_calls, level, polygon, chaos_map, red_rate=0.5, verbose=True)

        Initialize CLS class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y
        search_space : Searchspace
            Search space object containing bounds of the search space
        f_calls : int
            Maximum number of loss_func calls
        level : int
            Chaotic level corresponds to the number of iteration of the chaotic map
        polygon : int
            Vertex number of the rotating polygon (has an influence on the number of evaluated points)
        map : Chaos_map
            Chaotic map used to sample points. See Chaos_map object.
        red_rate : float
            Reduction rate of the progressive zoom on the best solution found
        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func, search_space, f_calls, verbose)

        self.level = level
        self.polygon = polygon
        self.map = chaos_map
        self.red_rate = red_rate

        #############
        # VARIABLES #
        #############

        self.up_bounds = np.array([1 for _ in self.search_space.values])
        self.lo_bounds = np.array([0 for _ in self.search_space.values])

        self.up_plus_lo = self.up_bounds + self.lo_bounds
        self.up_m_lo = self.up_bounds - self.lo_bounds

        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.lo_bounds

        trigo_val = 2 * np.pi / self.polygon
        self.H = [np.zeros(self.polygon), np.zeros(self.polygon)]

        for i in range(1, self.polygon + 1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i - 1] = np.cos(trigo_val * i)
            self.H[1][i - 1] = np.sin(trigo_val * i)

    def run(self, X0, Y0, chaos_level=0, shift=1, n_process=1):

        """run(self,shift=1, n_process=1)

        Parameters
        ----------
        X0 : list[float]
            Initial solution
        Y0 : {int, float}
            Score of the initial solution
        chaos_level : int, default=0
            Determine at which level of the chaos map, the algorithm starts
        shift : int, default=1
            Determine the starting point of the chaotic map.
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        self.X0 = X0
        self.Y0 = Y0
        self.k = shift
        self.chaos_level = chaos_level

        # Initialization
        shift = self.chaos_level * (self.k - 1) * self.level
        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(self.up_bounds - self.X0, self.X0 - self.lo_bounds)

        # Local search area radius
        Rl = self.radius * self.red_rate

        center_m_solution = self.center - self.X0
        points = np.empty((0, self.search_space.n_variables), dtype=float)

        n_points = self.loss_func.calls
        l = 0
        # for each level of chaos
        while l < self.level and n_points < self.f_calls:

            # Decomposition vector
            d = np.random.randint(self.search_space.n_variables)

            # zoom speed
            gamma = 1 / (10 ** (2 * self.red_rate * l) * (l + 1))

            # for each parameter of a solution determine the improved radius
            xx = np.minimum(gamma * Rl, db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [
                np.multiply(xx, self.map.chaos_map[shift + l]),
                np.multiply(xx, self.map.inverted_choas_map[shift + l]),
            ]

            # For both chaotic variable
            for x in xv:
                xi = np.outer(self.H[1], x)
                xi[:, d] = x[d] * self.H[0]
                xt = self.X0 + xi

                points = np.append(points, xt, axis=0)
                n_points += self.polygon

            l += 1

        ys = self.loss_func(
            self.search_space.convert_to_continuous(points, True),
            algorithm="CLS",
        )
        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        # best solution found
        best_sol = points[idx]
        best_scores = ys[idx]

        return best_sol, best_scores


class CFS(Metaheuristic):

    """Chaotic Fine Search

    CFS is an exploitation Metaheuristic using chaos to wiggle points arround an initial solution.\
     Contrary to CLS, CFS uses an exponential zoom on the best solution found, it works at a much smaller scale than the CLS.
    It is continuous optimization, so the Searchspace is converted to continuous.
    To do so, it uses a chaotic map, such as Henon or Kent map.

    Attributes
    ----------
    level : int
        Chaotic level corresponds to the number of iteration of the chaotic map
    map : Chaos_map
        Chaotic map used to sample points. See Chaos_map object.
    polygon : int
        Vertex number of the rotating polygon (has an influence on the number of evaluated points)
    red_rate : float
        Reduction rate of the progressive zoom on the best solution found
    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous
    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous
    center : float
        List of floats containing the coordinates of the search space center converted to continuous
    radius : float
        List of floats containing the radius for each dimensions of the search space converted to continuous

    Methods
    -------
    __init__(self, loss_func, search_space, f_calls, level, polygon, chaos_map, red_rate=0.5, verbose=True)
        Initializes CFS

    run(self,X0,Y0,chaos_level=0,shift=1, n_process=1)
        Runs CFS

    stochastic_round(self, solution, k)
        Implements random perturbations to the exponential zoom.

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    Chaotic_optimization : CLS is used here to perform an exploitation
    CGS : Chaotic Global Search
    CLS : Chaotic Local Search
    """

    def __init__(self, loss_func, search_space, f_calls, level, polygon, chaos_map, red_rate=0.5, verbose=True):

        """__init__(self, loss_func, search_space, f_calls, level, polygon, chaos_map, red_rate=0.5, verbose=True)

        Initialize CLS class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y
        search_space : Searchspace
            Search space object containing bounds of the search space
        f_calls : int
            Maximum number of loss_func calls
        level : int
            Chaotic level corresponds to the number of iteration of the chaotic map
        polygon : int
            Vertex number of the rotating polygon (has an influence on the number of evaluated points)
        map : Chaos_map
            Chaotic map used to sample points. See Chaos_map object.
        red_rate : float
            Reduction rate of the progressive zoom on the best solution found
        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func, search_space, f_calls, verbose)

        self.level = level
        self.polygon = polygon
        self.map = chaos_map
        self.red_rate = red_rate

        #############
        # VARIABLES #
        #############

        self.up_bounds = np.array([1 for _ in self.search_space.values])
        self.lo_bounds = np.array([0 for _ in self.search_space.values])

        self.up_plus_lo = self.up_bounds + self.lo_bounds
        self.up_m_lo = self.up_bounds - self.lo_bounds

        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.lo_bounds

        trigo_val = 2 * np.pi / self.polygon
        self.H = [np.zeros(self.polygon), np.zeros(self.polygon)]

        for i in range(1, self.polygon + 1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i - 1] = np.cos(trigo_val * i)
            self.H[1][i - 1] = np.sin(trigo_val * i)

    def stochastic_round(self, solution, k):

        r = np.random.uniform(-1, 1, len(solution))
        # perturbation on CFS zoom
        z = np.round(solution) + (k % 2) * r

        return z

    def run(self, X0, Y0, chaos_level=0, shift=1, n_process=1):

        """run(self,X0,Y0,chaos_level=0,shift=1, n_process=1)

        Parameters
        ----------
        X0 : list[float]
            Initial solution
        Y0 : {int, float}
            Score of the initial solution
        chaos_level : int, default=0
            Determine at which level of the chaos map, the algorithm starts
        shift : int, default=1
            Determine the starting point of the chaotic map.
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        self.X0 = X0
        self.Y0 = Y0
        self.k = shift
        self.chaos_level = chaos_level

        shift = self.chaos_level * (self.k - 1) * self.level

        y = self.map.chaos_map[shift] * self.map.chaos_map[self.k - 1]
        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(self.up_bounds - self.X0, self.X0 - self.lo_bounds)

        r_g = np.zeros(self.search_space.n_variables)

        # Randomly select the reduction rate
        # red_rate = random.random()*0.5

        # Local search area radius
        Rl = self.radius * self.red_rate

        xc = self.X0
        zc = self.Y0

        center_m_solution = self.center - self.X0
        points = np.empty((0, self.search_space.n_variables), dtype=float)

        n_points = self.loss_func.calls
        l = 0
        # for each level of chaos
        while l < self.level and n_points < self.f_calls:
            # Decomposition vector
            d = np.random.randint(self.search_space.n_variables)

            # Exponential Zoom factor on the search window
            pc = 10 ** (l + 1)

            # Compute the error/the perturbation applied to the solution
            error_g = np.absolute(self.X0 - (self.stochastic_round(pc * self.X0, shift + l) / pc))

            r = np.random.random()

            # for each parameter of a solution determine the improved radius
            r_g = np.minimum((Rl * error_g) / (l ** 2 + 1), db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [np.multiply(r_g, y), np.multiply(r_g, y)]

            # For both chaotic variable
            for x in xv:
                xi = np.outer(self.H[1], x)
                xi[:, d] = x[d] * self.H[0]
                xt = self.X0 + xi

                points = np.append(points, xt, axis=0)
                n_points += self.polygon

            l += 1

        ys = self.loss_func(
            self.search_space.convert_to_continuous(points, True),
            algorithm="CFS",
        )

        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        # best solution found
        best_sol = points[idx]
        best_scores = ys[idx]

        return best_sol, best_scores


class Chaotic_optimization(Metaheuristic):

    """Chaotic_optimization

    Chaotic optimization combines CGS, CLS and CFS. Using a unique chaos map. You can determine the number of outer and inner iteration is determine using an exploration ratio,\
     and according to chaotic levels associated to CGS, CLS and CFS. The best solution found by CGS is used as a starting for CLS, and the best solution found by CLS is used by CFS.

    Attributes
    ----------

    chaos_map : {'henon', 'kent', 'tent', 'logistic', 'random', Chaos_map}
        If a string is given, the algorithm will select the corresponding map. The chaotic map is used to sample points.\
         If it is a map, it will directly use it. Be carefull, the map size must be sufficient according to the parametrization.

    exploration_ratio : float
        It will determine the number of calls to the loss function dedicated to exploration and exploitation, according to chaotic levels associated to CGS, CLS and CFS.

    polygon : int
        Vertex number of the rotating polygon (has an influence on the number of evaluated points) for CLS and CFS

    red_rate : float
        Reduction rate of the progressive zoom on the best solution found for CLS and CFS

    CGS_level : int
        Number of chaotic level associated to CGS

    CLS_level : int
        Number of chaotic level associated to CLS

    CFS_level : int
        Number of chaotic level associated to CFS

    verbose : boolean, default=True
        Algorithm verbosity

    Methods
    -------

    run(self, n_process=1)
        Runs Chaotic_optimization

    show(filename=None)
        Plots results

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    CGS : Chaotic Global Search
    CLS : Chaotic Local Search
    CFS : Chaotic Fine Search
    """

    def __init__(self, loss_func, search_space, f_calls, chaos_map="henon", exploration_ratio=0.30, levels=(32, 6, 2), polygon=4, red_rate=0.5, verbose=True):

        """__init__(self, loss_func, search_space, f_calls,chaos_map="henon", exploration_ratio = 0.70, levels = (32,6,2), polygon=4, red_rate=0.5, verbose=True)

        Initialize CGS class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y
        search_space : Searchspace
            Search space object containing bounds of the search space
        f_calls : int
            Maximum number of loss_func calls
        chaos_map : {'henon', 'kent', 'tent', 'logistic', 'random', Chaos_map}
            If a string is given, the algorithm will select the corresponding map. The chaotic map is used to sample points.\
             If it is a map, it will directly use it. Be carefull, the map size must be sufficient according to the parametrization.
        exploration_ratio : float, default=0.80
            Must be between 0 and 1.\
            It will determine the number of calls to the loss function dedicated to exploration and exploitation, according to chaotic levels associated to CGS, CLS and CFS.
        levels : (int, int, int)
            Used to determine the number of chaotic levels for respectively, CGS, CLS and CFS.
        polygon : int, default=4
            Vertex number of the rotating polygon (has an influence on the number of evaluated points) for CLS and CFS
        red_rate : float, default=0.5
            Reduction rate of the progressive zoom on the best solution found
        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func, search_space, f_calls, verbose)

        self.chaos_map = chaos_map
        self.exploration_ratio = exploration_ratio
        self.polygon = polygon
        self.red_rate = red_rate

        self.CGS_level = levels[0]
        self.CLS_level = levels[1]
        self.CFS_level = levels[2]

        #############
        # VARIABLES #
        #############

        if self.CGS_level > 0:
            if self.CLS_level != 0 or self.CFS_level != 0:
                self.iterations = np.ceil((self.f_calls * self.exploration_ratio) / (4 * self.CGS_level))
                self.inner_iterations = np.ceil((self.f_calls * (1 - self.exploration_ratio)) / ((self.CLS_level + self.CFS_level) * self.polygon * self.iterations))
            else:
                self.iterations = np.ceil(self.f_calls / (4 * self.CGS_level))
                self.inner_iterations = 0
        else:
            raise ValueError("CGS level must be > 0")

        if type(chaos_map) == str:
            self.map_size = int(
                np.max(
                    [
                        self.iterations * self.CGS_level,
                        self.iterations * self.inner_iterations * self.CLS_level,
                        self.iterations * self.inner_iterations * self.CFS_level,
                    ]
                )
            )
        else:
            self.map_size = int(
                np.ceil(
                    np.max(
                        [
                            self.iterations * self.CGS_level,
                            self.iterations * self.inner_iterations * self.CLS_level,
                            self.iterations * self.inner_iterations * self.CFS_level,
                        ]
                    )
                    / len(self.chaos_map)
                )
            )

        self.map = Chaos_map(self.chaos_map, self.map_size, self.search_space.n_variables)

        if self.verbose:
            print(str(self))

    def run(self, n_process=1):

        """run(self, n_process=1)

        Runs the Chaotic_optimization

        Parameters
        ----------
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        # Initialize CGS/CLS/CFS
        cgs = CGS(self.loss_func, self.search_space, self.f_calls, self.CGS_level, self.map)
        cls = CLS(
            self.loss_func,
            self.search_space,
            self.f_calls,
            self.CLS_level,
            self.polygon,
            self.map,
            self.red_rate,
        )
        cfs = CFS(
            self.loss_func,
            self.search_space,
            self.f_calls,
            self.CFS_level,
            self.polygon,
            self.map,
            self.red_rate,
        )

        # Initialize historic vector
        best_sol = np.array([])
        best_scores = np.array([])

        k = 1

        # Outer loop (exploration)
        while k <= self.iterations and self.loss_func.calls < self.f_calls:

            # If there is CGS
            if self.CGS_level > 0:
                x_inter, loss_value = cgs.run(k)

                # Store to return best solution found
                best_sol = np.append(best_sol, x_inter)
                best_scores = np.append(best_scores, loss_value)

            # Else select random point for the exploitation
            else:
                x_inter = [np.random.random(self.search_space.n_variables)]
                loss_value = self.loss_func.evaluate(x_inter)

                # Store to return best solution found
                best_sol = np.append(x_inter)
                best_scores = np.append(loss_value)

            if self.verbose:
                out = "\n\n=======>   Iterations | Loss function calls | Best value from CGS"
                out += "\n=======>" + str(k) + "<" + str(self.iterations) + "|" + str(self.loss_func.calls) + "<" + str(self.f_calls) + " |" + str(loss_value)
                if self.loss_func.new_best:
                    out += "\n=======> !!--!! New best solution found !!--!! "
                print(out)

            inner = 0

            # Inner loop (exploitation)
            while inner < self.inner_iterations and self.loss_func.calls < self.f_calls:

                if self.CLS_level > 0:
                    x_inter, loss_value = cls.run(x_inter[0], loss_value[0], inner, k)

                    # Store to return best solution found
                    best_sol = np.append(best_sol, x_inter)
                    best_scores = np.append(best_scores, loss_value)

                if self.CFS_level > 0:
                    x_inter, loss_value = cfs.run(x_inter[0], loss_value[0], inner, k)

                    # Store to return best solution found
                    best_sol = np.append(best_sol, x_inter)
                    best_scores = np.append(best_scores, loss_value)

                if self.verbose:
                    out = "-->" + str(inner) + "<" + str(self.inner_iterations) + "  |" + str(self.loss_func.calls) + "<" + str(self.f_calls) + " |" + str(loss_value)
                    out += "\n=======>" + str(k) + "<" + str(self.iterations) + "|" + str(self.loss_func.calls) + "<" + str(self.f_calls) + " |" + str(loss_value)
                    if self.loss_func.new_best:
                        out += "\n=======> !!--!! New best solution found !!--!! "
                    print(out)

                inner += 1

            ind_min = np.argsort(best_scores)[0:n_process]
            best_scores = np.array(best_scores)[ind_min].tolist()
            best_sol = np.array(best_sol)[ind_min].tolist()

            k += 1

        return best_sol, best_scores

    def show(self, filepath="", save=False):

        """show(self, filename="")

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        save : boolean, default=False
            Save figures
        """

        super().show(filepath, save)

    def __str__(self):
        return f"Max Loss function calls:{self.f_calls}\nDimensions:{self.search_space.n_variables}\nExploration/Exploitation:{self.exploration_ratio}|{1-self.exploration_ratio}\nRegular polygon:{self.polygon}\nZoom:{self.red_rate}\nIterations:\n\tGlobal:{self.iterations}\n\tInner:{self.inner_iterations}\nChaos Levels:\n\tCGS:{self.CGS_level}\n\tCLS:{self.CLS_level}\n\tCFS:{self.CFS_level}\nMap size:{self.map_size}x{self.search_space.n_variables}"
