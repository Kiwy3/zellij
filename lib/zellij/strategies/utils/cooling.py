import numpy as np
from abc import abstractmethod


class Cooling(object):
    """Cooling

    Cooling is a base object which define what a cooling Schedule is.

    Parameters
    ----------
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------

    Tcurrent : float
        Current temperature

    cross : int
        Count the number of times Tend is crossed.

    T0
    Tend
    peaks

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, T0, Tend, peaks=1):

        ##############
        # PARAMETERS #
        ##############

        self.T0 = T0
        self.Tend = Tend
        self.peaks = peaks

        #############
        # VARIABLES #
        #############
        self.Tcurrent = self.T0
        self.k = 0
        self.cross = 0

    @abstractmethod
    def cool(self):
        pass

    def reset(self):
        self.Tcurrent = 0
        self.k = 0
        self.cross = 0

    @abstractmethod
    def iterations(self):
        pass


class MulExponential(Cooling):
    """MulExponential

    Exponential multiplicative monotonic cooling.
    Tk = T0.alpha^k

    Parameters
    ----------
    alpha : float
        Decrease factor. 0.8<=`alpha`<=0.9
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    alpha

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, alpha, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    def cool(self):

        self.Tcurrent = self.T0 * self.alpha ** self.k

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return int(np.ceil(np.log(self.Tend / self.T0) / np.log(self.alpha))) * self.peaks


class MulLogarithmic(Cooling):
    """MulLogarithmic

    Logarithmic multiplicative monotonic cooling.
    Tk = T0/(1+alpha.log(1+k))

    Parameters
    ----------
    alpha : float
        Decrease factor. `alpha`>1
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    alpha

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, alpha, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    def cool(self):

        self.Tcurrent = self.T0 / (1 + self.alpha * np.log(1 + self.k))

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return int(np.ceil(np.exp((self.T0 / self.Tend - 1 / self.alpha)) - 1)) * self.peaks


class MulLinear(Cooling):
    """MulLinear

    Linear multiplicative monotonic cooling.
    Tk = T0/(1+alpha.k)

    Parameters
    ----------
    alpha : float
        Decrease factor. `alpha`>0
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    alpha

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, alpha, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    def cool(self):

        self.Tcurrent = self.T0 / (1 + self.alpha * self.k)

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return int(np.ceil(self.T0 / (self.Tend * self.alpha))) * self.peaks


class MulQuadratic(Cooling):
    """MulQuadratic

    Quadratic multiplicative monotonic cooling.
    Tk = T0/(1+alpha.k^2)

    Parameters
    ----------
    alpha : float
        Decrease factor. `alpha`>0
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    alpha

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, alpha, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    def cool(self):

        self.Tcurrent = self.T0 / (1 + self.alpha * self.k ** 2)

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return int(np.ceil(np.sqrt(self.T0 / (self.Tend * self.alpha)))) * self.peaks


class AddLinear(Cooling):
    """AddLinear

    Linear additive monotonic cooling.
    Tk = Tend + (T0-Tend)((cycles-k))/cycles)

    Parameters
    ----------
    cycles : int
        Number of cooling cycles.

    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    cycles

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles

    def cool(self):

        self.Tcurrent = self.Tend + (self.T0 - self.Tend) * ((self.cycles - self.k) / self.cycles)

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks


class AddQuadratic(Cooling):
    """AddQuadratic

    Quadratic additive monotonic cooling.
    Tk = Tend + (T0-Tend)((cycles-k))/cycles)^2

    Parameters
    ----------
    cycles : int
        Number of cooling cycles.

    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    cycles

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles

    def cool(self):

        self.Tcurrent = self.Tend + (self.T0 - self.Tend) * ((self.cycles - self.k) / self.cycles) ** 2

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks


class AddExponential(Cooling):
    """AddExponential

    Exponential additive monotonic cooling.
    Tk = Tend + (T0-Tend)(1/(1+exp(2*ln(T0-Tend)/n*(k-1/2*n))))

    Parameters
    ----------
    cycles : int
        Number of cooling cycles.

    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    cycles

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles

    def cool(self):

        self.Tcurrent = self.Tend + (self.T0 - self.Tend) * (1 / (1 + np.exp(2 * np.log(self.T0 - self.Tend) / self.cycles * (self.k - 0.5 * self.cycles))))

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks


class AddTrigonometric(Cooling):
    """AddExponential

    Trigonometric additive monotonic cooling.
    Tk = Tend + 0,5.(T0-Tend)(1+cos(k.pi/cycles))

    Parameters
    ----------
    cycles : int
        Number of cooling cycles.

    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    cycles

    Methods
    -------

    cool()
        Decrease temperature and return the current temperature.

    reset()
        Reset cooling schedule

    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles

    def cool(self):

        self.Tcurrent = self.Tend + 0.5 * (self.T0 - self.Tend) * (1 + np.cos(self.k * np.pi / self.cycles))

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks
