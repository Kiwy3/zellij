# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-06T12:07:22+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-02T11:33:23+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from zellij.core.addons import VarNeighborhood, Neighborhood
from zellij.core.variables import (
    FloatVar,
    IntVar,
    CatVar,
    Constant,
    ArrayVar,
    Block, DynamicBlock, DAGraphVariable,
)
import numpy as np
import random
import logging
import copy

logger = logging.getLogger("zellij.neighborhoods")


class ArrayInterval(VarNeighborhood):
    def __init__(self, variable=None, neighborhood=None):
        super(ArrayInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        variables = np.random.choice(self.target.values, size=size)

        res = []

        for v in variables:
            inter = copy.deepcopy(value)
            inter[v._idx] = v.neighbor(value[v._idx])
            res.append(inter)

        return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood:
            for var, neig in zip(self.target.values, neighborhood):
                var.neighborhood = neig

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, ArrayVar) or variable is None, logger.error(
            f"Target object must be an `ArrayVar` for {self.__class__.__name__},\
             got {variable}"
        )

        self._target = variable

        if variable != None:
            assert all(
                hasattr(v, "neighbor") for v in self.target.values
            ), logger.error(
                f"To use `ArrayInterval`, values in `ArrayVar` must have a `neighbor` method. Use `neighbor` kwarg "
                f"when defining a variable "
            )


class BlockInterval(VarNeighborhood):
    def __init__(self, variable=None, neighborhood=None):
        super(BlockInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        res = []
        for _ in size:
            variables_idx = list(set(np.random.choice(range(self.target.repeat), size=self.target.repeat)))
            inter = copy.deepcopy(value)
            for i in variables_idx:
                inter[i] = self.target.value[i].neighbor(value[i])
            res.append(inter)
        return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood:
            self.target.value.neighborhood = neighborhood
        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, Block) or variable is None, logger.error(
            f"Target object must be a `Block` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

        if variable is not None:
            assert hasattr(self.target.value, "neighbor"), logger.error(
                f"To use `Block`, value for `Block` must have a `neighbor` method. Use `neighbor` kwarg "
                f"when defining a variable "
            )


class DAGraphInterval(VarNeighborhood):

    def __init__(self, variable=None, neighborhood=None):
        super(DAGraphInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        res = []
        for _ in size:
            inter = value.copy()
            for n in inter.nodes:
                n.operation = self.target.operations.value.neighborhood(n.operation)
            res.append(inter)
        return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood is not None:
            self.target.operations.value.neighborhood = neighborhood
        self._neighborhood = None


    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, DAGraphVariable) or variable is None, logger.error(
            f"Target object must be a `DAGraphVariable` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

        if variable is not None:
            assert hasattr(self.target.operations.value, "neighbor"), logger.error(
                f"To use `DAGraphVariable`, value for operations for `DAGraphVariable` must have a `neighbor` method. "
                f"Use `neighbor` kwarg when defining a variable "
            )


class Interval(VarNeighborhood):

    def __init__(self, variable=None, neighborhood=None):
        super(Interval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        res = []
        for _ in size:
            new_repeat = np.random.randint(self.target.repeat - self.neighborhood,
                                           self.target.repeat - self.neighborhood)
            inter = copy.deepcopy(value)
            if new_repeat > self.target.repeat:
                inter = inter + self.target.random(new_repeat - self.target.repeat)
            if new_repeat < self.target.repeat:
                deleted_idx = list(set(random.sample(range(self.target.repeat), self.target.repeat - new_repeat)))
                for index in sorted(deleted_idx, reverse=True):
                    del inter[index]
            variables_idx = list(set(np.random.choice(range(new_repeat), size=new_repeat)))
            for i in variables_idx:
                inter[i] = self.target.value[i].neighbor(value[i])
            res.append(inter)
        return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if isinstance(neighborhood, list):
            self._neighborhood = neighborhood[0]
            self.target.value.neighborhood = neighborhood[1]
        else:
            self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, DynamicBlock) or variable is None, logger.error(
            f"Target object must be a `DynamicBlock` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable

        if variable is not None:
            assert hasattr(self.target.value, "neighbor"), logger.error(
                f"To use `DynamicBlock`, value for `DynamicBlock` must have a `neighbor` method. Use `neighbor` kwarg "
                f"when defining a variable "
            )


class FloatInterval(VarNeighborhood):
    def __call__(self, value, size=1):
        upper = np.min([value + self.neighborhood, self.target.up_bound])
        lower = np.max([value - self.neighborhood, self.target.low_bound])

        if size > 1:
            res = []
            for _ in range(size):
                v = np.random.uniform(lower, upper)
                while v == value:
                    v = np.random.uniform(lower, upper)
                res.append(float(v))
            return res
        else:
            v = np.random.uniform(lower, upper)
            while v == value:
                v = np.random.uniform(lower, upper)

            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        assert isinstance(neighborhood, int) or isinstance(
            neighborhood, float
        ), logger.error(
            f"`neighborhood` must be a float or an int, for `FloatInterval`,\
            got{neighborhood}"
        )

        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, FloatVar) or variable == None, logger.error(
            f"Target object must be a `FloatVar` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable


class IntInterval(VarNeighborhood):
    def __call__(self, value, size=1):

        upper = np.min([value + self.neighborhood, self.target.up_bound])
        lower = np.max([value - self.neighborhood, self.target.low_bound])

        if size > 1:
            res = []
            for _ in range(size):
                v = np.random.randint(lower, upper)
                while v == value:
                    v = np.random.randint(lower, upper)
                res.append(int(v))
            return res
        else:
            v = np.random.randint(lower, upper)
            while v == value:
                v = np.random.randint(lower, upper)

            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        assert isinstance(neighborhood, int) or isinstance(
            neighborhood, float
        ), logger.error(
            f"`neighborhood` must be an int, for `IntInterval`,\
            got{neighborhood}"
        )

        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, IntVar) or variable == None, logger.error(
            f"Target object must be a `IntInterval` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable


class CatInterval(VarNeighborhood):
    def __init__(self, variable=None, neighborhood=None):
        super(CatInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        if size > 1:
            res = []
            for _ in range(size):
                v = self.target.random()
                while v == value:
                    v = self.target.random()
                res.append(v)
            return res
        else:
            v = self.target.random()
            while v == value:
                v = self.target.random()
            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood is not None:
            logger.warning(
                f"`neighborhood`= {neighborhood} is useless for \
            {self.__class__.__name__}, it will be replaced by None"
            )

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, CatVar) or variable is None, logger.error(
            f"Target object must be a `CatInterval` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable


class ConstantInterval(VarNeighborhood):
    def __init__(self, variable=None, neighborhood=None):
        super(ConstantInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        logger.warning("Calling `neighbor` of a constant is useless")
        return self.target.value

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood != None:
            logger.warning(
                f"`neighborhood`= {neighborhood} is useless for \
            {self.__class__.__name__}, it will be replaced by None"
            )

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):
        assert isinstance(variable, Constant) or variable == None, logger.error(
            f"Target object must be a `ConstantInterval` for {self.__class__.__name__}\
            , got {variable}"
        )
        self._target = variable


class Intervals(Neighborhood):
    def __init__(self, search_space=None, neighborhood=None):
        super(Intervals, self).__init__(search_space, neighborhood)

    @Neighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        if neighborhood:
            for var, neig in zip(self.target.values, neighborhood):
                var.neighbor.neighborhood = neig

        self._neighborhood = None

    @Neighborhood.target.setter
    def target(self, object):
        self._target = object
        if object:
            assert hasattr(self.target.values, "neighbor"), logger.error(
                f"To use `Intervals`, values in Searchspace must have a `neighbor` method. Use `neighbor` kwarg when defining a variable"
            )

    def get_neighbor(self, point, size=1):

        """get_neighbor(point, size=1)

        Draw a neighbor of a solution, according to the search space bounds and\
        dimensions types.

        Parameters
        ----------

        point : list
            Initial point.
        size : int, default=1
            Draw <size> neighbors of <point>.

        Returns
        -------

        out : list
            List of neighbors of <point>.

        """
        attribute = self.target.random_attribute(size=size, exclude=Constant)

        points = []

        for att in attribute:
            inter = copy.deepcopy(point)
            inter[att._idx] = att.neighbor(point[att._idx])
            points.append(inter)

        return points
