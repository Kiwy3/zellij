# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)


from __future__ import annotations
from zellij.core.errors import InputError, InitializationError
from zellij.core.metaheuristic import UnitMetaheuristic, MonoObjective

from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import UnitSearchspace

import torch
import numpy as np

import gpytorch

from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from botorch.utils import standardize
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError


if TYPE_CHECKING:
    from zellij.strategies.tools.tree_search import TreeSearch
    from zellij.strategies.tools.scoring import Scoring
    from zellij.strategies.fractals import Sampling
    from zellij.core.search_space import BaseFractal

from torch.quasirandom import SobolEngine

import logging

logger = logging.getLogger("zellij.BO")


class BaMSOO(UnitMetaheuristic, MonoObjective):
    """BaMSOO

    Bayesian optimization (BO) is a surrogate based optimization method which
    interpolates the actual loss function with a surrogate model, here a
    gaussian process.
    It is based on `BoTorch <https://botorch.org/>`_ and `GPyTorch <https://gpytorch.ai/>`__.

    Attributes
    ----------
    search_space : UnitSearchspace
        Search space object containing bounds of the search space
    tree_search : Tree_search
            Tree search algorithm applied on the partition tree.
    sampling : Sampling
        A :code:`Sampling` object.
    scoring : Scoring
        Function that defines how promising a space is according to sampled
        points.
    verbose : bool
        If False, there will be no print.
    surrogate : botorch.models.model.Model, default=SingleTaskGP
        Gaussian Process Regressor object from 'botorch'.
        Determines the surrogate model that Bayesian optimization will use to
        interpolate the loss function
    likelihood : gpytorch.mlls, default=ExactMarginalLogLikelihood
        gpytorch.mlls object it determines which MarginalLogLikelihood to use
        when optimizing kernel's hyperparameters
    acquisition : botorch.acquisition.acquisition.AcquisitionFunction, default = ExpectedImprovement
        An acquisition function or infill criteria, determines how 'promising'
        a point sampled from the surrogate is.
    initial_size : int, default=10
        Size of the initial set of solution to draw randomly.
    gpu: bool, default=True
        Use GPU if available
    kwargs
        Key word arguments linked to the surrogate and the acquisition function.

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space: UnitSearchspace,
        tree_search: TreeSearch,
        sampling: Sampling,
        scoring: Scoring,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        initial_size: int = 10,
        gpu: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        """__init__

        Parameters
        ----------
        search_space : ContinuousSearchspace
            Search space object containing bounds of the search space
        verbose : bool
            If False, there will be no print.
        surrogate : botorch.models.model.Model, default=SingleTaskGP
            Gaussian Process Regressor object from 'botorch'.
            Determines the surrogate model that Bayesian optimization will use to
            interpolate the loss function
        mll : gpytorch.mlls, default=ExactMarginalLogLikelihood
            Object from gpytorch.mlls it determines which marginal loglikelihood to use
            when optimizing kernel's hyperparameters
        likelihood : gpytorch.likelihoods, default=GaussianLikelihood
            Object from gpytorch.likelihoods defining the likelihood.
        acquisition : botorch.acquisition.acquisition.AcquisitionFunction, default = ExpectedImprovement
            An acquisition function or infill criteria, determines how 'promising'
            a point sampled from the surrogate is.
        initial_size : int, default=10
            Size of the initial set of solution to draw randomly.
        gpu: bool, default=True
            Use GPU if available
        kwargs
            Key word arguments linked to the surrogate and the acquisition function.

        """

        super().__init__(search_space=search_space, verbose=verbose)

        ##################
        # DBA PARAMETERS #
        ##################

        self.tree_search = tree_search
        self.sampling = sampling
        self.scoring = scoring

        #################
        # BO PARAMETERS #
        #################

        self.acquisition = UpperConfidenceBound
        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood
        self.initial_size = initial_size

        self.kwargs = kwargs

        ################
        # BO VARIABLES #
        ################

        # Prior points
        self.train_x = torch.empty((0, self.search_space.size))
        self.train_obj = torch.empty((0, 1))
        self.state_dict = {}

        # Determine if BO is initialized or not
        self.initialized = False

        # Number of iterations
        self.iterations = 0

        if gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                logger.warning(f"No GPU available for BaMSOO.")
        else:
            self.device = torch.device("cpu")

        self.dtype = torch.double

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)
        self._build_kwargs()

        #################
        # DBA VARIABLES #
        #################

        self.info = []
        self.default_info = {}
        self.xinfo = ["fracid"]

        # Children fractals
        self.children = []
        self.parents = []
        self._fidx = []

        self.extract_sample_info = []
        self.extract_sample_xinfo = []
        self.order_sample_info = []
        self.order_sample_xinfo = []

        for i, k in enumerate(self.info):
            if self.sampling.info and k in self.sampling.info:
                self.extract_sample_info.append(i)
                self.order_sample_info.append(self.sampling.info.index(k))

        for i, k in enumerate(self.xinfo):
            if self.sampling.xinfo and k in self.sampling.xinfo:
                self.extract_sample_xinfo.append(i)
                self.order_sample_xinfo.append(self.sampling.xinfo.index(k))

    @property
    def search_space(self) -> BaseFractal:
        return self._search_space

    @search_space.setter
    def search_space(self, value: BaseFractal):
        if isinstance(value, BaseFractal):
            self._search_space = value
        else:
            raise InitializationError(
                f"Searchspace in DBA must be of type BaseFractal. Got {type(value).__name__}."
            )

    @property
    def sampling(self) -> Sampling:
        return self._sampling

    @sampling.setter
    def sampling(self, value: Sampling):
        if value:  # If there is sampling
            self._sampling = value
            if self._sampling.info is not None:
                self.info = list(set(self.info + self._sampling.info))
                self.default_info = dict.fromkeys(self.info, np.nan)
            if self._sampling.xinfo is not None:
                self.xinfo = list(set(self.xinfo + self._sampling.xinfo))
        else:
            raise InitializationError(f"DBA must implement at least an exploration.")

    # Add more info to ouputs
    def _add_info(self, info: dict) -> dict:
        info = self.default_info | info
        return info

    def _add_xinfo(self, xinfo: dict, npoints: int) -> dict:
        if len(self.xinfo) > 0:
            default_xinfo = dict.fromkeys(self.xinfo, np.full(npoints, np.nan))
            xinfo = default_xinfo | xinfo
        return xinfo

    def _sample(
        self,
        sample: Sampling,
        X: Optional[list],
        Y: Optional[np.ndarray],
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ):
        points, info_dict, xinfo_dict = sample.forward(X, Y, constraint, info, xinfo)
        if len(points) > 0:
            info_dict = self._add_info(info_dict)
            xinfo_dict = self._add_xinfo(xinfo_dict, len(points))
            return points, info_dict, xinfo_dict  # Continue exploration
        else:
            return [], {"algorithm": "EndSample"}, {}  # Exploration ending

    def _initialize_model(
        self,
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        state_dict: Optional[dict] = None,
    ):
        train_x.to(self.device, dtype=self.dtype)
        train_obj.to(self.device, dtype=self.dtype)

        likelihood = self.likelihood(**self.likelihood_kwargs)

        # define models for objective and constraint
        model = self.surrogate(
            train_x,
            train_obj,
            likelihood=likelihood,
            **self.model_kwargs,
        )

        mll = self.mll(
            model.likelihood,
            model,
            **self.mll_kwargs,
        )

        # load state dict if it is passed
        if state_dict:
            model.load_state_dict(state_dict)

        model.to(self.device)

        return mll, model

    def get_posterior(self, X, model):
        pass

    def _build_kwargs(self):
        # Surrogate kwargs
        self.model_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.surrogate.__init__.__code__.co_varnames
        }

        for m in self.model_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # Likelihood kwargs
        self.likelihood_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.likelihood.__init__.__code__.co_varnames
        }
        for m in self.likelihood_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # MLL kwargs
        self.mll_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.mll.__init__.__code__.co_varnames
        }
        for m in self.mll_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # Acquisition function kwargs
        self.acqf_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.acquisition.__init__.__code__.co_varnames
        }
        for m in self.acqf_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        logger.debug(self.model_kwargs, self.likelihood_kwargs, self.mll_kwargs)

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty((0, self.search_space.size))
        self.train_obj = torch.empty((0, 1))
        self.state_dict = {}

        self.n_h = 0
        self.current_calls = 0

    def forward(
        self,
        X: Optional[list],
        Y: Optional[np.ndarray],
        constraint: Optional[np.ndarray],
        info: Optional[np.ndarray],
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """forward

        Abstract method describing one step of the :ref:`meta`.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Dictionnary of additionnal information linked to :code:`points`.
        """

        self.sampling.reset()

        if info is None:
            sample_info = None
        else:
            sample_info = info[self.extract_sample_info][self.order_sample_info]

        if xinfo is None:
            sample_xinfo = None
        else:
            sample_xinfo = xinfo[:, self.extract_sample_xinfo][
                :, self.order_sample_xinfo
            ]

        if X is not None and Y is not None and sample_xinfo is not None:

            npx = np.array(X)
            npy = np.array(Y)
            mask = np.isfinite(npy)

            mx = npx[mask]
            my = npy[mask]

            if len(my) > 0:
                new_x = torch.tensor(mx, dtype=self.dtype)
                new_obj = torch.tensor(my, dtype=self.dtype).unsqueeze(-1)

                # update training points
                self.train_x = torch.cat([self.train_x, new_x])
                self.train_obj = torch.cat([self.train_obj, new_obj])

                train_obj_std = -standardize(self.train_obj)

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                mll, model = self._initialize_model(
                    self.train_x,
                    train_obj_std,
                    self.state_dict,
                )

                self.state_dict = model.state_dict()

            if isinstance(self.sampling.search_space, list):
                for x, y, fracid in zip(X, Y, sample_xinfo[:, 0]):
                    self.sampling.search_space[int(fracid)].add_solutions(x, y)
                    self.parents[self._fidx[int(fracid)]].add_solutions(x, y)
            else:
                self.sampling.search_space.add_solutions(X, Y)
                self.parents[0].add_solutions(X, Y)

            for f in self.parents:
                self.scoring(f)

            for fracid in sample_xinfo[:, 0]:
                c = self.sampling.search_space[int(fracid)]
                f = self.parents[self._fidx[int(fracid)]]
                self.scoring(c)
                c.var = 0
                self.tree_search.add(c)

        self.iteration += 1

        try:
            with gpytorch.settings.max_cholesky_size(300):
                # run N_BATCH rounds of BayesOpt after the initial random batch
                # fit the models
                fit_gpytorch_mll(mll)

                # Add potentially usefull kwargs for acqf kwargs
                self.acqf_kwargs["best_f"] = torch.max(train_obj_std)
                if "X_baseline" in self.acquisition.__init__.__code__.co_varnames:
                    self.acqf_kwargs["X_baseline"] = (self.train_x,)

                # Build acqf kwargs
                acqf = self.acquisition(model=model, **self.acqf_kwargs)

                # optimize and get new observation
                new_x, acqf_value = self._optimize_acqf_and_get_observation(acqf)

                return (
                    new_x.cpu().numpy().tolist(),
                    {
                        "acquisition": acqf_value.cpu().item(),
                        "algorithm": "BO",
                    },
                    {},
                )
        except ModelFittingError:
            new_x = self._generate_initial_data(1)
            return (
                new_x.cpu().numpy().tolist(),
                {
                    "acquisition": 0,
                    "algorithm": "ModelFittingError",
                },
                {},
            )
