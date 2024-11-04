# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
import os
import time
from datetime import datetime
import logging
from collections.abc import Callable
from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.objective import Objective
    from zellij.core.stop import Stopping

from queue import Queue
import numpy as np

from zellij.core.errors import ModelError, OutputError, DimensionalityError, InputError

logger = logging.getLogger("zellij.loss")

try:
    from mpi4py import MPI
except ImportError as err:
    logger.info(
        "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[MPI]"
    )


class LossFunc(ABC):
    """LossFunc

    LossFunc allows to wrap function of type :math:`f(x)=dict`.
    You must wrap your function so it can be used in Zellij by adding
    several features, such as calls count, saves, parallelization...

    Attributes
    ----------
    model : Callable[..., dict]
        Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
        of hyperparameters for example.
        And :math:`y` can be a single value, a list, a tuple, or a dict,
        containing the loss value and other optionnal information.
        It can also be of mixed types, containing, strings, float, int...
    objective : Objective
        An :code:`Objective` object determines what the optimization problem is.
    constraint : list[str], default=None
        List of keys linked to :ref:`lf` outputs.
        Constraints are modeled by inequation. The constraint is met if the value is
        strictly inferior to zero.
        If a list of strings is passed, constraints values will be passed to
        the :code:`forward` method of :ref:`meta`.
    default : dict, optionnal
        Dictionnary of defaults arguments, kwargs, to pass to the loss function.
        They are not affected by any :ref:`meta` or other methods.
    record_time : boolean, default=False
        If True, :code:`start_time`, :code:`end_time`, :code:`start_date`, :code:`end_date` will be recorded
        and saved in the save file for each :code:`__call__`.
    only_score : bool, default=False
        If True, then only the score of evaluated solutions are saved.
        Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
        saved.
    kwargs_mode : bool, default=False
        If True, then solutions are passed as kwargs to :ref:`lf`. Keys, are
        the names of the :ref:`var` within the :ref:`sp`.
    best_score : np.ndarray
        Best score found so far.
    best_point : list
        Best solution found so far.
    calls : int
        Number of loss function calls

    See Also
    --------
    Loss : Wrapper function
    MPILoss : Distributed version of LossFunc
    SerialLoss : Basic version of LossFunc
    """

    def __init__(
        self,
        model: Callable[..., dict],
        objective: List[Objective],
        constraint: Optional[list[str]] = None,
        record_time: bool = False,
        only_score: bool = False,
        kwargs_mode: bool = False,
        default: Optional[dict] = None,
    ):
        """__init__

        Initialize LossFunc

        Parameters
        ----------
        model : Callable
            Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
            of hyperparameters for example.
            And :math:`y` can be a single value, a list, a tuple, or a dict,
            containing the loss value and other optionnal information.
            It can also be of mixed types, containing, strings, float, int...
        objective : Objective
            An :code:`Objective` object determines what the optimization problem is.
        constraint : list[str], default=None
            List of keys linked to :ref:`lf` outputs.
            Constraints are modeled by inequation. The constraint is met if the value is
            strictly inferior to zero.
            If a list of strings is passed, constraints values will be passed to
            the :code:`forward` method of :ref:`meta`.
        default : dict, optionnal
            Dictionnary of defaults arguments, kwargs, to pass to the loss function.
            They are not affected by any :ref:`meta` or other methods.
        record_time : boolean, default=False
            If True, :code:`start_time`, :code:`end_time`, :code:`start_date`, :code:`end_date` will be recorded
            and saved in the save file for each :code:`__call__`.
        only_score : bool, default=False
            If True, then only the score of evaluated solutions are saved.
            Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
            saved.
        kwargs_mode : bool, default=False
            If True, then solutions are passed as kwargs to :ref:`lf`. Keys, are
            the names of the :ref:`var` within the :ref:`sp`.
        """
        ##############
        # PARAMETERS #
        ##############
        self.model = model

        self.objective = objective
        self.constraint = constraint

        self.save = None
        self.only_score = only_score
        self.record_time = record_time

        self.kwargs_mode = kwargs_mode
        self.default = default

        #############
        # VARIABLES #
        #############

        self.info = None
        self.xinfo = None

        self.calls = 0

        self.labels = []

        self.outputs_path = ""
        self.model_path = ""
        self.plots_path = ""
        self.loss_file = ""

        self.folder_created = False
        self.file_created = False

        self._init_time = time.time()

    @property
    def objective(self) -> List[Objective]:
        return self._objective

    @objective.setter
    def objective(self, value: List[Objective]):
        self._objective = value
        if isinstance(value, list) and len(value) > 0:
            self.best_score = np.array([float("inf")] * len(value), dtype=float)
            self.best_point = [None] * len(value)
        else:
            raise InputError(
                f"Objective must be a list containing objectives, got {value} "
            )

    @property
    def constraint(self) -> Optional[List[str]]:
        return self._constraint

    @constraint.setter
    def constraint(self, value: Optional[List[str]]):
        self._constraint = value
        if value:
            self.best_constraint = np.array(
                [[float("inf")] * len(value)] * len(self.objective), dtype=float
            )
        else:
            self.best_constraint = [None] * len(self.objective)

    @property
    def info(self) -> Optional[List[str]]:
        return self._info

    @info.setter
    def info(self, value: Optional[List[str]]):
        self._info = value
        if value:
            self.best_info = np.array(
                [[float("inf")] * len(value)] * len(self.objective), dtype=float
            )
        else:
            self.best_info = [None] * len(self.objective)

    @property
    def xinfo(self) -> Optional[List[str]]:
        return self._xinfo

    @xinfo.setter
    def xinfo(self, value: Optional[List[str]]):
        self._xinfo = value
        if value:
            self.best_xinfo = np.array(
                [[float("inf")] * len(value)] * len(self.objective), dtype=float
            )
        else:
            self.best_xinfo = [None] * len(self.objective)

    @property
    def save(self) -> bool:
        return self._save

    @save.setter
    def save(self, value: Optional[str]):
        if value:
            self._save = True
            self.folder_name = value
        else:
            self._save = False
            self.folder_name = ""

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]
        del state["_init_time"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        logger.warning("In Loss, after unpickling, the `model` has to be set manually.")
        self._init_time = time.time()

    @abstractmethod
    def _save_model(self, *args):
        """_save_model

        Private abstract method to save a model.
        The dictionnary of outputs from the model must have a 'model' key
        with a model object having a "save" method.
        (model.save(filename)).
        """
        pass

    @abstractmethod
    def __call__(
        self,
        X: list,
        stop_obj: Optional[Stopping] = None,
        info: Optional[dict] = None,
        xinfo: Optional[dict] = None,
    ) -> Tuple[
        list,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        pass

    def _compute_loss(self, point: list):
        if self.kwargs_mode:
            new_kwargs = {key: value for key, value in zip(self.labels, point)}  # type: ignore
            if self.default:
                new_kwargs.update(self.default)

            if self.record_time:
                start = time.time()
                start_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

            # compute loss
            res, trained_model = self._build_return(self.model(**new_kwargs))  # type: ignore

            if self.record_time:
                end = time.time()
                end_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                res["eval_time"] = end - start
                res["start_time"] = start - self._init_time
                res["end_time"] = end - self._init_time
                res["start_date"] = start_date
                res["end_date"] = end_date
        else:
            if self.record_time:
                start = time.time()
                start_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

            if self.default:
                lossv = self.model(point, **self.default)
            else:
                lossv = self.model(point)

            res, trained_model = self._build_return(lossv)

            if self.record_time:
                end = time.time()
                end_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                res["eval_time"] = end - start
                res["start_time"] = start - self._init_time
                res["end_time"] = end - self._init_time
                res["start_date"] = start_date
                res["end_date"] = end_date

        return res, trained_model

    def _create_subfolder(self, path: str, name: str):
        """_create_subfolder

        Private.
        Create folder of a given path. Name is used for Error.

        Parameters
        ----------
        path : str
            Valid path.
        name : str
            Step name of the folder creation.

        Raises
        ------
        FileExistsError
        """
        # Create a valid folder
        try:
            os.makedirs(path)
        except FileExistsError as error:
            raise FileExistsError(
                f"{name} folder already exists, got {path}. The experiment will end. Try another folder name to save your experiment."
            )

    def _create_folder(self):
        """create_file

        Create a save file.

        Parameters
        ----------
        x : list
           A given point/solution to determine the header of the save file.
        args : list[str], optionnal
            Additionnal info to add after the score/evaluation of a point.
        """

        # Create a valid folder
        self._create_subfolder(self.folder_name, "Experiment")
        # Create ouputs folder
        self.outputs_path = os.path.join(self.folder_name, "outputs")
        self._create_subfolder(self.outputs_path, "Outputs")
        # Create a model folder
        self.model_path = os.path.join(self.folder_name, "model")
        self._create_subfolder(self.model_path, "Models")
        # Create plots path
        self.plots_path = os.path.join(self.folder_name, "plots")
        self._create_subfolder(self.plots_path, "Plots")

        self.folder_created = True

    def _create_file(self, x: list, *args):
        if not self.folder_created:
            self._create_folder()

        # Additionnal header for the outputs file
        if self.only_score:
            suffix = ",".join(
                str(f"objective{idx}") for idx in range(0, len(self.objective))
            )
            if self.constraint:
                suffix += "," + ",".join(str(f"{con}") for con in self.constraint)
        else:
            if len(args) > 0:
                suffix = "," + ",".join(str(e) for e in args)
            else:
                suffix = ""

        # Create base outputs file for loss func
        self.loss_file = os.path.join(self.outputs_path, "all_evaluations.csv")

        # Determine header
        if len(self.labels) != len(x):
            raise DimensionalityError(
                f"WARNING: Dimensionality mismatch between Search space and a solution passed to LossFunc during file creation. Got dim={len(self.labels)} and x={x}."
            )

        with open(self.loss_file, "w") as f:
            if self.only_score:
                f.write(f"{suffix}\n")
            else:
                f.write(",".join(str(e) for e in self.labels) + suffix + "\n")

        logger.info(
            f"INFO: Results will be saved at: {os.path.abspath(self.folder_name)}"
        )

        self.file_created = True

    def _save_file(self, x: list, **kwargs):
        """_save_file

        Private method to save information about an evaluation of the loss function.

        Parameters
        ----------
        x : list
            Solution to save.
        kwargs : dict, optional
            Other information to save.
        """

        if not self.file_created:
            self._create_file(x, *list(kwargs.keys()))

        # Determine if additionnal contents must be added to the save
        if len(kwargs) > 0:
            suffix = ",".join(str(e) for e in kwargs.values())
        else:
            suffix = ""

        # Save a solution and additionnal contents
        with open(self.loss_file, "a+") as f:
            if self.only_score:
                header = ",".join(
                    str(kwargs[f"objective{idx}"])
                    for idx in range(0, len(self.objective))
                )
                if self.constraint:
                    header += "," + ",".join(
                        str(kwargs[f"{con}"]) for con in self.constraint
                    )

                f.write(f"{header}\n")
            else:
                f.write(",".join(str(e) for e in x) + "," + suffix + "\n")

    # Save best found solution
    def _save_best(
        self,
        x: list,
        y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ):
        """_save_best

        Save point :code:`x` with score :code:`y`, and verify if this point is the best found so far.

        Parameters
        ----------
        x : list
            Set of hyperparameters (a solution)
        y : np.ndarray(float)
            Objective values associated to x.
        constraint : np.ndarray, optional
            Constraints values.
        info : np.ndarray, optional
            Info values.
        """

        if self.constraint:
            if constraint is None:
                raise InputError(
                    f"In LossFunc, constraints values are expected. Got {constraint} "
                )
            elif (constraint <= 0).all():
                summed = -float("inf")
            else:
                summed = constraint.clip(min=0).sum()

        for i in range(len(self.objective)):
            sc_bool = y[i] < self.best_score[i]
            if self.constraint:
                if np.isfinite(summed):
                    bsummed = np.sum(self.best_constraint[i].clip(min=0))  # type: ignore
                    if summed < bsummed and sc_bool:
                        self.best_score[i] = y[i]
                        self.best_point[i] = list(x)[:]  # type: ignore
                        self.best_constraint[i] = constraint  # type: ignore
                        self.best_info[i] = info  # type: ignore
                        self.best_xinfo[i] = xinfo  # type: ignore
                else:
                    self.best_score[i] = y[i]
                    self.best_point[i] = list(x)[:]  # type: ignore
                    self.best_constraint[i] = constraint  # type: ignore
                    self.best_info[i] = info  # type: ignore
                    self.best_xinfo[i] = xinfo  # type: ignore

            elif sc_bool:
                self.best_score[i] = y[i]
                self.best_point[i] = list(x)[:]  # type: ignore
                self.best_info[i] = info  # type: ignore
                self.best_xinfo[i] = xinfo  # type: ignore

        self.calls += 1

    def _build_return(self, r: dict) -> Tuple[dict, object]:
        """_build_return

        This method builds a unique return according to the outputs of the loss function

        Parameters
        ----------
        r : dict
            Returns of the loss function

        Returns
        -------
        outputs : dict
            Dictionnary mapping outputs from the loss function
        model : object
            Model object with a 'save' method
        """

        # Separate results and model
        model = r.pop("model", None)
        if model:
            save_mth = getattr(model, "save", None)
            if not callable(save_mth):
                raise ModelError("Returned model does not have a save method.")
        outputs = r
        for idx, obj in enumerate(self.objective):
            outputs = obj(outputs, num=idx)

        return outputs, model

    def reset(self):
        """reset

        Reset all attributes of :code:`LossFunc` at their initial values.
        """

        self.secondary = self.secondary
        self.constraint = self.constraint
        self.info = self.info

        self.calls = 0

        self.outputs_path = ""
        self.model_path = ""
        self.plots_path = ""
        self.loss_file = ""

        self.folder_created = False
        self.file_created = False

        self._init_time = time.time()


class SequentialLoss(LossFunc):
    """SequentialLoss

    SequentialLoss adds methods to save and evaluate the original loss function.

    See Also
    --------
    Loss : Wrapper function
    LossFunc : Inherited class
    MPILoss : Distributed version of LossFunc
    """

    def __init__(
        self,
        model: Callable[..., dict],
        objective: List[Objective],
        constraint: Optional[list[str]] = None,
        default: Optional[dict] = None,
        record_time: bool = False,
        only_score: bool = False,
        kwargs_mode: bool = False,
    ):
        """__init__

        Initialize SequentialLoss.

        """

        super().__init__(
            model=model,
            objective=objective,
            constraint=constraint,
            default=default,
            record_time=record_time,
            only_score=only_score,
            kwargs_mode=kwargs_mode,
        )

    def __call__(
        self,
        X: list,
        stop_obj: Optional[Stopping] = None,
        info: dict = {},
        xinfo: dict = {},
    ) -> Tuple[
        list,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """__call__

        Evaluate a list X of solutions with the original loss function.

        Parameters
        ----------
        X : list
            List of solutions to evaluate. be carefull if a solution is a list X must be a list of lists.
        info : dict, optional
            Additionnal informations to save before the score.
        xinfo : dict, optional
            Point wise additionnal informations to save before the score.

        Returns
        -------
        res : list
            Return a list of all the scores corresponding to each evaluated solution of X.

        """
        # Define secondaary stop
        if stop_obj:
            stopping = stop_obj
        else:
            stopping = lambda *args: False

        # Define outputs
        y = np.ones((len(X), len(self.objective)), dtype=float)

        if self.constraint:
            list_constraint = np.ones((len(X), len(self.constraint)), dtype=float)
        else:
            list_constraint = None

        if self.info:
            list_info = np.ones((len(X), len(self.info)), dtype=float)
        else:
            list_info = None

        if self.xinfo:
            list_xinfo = np.ones((len(X), len(self.xinfo)), dtype=float)
            dict_xinfo = dict.fromkeys(self.xinfo, 0.0)
        else:
            list_xinfo = None
            dict_xinfo = {}

        idx = 0
        while idx < len(X) and not stopping():
            x = X[idx]
            outputs, model = self._compute_loss(x)

            for i in range(len(self.objective)):
                y[idx][i] = outputs[f"objective{i}"]

            if self.constraint:
                list_constraint[idx] = [outputs[k] for k in self.constraint]  # type: ignore
                current_con = list_constraint[idx]  # type: ignore
            else:
                current_con = None

            if self.info:
                for i, k in enumerate(self.info):
                    list_info[idx][i] = info[k]  # type: ignore
                current_info = list_info[idx]  # type: ignore
            else:
                current_info = None

            if self.xinfo:
                for i, k in enumerate(self.xinfo):
                    list_xinfo[idx][i] = xinfo[k][idx]  # type: ignore
                    dict_xinfo[k] = xinfo[k][idx]  # type: ignore
                current_xinfo = list_xinfo[idx]  # type: ignore
            else:
                current_xinfo = None
                dict_xinfo = {}

            # Saving
            if self.save:
                # Save model into a file if it is better than the best found one
                if model:
                    self._save_model(self.calls, model)

                # Save score and solution into a file
                self._save_file(x, **outputs, id=self.calls, **info, **dict_xinfo)

            self._save_best(
                x,
                y[idx],
                constraint=current_con,
                info=current_info,
                xinfo=current_xinfo,
            )

            idx += 1

        return X, y, list_constraint, list_info, list_xinfo

    def _save_model(self, mid: int, trained_model: object):
        # Save model into a file if it is better than the best found one
        save_path = os.path.join(
            self.model_path, f"{self.model.__class__.__name__}_{mid}"
        )
        os.system(f"rm -rf {save_path}")
        trained_model.save(save_path)  # type: ignore


class MPILoss(LossFunc):
    def __init__(
        self,
        model: Callable[..., dict],
        objective: List[Objective],
        constraint: Optional[list[str]] = None,
        default: Optional[dict] = None,
        record_time: bool = False,
        only_score: bool = False,
        kwargs_mode: bool = False,
        strategy="synchronous",
        workers=None,
        **kwargs,
    ):
        """MPILoss

        MPILoss adds method to dynamically distribute the evaluation
        of multiple solutions within a distributed environment, where a version of
        `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`__
        is available.

        Attributes
        ----------
        model : Callable[..., dict]
            Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
            of hyperparameters for example.
            And :math:`y` can be a single value, a list, a tuple, or a dict,
            containing the loss value and other optionnal information.
            It can also be of mixed types, containing, strings, float, int...
        objective : List[Objective]
            A list of :code:`Objective` object determines what the optimization problem is.
            Mono-objective if the list contains only one :code:`Objective`, otherwise multi-objective.
        constraint : list[str], default=None
            List of keys linked to :ref:`lf` outputs.
            Constraints are modeled by inequation. The constraint is met if the value is
            strictly inferior to zero.
            If a list of strings is passed, constraints values will be passed to
            the :code:`forward` method of :ref:`meta`.
        default : dict, optionnal
            Dictionnary of defaults arguments, kwargs, to pass to the loss function.
            They are not affected by any :ref:`meta` or other methods.
        save : str, optionnal
            If a :code:`str` is given, then outputs will be saved in :code:`save`.
        record_time : boolean, default=False
            If True, :code:`start_time`, :code:`end_time`, :code:`start_date`, :code:`end_date` will be recorded
            and saved in the save file for each :code:`__call__`.
        only_score : bool, default=False
            If True, then only the score of evaluated solutions are saved.
            Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
            saved.
        kwargs_mode : bool, default=False
            If True, then solutions are passed as kwargs to :ref:`lf`. Keys, are
            the names of the :ref:`var` within the :ref:`sp`.
        strategy : str, default=synchronous
            if :code:`strategy='synchronous`: then :code:`__call__` will return all results from all
            solutions passed, once all of them have been evaluated.
            if :code:`strategy='asynchronous`: then :code:`__call__` will return
            the result from an evaluation of a solution assoon as it receives a
            result from a worker. Other solutions, are still being evaluated in
            background.
            if :code:`strategy='flexible`: then :code:`__call__` will return
            all computed results, only if the number of remaining uncomputed solutions
            is below a certain threshold. Pass: :code:`threshold=int` kwarg, to :code:`Loss`
            or :code:`MPILoss`.
        workers : int, optionnal
            Number of workers among the total number of processes spawned by
            MPI. At least, one process is dedicated to the master.
        comm : MPI_COMM_WORLD
            All created processes and their communication context are grouped in comm.
        status : MPI_Status
            Data structure containing information about a received message.
        rank : int
            Process rank
        p : int
            comm size
        master : boolean
            If True the process is the master, else it is the worker.
        kwargs : dict
            Arguments of parallel strategy.

        See Also
        --------
        Loss : Wrapper function
        LossFunc : Inherited class
        SequentialLoss : Basic version of LossFunc
        """
        #################
        # MPI VARIABLES #
        #################

        try:
            self.comm = MPI.COMM_WORLD
            self.status = MPI.Status()
            self.p_name = MPI.Get_processor_name()

            self.rank = self.comm.Get_rank()
            self.p = self.comm.Get_size()
        except Exception as err:
            logger.error(
                """To use MPILoss object you need to install mpi4py and an MPI
                distribution.\nYou can use: pip install zellij[Parallel]"""
            )

            raise err

        ##############
        # PARAMETERS #
        ##############

        super().__init__(
            model=model,
            objective=objective,
            constraint=constraint,
            default=default,
            record_time=record_time,
            only_score=only_score,
            kwargs_mode=kwargs_mode,
        )

        if workers:
            self.workers_size = workers
        else:
            self.workers_size = self.p - 1

        self.recv_msg = 0
        self.sent_msg = 0

        self._personnal_folder = os.path.join(
            os.path.join(self.folder_name, "tmp_wks"), f"worker{self.rank}"
        )

        self.strat_name = strategy

        # Strategy kwargs
        self.skwargs = kwargs

        self.pqueue = Queue()

        # list of idle workers
        self.idle = list(range(1, self.workers_size + 1))

        # loss worker rank : (point, id, info, xinfo, source, point_id within batch)
        self.p_historic = {
            i: [None, None, None, None, None, None]
            for i in range(1, self.workers_size + 1)
        }  # historic of points sent to workers

        self._strategy = None  # set to None for definition issue

        self.is_master = self.rank == 0
        self.is_worker = self.rank != 0

        # Property, defines parallelisation strategy
        self._master_rank = 0  # type: ignore

    @property
    def save(self) -> Optional[str]:
        return self._save

    @save.setter
    def save(self, value: Optional[str]):
        self._save = value
        if isinstance(value, str):
            self.folder_name = value
            self._personnal_folder = os.path.join(
                os.path.join(self.folder_name, "tmp_wks"), f"worker{self.rank}"
            )
        else:
            self.folder_name = f"{self.model.__class__.__name__}_zlj_save"
            self._personnal_folder = os.path.join(
                os.path.join(self.folder_name, "tmp_wks"), f"worker{self.rank}"
            )

    @property
    def _master_rank(self) -> int:
        return self.__master_rank

    @_master_rank.setter
    def _master_rank(self, value: int):
        self.__master_rank = value

        if self.rank == value:
            self.is_master = True

        if self.strat_name == "asynchronous":
            self._strategy = _MonoAsynchronous_strat(self, value, **self.skwargs)
        elif self.strat_name == "synchronous":
            self._strategy = _MonoSynchronous_strat(self, value, **self.skwargs)
        elif self.strat_name == "flexible":
            self._strategy = _MonoFlexible_strat(self, value, **self.skwargs)
        else:
            raise NotImplementedError(
                f"""
                    {self.strat_name} parallelisation is not implemented.
                    Use MPI='asynchronous', 'synchronous', 'flexible', or False for non
                    distributed loss function.
                    """
            )

    def __call__(
        self,
        X: list,
        stop_obj: Optional[Stopping] = None,
        info: dict = {},
        xinfo: dict = {},
    ) -> Tuple[
        list,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        new_x, score, constraints, linfo, lxinfo = self._strategy(X, stop_obj=stop_obj, info=info, xinfo=xinfo)  # type: ignore
        return new_x, score, constraints, linfo, lxinfo

    def master(
        self, pqueue: Optional[Queue] = None, stop_obj: Optional[Stopping] = None
    ):
        """master

        Evaluate a list :code:`X` of solutions with the original loss function.

        Returns
        -------
        res : list
            Return a list of all the scores corresponding to each evaluated solution of X.
        """

        logger.debug(f"Master of rank {self.rank} Starting")

        # if there is a stopping criterion
        if stop_obj:
            stopping = stop_obj
        else:
            stopping = lambda *args: False
        # second stopping criterion determine by the parallelization itself
        cnt = True

        if pqueue:
            self.pqueue = pqueue

        while not stopping() and cnt:
            # Send solutions to workers
            # if a worker is idle and if there are solutions
            while not self.comm.iprobe() and (
                len(self.idle) > 0 and not self.pqueue.empty()
            ):
                self._send_point(self.idle, self.pqueue, self.p_historic)

            if self.comm.iprobe():
                msg = self.comm.recv(status=self.status)
                cnt = self._parse_message(
                    msg, self.pqueue, self.p_historic, self.idle, self.status
                )

        logger.debug(f"MASTER{self.rank}, calls:{self.calls} |!| STOPPING |!|")

    def _parse_message(
        self, msg, pqueue: Queue, historic: dict, idle: List[int], status
    ) -> bool:
        tag = status.Get_tag()
        source = status.Get_source()

        # receive score
        if tag == 1:
            (
                point,
                outputs,
                point_id,
                point_info,
                point_xinfo,
                point_source,
                batch_id,
            ) = self._recv_score(msg, source, idle, historic)

            cnt = self._process_outputs(
                point,
                outputs,
                point_id,
                point_info,
                point_xinfo,
                point_source,
                batch_id,
            )

        # receive a point to add to the queue
        elif tag == 2:
            cnt = self._recv_point(msg, source, pqueue)
            # STOP
        elif tag == 9:
            cnt = False
        # error: abort
        else:
            logger.error(
                f"Unknown message tag, got {tag} from process {source}. Processes will abort"
            )
            cnt = False

        return cnt

    # send point from master to worker
    def _send_point(self, idle: List[int], pqueue: Queue, historic: dict):
        next_point = pqueue.get()
        dest = idle.pop()
        historic[dest] = next_point
        logger.debug(
            f"MASTER {self.rank} sending point to WORKER {dest}.\n Remaining points in queue: {pqueue.qsize()}"
        )
        self.comm.send(dest=dest, tag=0, obj=next_point[0])

    # receive a new point to put in the point queue. (from a forward)
    def _recv_point(self, msg, source: int, pqueue: Queue) -> bool:
        logger.debug(
            f"MASTER {self.rank} receiving point from PROCESS {source}\n{msg}\n"
        )
        pqueue.put(msg)
        return True

    # receive score from workers
    def _recv_score(
        self, msg, source: int, idle: List[int], historic: dict
    ) -> Tuple[list, dict, int, dict, dict, int, int]:
        logger.debug(
            f"MASTER {self.rank} receiving score from WORKER {source} : {msg}, historic : {historic[source]}"
        )
        point = historic[source][0][:]
        point_id = historic[source][1]
        point_info = historic[source][2].copy()
        point_xinfo = historic[source][3].copy()
        point_source = historic[source][4]
        batch_id = historic[source][5]
        historic[source] = [None, None, None, None, None, None]

        outputs = msg

        idle.append(source)

        return point, outputs, point_id, point_info, point_xinfo, point_source, batch_id

    def _process_outputs(
        self,
        point: list,
        outputs: dict,
        id: int,
        info: dict,
        xinfo: dict,
        source: int,
        batch_id: int,
    ) -> bool:
        return self._strategy._process_outputs(point, outputs, id, info, xinfo, source, batch_id)  # type: ignore

    def _stop(self):
        """stop

        Send a stop message to all processes.
        """

        logger.debug(f"MASTER {self.rank} sending stop message")
        for i in range(0, self.p):
            if i != self.rank:
                self.comm.send(dest=i, tag=9, obj=False)

    def _save_model(self, score: np.ndarray, source: int):
        """_save_model

        Save a given model

        Parameters
        ----------
        score : int
            Score corresponding to the solution saved by the worker.
        source : int
            Worker rank which evaluate a solution and return score
        """

        # Save model into a file if it is better than the best found one
        for i in range(len(self.objective)):
            if score[i] < self.best_score[i]:
                master_path = os.path.join(
                    self.model_path, f"{self.folder_name}_best_obj{i}"
                )
                worker_path = os.path.join(
                    os.path.join(self.folder_name, "tmp_wks"), f"worker{source}"
                )

                if os.path.isdir(worker_path):
                    os.system(f"rm -rf {master_path}")
                    os.system(f"cp -rf {worker_path} {master_path}")

    def _wsave_model(self, model):
        if hasattr(model, "save") and callable(getattr(model, "save")):
            os.system(f"rm -rf {self._personnal_folder}")
            model.save(self._personnal_folder)
        else:
            logger.error("The model does not have a method called save")

    def worker(self):
        """worker

        Initialize worker. While it does not receive a stop message,
        a worker will wait for a solution to evaluate.
        """

        logger.debug(f"WORKER {self.rank} starting")

        stop = True
        while stop:
            logger.debug(f"WORKER {self.rank} receving message")
            # receive message from master
            msg = self.comm.recv(source=self._master_rank, status=self.status)  # type: ignore
            tag = self.status.Get_tag()
            source = self.status.Get_source()

            if tag == 9:
                logger.debug(f"WORKER{self.rank} |!| STOPPING |!|")
                stop = False

            elif tag == 0:
                logger.debug(f"WORKER {self.rank} receved a point, {msg}")
                point = msg
                # Verify if a model is returned or not
                outputs, model = self._compute_loss(point)
                outputs["worker"] = self.rank
                # Save the model using its save method
                if model and self.save:
                    logger.debug(f"WORKER {self.rank} saving model")
                    if model:
                        self._wsave_model(model)

                # Send results
                logger.debug(
                    f"WORKER {self.rank} sending {outputs} to {self._master_rank}"
                )

                self.comm.send(dest=self._master_rank, tag=1, obj=outputs)  # type: ignore
            else:
                logger.debug(f"WORKER {self.rank} unknown tag, got {tag}")


class _Parallel_strat:
    def __init__(self, loss: MPILoss, master_rank: int, **kwargs):
        super().__init__()
        self.master_rank = master_rank
        try:
            self.comm = MPI.COMM_WORLD
        except Exception as err:
            logger.error(
                """To use MPILoss object you need to install mpi4py and an MPI
                distribution.\nYou can use: pip install zellij[Parallel]"""
            )

            raise err

        # counter for computed point
        self._computed = 0

        self._lf = loss

    @abstractmethod
    def __call__(
        self,
        X: list,
        stop_obj: Optional[Stopping] = None,
        info: dict = {},
        xinfo: dict = {},
    ) -> Tuple[list, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    def _do_save(
        self,
        point: list,
        y: np.ndarray,
        outputs: dict,
        constraint: Optional[np.ndarray],
        info: Optional[np.ndarray],
        xinfo: Optional[np.ndarray],
        info_dict: dict,
        xinfo_dict: dict,
        source: int,
    ):
        # Save score and solution into the object
        self._lf._save_best(
            point,
            y,
            constraint=constraint,
            info=info,
            xinfo=xinfo,
        )

        if self._lf.save:
            # Save model into a file if it is better than the best found one
            self._lf._save_model(y, source)

            # Save score and solution into a file
            self._lf._save_file(point, **outputs, **info_dict, **xinfo_dict)


# Mono Synchrone -> Save score return list of score
class _MonoSynchronous_strat(_Parallel_strat):
    def __init__(self, loss: MPILoss, master_rank: int, **kwargs):
        super().__init__(loss, master_rank, **kwargs)

        self.y = np.empty(1, dtype=float)  # Initialized during call
        self.list_constraint = None  # Initialized during call
        self.list_info = None  # Initialized during call
        self.list_xinfo = None  # Initialized during call

    # Executed by Experiment to compute X
    def __call__(
        self,
        X: list,
        stop_obj: Optional[Stopping] = None,
        info: dict = {},
        xinfo: dict = {},
    ) -> Tuple[
        list,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:

        pqueue = Queue()
        for id, x in enumerate(X):
            pqueue.put((x, id, info, xinfo, None, id))

        self.y = np.ones((len(X), len(self._lf.objective)), dtype=float)

        if self._lf.constraint:
            self.list_constraint = np.ones(
                (len(X), len(self._lf.constraint)), dtype=float
            )
        else:
            self.list_constraint = None

        if self._lf.info:
            self.list_info = np.ones((len(X), len(self._lf.info)), dtype=float)
        else:
            self.list_info = None

        if self._lf.xinfo:
            self.list_xinfo = np.ones((len(X), len(self._lf.xinfo)), dtype=float)
        else:
            self.list_xinfo = None

        self._lf.master(pqueue, stop_obj=stop_obj)

        return X, self.y, self.list_constraint, self.list_info, self.list_xinfo

    # Executed by master when it receives a score from a worker
    # Here Meta master and loss master are the same process, so shared memory
    def _process_outputs(
        self,
        point: list,
        outputs: dict,
        id: int,
        info_dict: dict,
        xinfo_dict: dict,
        source: int,
        batch_id: int,
    ) -> bool:

        for i in range(len(self._lf.objective)):
            self.y[id][i] = outputs[f"objective{i}"]

        if self._lf.constraint:
            constraint = np.array(
                [outputs[k] for k in self._lf.constraint], dtype=float
            )
            self.list_constraint[id] = constraint  # type: ignore
        else:
            constraint = None

        if self._lf.info:
            for i, k in enumerate(self._lf.info):
                self.list_info[id][i] = info_dict[k]  # type: ignore
            info = self.list_info[id]  # type: ignore
        else:
            info = None

        if self._lf.xinfo:
            pxinfo_dict = dict.fromkeys(self._lf.xinfo, 0.0)
            for i, k in enumerate(self._lf.xinfo):
                self.list_xinfo[batch_id][i] = xinfo_dict[k][batch_id]  # type: ignore
                pxinfo_dict[k] = xinfo_dict[k][batch_id]
            xinfo = self.list_xinfo[batch_id]  # type: ignore
        else:
            pxinfo_dict = {}
            xinfo = None

        self._do_save(
            point=point,
            y=self.y[id],
            outputs=outputs,
            constraint=constraint,
            info=info,
            xinfo=xinfo,
            info_dict=info_dict,
            xinfo_dict=pxinfo_dict,
            source=source,
        )

        self._computed += 1

        if self._computed < len(self.y):  # Continue receiving/sending points
            logger.info(
                f"COMPUTED POINT {self._computed}/{len(self.y)}, calls:{self._lf.calls}"
            )
            return True
        else:  # Stop receiving/sending points
            logger.info(
                f"STOP COMPUTED POINT {self._computed}/{len(self.y)}, calls:{self._lf.calls}"
            )
            self._computed = 0
            return False


class _MonoAsynchronous_strat(_Parallel_strat):
    def __init__(self, loss: MPILoss, master_rank: int, **kwargs):
        super().__init__(loss, master_rank, **kwargs)

        self._current_points = {}

        self._computed_point = None
        self._computed_y = None
        self._computed_constraint = None
        self._computed_info = None
        self._computed_xinfo = None

    # send a point to loss master
    def _send_to_master(
        self, point: list, batch_id: int, info: dict = {}, xinfo: dict = {}
    ):
        id = self._computed  # number of computed points used as id
        self._current_points[id] = point
        self._lf.pqueue.put((point, id, info, xinfo, None, batch_id))
        self._computed += 1

    # Executed by Experiment to compute X
    def __call__(
        self,
        X: list,
        stop_obj: Optional[Stopping] = None,
        info: dict = {},
        xinfo: dict = {},
    ) -> Tuple[
        list,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:

        for id, x in enumerate(X):
            self._send_to_master(x, id, info=info, xinfo=xinfo)  # send point

        self._lf.master(stop_obj=stop_obj)

        if self._computed_point:
            new_x = self._computed_point.copy()
        else:
            raise OutputError("Master process returned a None point.")

        if self._computed_y is not None:
            y = self._computed_y
        else:
            raise OutputError("Master process returned a None score.")

        if self._computed_constraint is not None:
            constraint = np.array([self._computed_constraint.copy()], dtype=float)
        else:
            constraint = None

        if self._computed_info is not None:
            linfo = np.array([self._computed_info.copy()], dtype=float)
        else:
            linfo = None

        if self._computed_xinfo is not None:
            lxinfo = np.array([self._computed_xinfo.copy()], dtype=float)
        else:
            lxinfo = None

        return [new_x], np.array(y, dtype=float), constraint, linfo, lxinfo

    # Executed by master when it receives a score from a worker
    def _process_outputs(
        self,
        point: list,
        outputs: dict,
        id: int,
        info_dict: dict,
        xinfo_dict: dict,
        source: int,
        batch_id: int,
    ) -> bool:
        y = np.array([outputs[f"objective{i}"] for i in range(len(self._lf.objective))])

        if self._lf.constraint:
            constraint = np.array(
                [outputs[k] for k in self._lf.constraint], dtype=float
            )
        else:
            constraint = None

        if self._lf.info:
            info = np.array([info_dict[k] for k in self._lf.info], dtype=float)
        else:
            info = None

        if self._lf.xinfo:
            pxinfo_dict = dict.fromkeys(self._lf.xinfo, 0.0)
            xinfo = np.zeros(len(self._lf.xinfo), dtype=float)
            for i, k in enumerate(self._lf.xinfo):
                xinfo[i] = xinfo_dict[k][batch_id]
                pxinfo_dict[k] = xinfo_dict[k][batch_id]
        else:
            pxinfo_dict = {}
            xinfo = None

        self._do_save(
            point=point,
            y=y,
            outputs=outputs,
            constraint=constraint,
            info=info,
            xinfo=xinfo,
            info_dict=info_dict,
            xinfo_dict=pxinfo_dict,
            source=source,
        )

        self._computed_point = point
        self._computed_y = y
        self._computed_constraint = constraint
        self._computed_info = info
        self._computed_xinfo = xinfo

        return False


class _MonoFlexible_strat(_Parallel_strat):
    def __init__(self, loss: MPILoss, master_rank: int, threshold: int):
        super().__init__(loss, master_rank)
        self._current_points = {}

        self._flex_x = []
        self._flex_y = []
        self._flex_c = []
        self._flex_i = []
        self._flex_xi = []

        # When queue size is < threshold, the loss master stop, so meta can return new points
        self.threshold = threshold

    # send a point to loss master
    def _send_to_master(self, point: list, batch_id: int, info: dict, xinfo: dict):
        id = self._computed  # number of computed points used as id
        self._current_points[id] = point
        self._lf.pqueue.put((point, id, info, xinfo, None, batch_id))
        self._computed += 1

    # Executed by Experiment to compute X
    def __call__(
        self,
        X: list,
        stop_obj: Optional[Stopping] = None,
        info: dict = {},
        xinfo: dict = {},
    ) -> Tuple[
        list,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        self._flex_x = []
        self._flex_y = []
        self._flex_c = []
        self._flex_i = []
        self._flex_xi = []

        # send point, point ID and point info
        for id, x in enumerate(X):
            self._send_to_master(x, id, info=info, xinfo=xinfo)  # send point

        self._lf.master(stop_obj=stop_obj)

        if self._lf.constraint:
            constraint = np.array(self._flex_c, dtype=float)
        else:
            constraint = None

        if self._lf.info:
            linfo = np.array(self._flex_i, dtype=float)
        else:
            linfo = None

        if self._lf.xinfo:
            lxinfo = np.array(self._flex_xi, dtype=float)
        else:
            lxinfo = None

        return (
            self._flex_x,
            np.array(self._flex_y, dtype=float),
            constraint,
            linfo,
            lxinfo,
        )

    # Executed by master when it receives a score from a worker
    def _process_outputs(
        self,
        point: list,
        outputs: dict,
        id: int,
        info_dict: dict,
        xinfo_dict: dict,
        source: int,
        batch_id: int,
    ) -> bool:
        logger.debug(
            f"FLEXIBLE: {len(self._flex_x)},{len(self._flex_y)},{len(self._flex_c)} | SIZE : {self._lf.pqueue.qsize()} <= {self.threshold}"
        )
        self._flex_x.append(point)

        y = np.array(
            [outputs[f"objective{i}"] for i in range(0, len(self._lf.objective))],
            dtype=float,
        )
        self._flex_y.append(y)

        if self._lf.constraint:
            constraint = np.array(
                [outputs[k] for k in self._lf.constraint], dtype=float
            )
            self._flex_c.append(constraint)
        else:
            constraint = None

        if self._lf.info:
            info = np.array([info_dict[k] for k in self._lf.info], dtype=float)
            self._flex_i.append(info)
        else:
            info = None

        if self._lf.xinfo:
            pxinfo_dict = dict.fromkeys(self._lf.xinfo, 0.0)
            xinfo = np.zeros(len(self._lf.xinfo), dtype=float)
            for i, k in enumerate(self._lf.xinfo):
                xinfo[i] = xinfo_dict[k][batch_id]
                pxinfo_dict[k] = xinfo_dict[k][batch_id]
            self._flex_xi.append(xinfo)
        else:
            pxinfo_dict = {}
            xinfo = None

        self._do_save(
            point=point,
            y=y,
            outputs=outputs,
            constraint=constraint,
            info=info,
            xinfo=xinfo,
            info_dict=info_dict,
            xinfo_dict=pxinfo_dict,
            source=source,
        )
        return self._lf.pqueue.qsize() >= self.threshold


# Wrap different loss functions
def Loss(
    objective: List[Objective],
    constraint: Optional[list[str]] = None,
    default: Optional[dict] = None,
    record_time: bool = False,
    only_score: bool = False,
    kwargs_mode: bool = False,
    mpi: Optional[str] = None,
    workers: Optional[int] = None,
    **kwargs,
):
    """Loss

    Wrap a function of type :math:`f(x)=dict`. See :ref:`lf` for more info.

    Parameters
    ----------
    model : Callable[..., dict]
        Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
        of hyperparameters for example.
        And :math:`y` can be a single value, a list, a tuple, or a dict,
        containing the loss value and other optionnal information.
        It can also be of mixed types, containing, strings, float, int...
    objective : List[Objective]
        An :code:`Objective` object determines what the optimization problem is.
        Mono-objective, if the list contains only one :code:`Objective`. Otherwise, multi-Objective.
    record_time : boolean, default=False
        If True, :code:`start_time`, :code:`end_time`, :code:`start_date`, :code:`end_date` will be recorded
        and saved in the save file for each :code:`__call__`.
    only_score : bool, default=False
        If True, then only the score of evaluated solutions are saved.
        Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
        saved.
    kwargs_mode : bool, default=False
        If True, then solutions are passed as kwargs to :ref:`lf`. Keys are
        the names of the :ref:`var` within the :ref:`sp`.
    mpi : {'asynchronous', 'synchronous', 'flexible'}, optional
        Wrap the function with :code:`MPILoss`. Default is :code:`SequentialLoss` else.
        if :code:`strategy='synchronous`: then :code:`__call__` will return all results from all
        solutions passed, once all of them have been evaluated.
        if :code:`strategy='asynchronous`: then :code:`__call__` will return
        the result from an evaluation of a solution assoon as it receives a
        result from a worker. Other solutions, are still being evaluated in
        background.
        if :code:`strategy='flexible`: then :code:`__call__` will return
        all computed results, only if the number of remaining uncomputed solutions
        is below a certain threshold. Pass: :code:`threshold=int` kwarg, to :code:`Loss`
        or :code:`MPILoss`.
    workers : int, optionnal
        Number of workers among the total number of processes spawned by
        MPI. At least, one process is dedicated to the master.
    default : dict, optionnal
        Dictionnary of defaults arguments, kwargs, to pass to the loss function.
        They are not affected by any :ref:`meta` or other methods.
    constraint : list[str], default=None
        Constraints works when the model returns a dictionnary of values.
        Constraints values returned by the model must be booleans.
        If a list of strings is passed, constraints values will be passed to
        the :code:`forward` method of :ref:`meta`.

    Returns
    -------
    wrapper : LossFunc
        Wrapped original function

    Examples
    --------

    This is an example using SequentialLoss. MPILoss requires a MPI distribution and
    a script ran using the following: :code:`mpiexec -n <n_processes> <script>.py`.

    >>> from zellij.core import Loss, Minimizer, Maximizer, Lambda
    >>> import numpy as np
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.core import ContinuousSearchspace

    >>> a = ArrayVar(FloatVar("f1", 0, 10), FloatVar("i2", -10, 0))
    >>> sp = ContinuousSearchspace(a)

    >>> def sec1vssec2(obj1, obj2): # Lambda objective
    ...     return obj1 / obj2

    >>> @Loss(
    ...    objective=Minimizer("obj"),
    ...    secondary=[
    ...        Minimizer("sec1"),
    ...        Maximizer("sec2"),
    ...        Lambda(["sec1", "sec2"], sec1vssec2),
    ...    ],
    ...    constraint=["c1", "c2"],
    ...)
    ... def composed_func(x):
    ...    x_ar = np.array(x)
    ...    return {
    ...        "obj": (x_ar[0] ** 2 + x_ar[1] - 11) ** 2 + (x_ar[0] + x_ar[1] ** 2 - 7) ** 2,
    ...        "sec1": x_ar[0] ** 2.0 + x_ar[1] ** 2.0,
    ...        "sec2": -(0.26 * (x_ar[0] ** 2 + x_ar[1] ** 2) - 0.48 * x_ar[0] * x_ar[1]),
    ...        "c1": -x_ar[0],
    ...        "c2": -x_ar[1],
    ...    }
    >>> points, losses, secondary, constraint = composed_func(sp.random_point(5))
    >>> for x, y, s, c in zip(points, losses, secondary, constraint):
    ...    sstr = np.char.mod("%.1f", s)
    ...    cstr = np.char.mod("%.1f", c)
    ...    print(
    ...        f"f({x[0]:.1f},{x[1]:.1f})=({y:.1f}{', '.join(sstr)}), s.t. ({'<0, '.join(cstr)}<0)"
    ...    )
    f(2.8,-8.2)=(4098.474.9, 30.4, -2.5), s.t. (-2.8<0, 8.2<0)
    f(2.5,-2.6)=(59.013.3, 6.6, -2.0), s.t. (-2.5<0, 2.6<0)
    f(5.3,-2.5)=(235.834.3, 15.2, -2.3), s.t. (-5.3<0, 2.5<0)
    f(7.5,-9.7)=(10123.0149.6, 73.6, -2.0), s.t. (-7.5<0, 9.7<0)
    f(0.6,-8.3)=(4207.468.8, 20.3, -3.4), s.t. (-0.6<0, 8.3<0)

    """

    def wrapper(model: Callable[..., dict]):
        if mpi:
            return MPILoss(
                model=model,
                objective=objective,
                constraint=constraint,
                default=default,
                record_time=record_time,
                only_score=only_score,
                kwargs_mode=kwargs_mode,
                workers=workers,
                strategy=mpi,
                **kwargs,
            )
        else:
            return SequentialLoss(
                model=model,
                objective=objective,
                constraint=constraint,
                default=default,
                record_time=record_time,
                only_score=only_score,
                kwargs_mode=kwargs_mode,
                **kwargs,
            )

    return wrapper


class MockModel(object):
    """MockModel

    This object allows to replace your real model with a costless object,
    by mimicking different available configurations in Zellij.
    This object does not replace any Loss wrapper.

    Parameters
    ----------
    outputs : dict
        Dictionnary containing outputs name (keys)
        and functions to execute to obtain outputs.
        Pass args and kwargs to these functions when calling this MockModel.
    verbose : bool
        If True logger.info information when saving and __call___.
    return_model : boolean
        Return :code:`(outputs, MockModel)` if True. Else, :code:`outputs`.

    See Also
    --------
    Loss : Wrapper function
    MPILoss : Distributed version of LossFunc
    SerialLoss : Basic version of LossFunc

    Examples
    --------
    >>> from zellij.core.loss_func import MockModel, Loss
    >>> mock = MockModel(outputs={"o1": lambda *args, **kwargs: np.random.random()})
    >>> print(mock("test", 1, 2.0, param1="Mock", param2=True))
    I am Mock !
        ->*args: ('test', 1, 2.0)
        ->**kwargs: {'param1': 'Mock', 'param2': True}
    ({'o1': 0.3440051802032301},
    <zellij.core.loss_func.MockModel at 0x7f5c8027a100>)
    >>> loss = Loss(save=True, verbose=False)(mock)
    >>> print(loss([["test", 1, 2.0, "Mock", True]], other_info="Hi !"))
    I am Mock !
        ->*args: (['test', 1, 2.0, 'Mock', True],)
        ->**kwargs: {}
    I am Mock !
        ->saving in MockModel_zlj_save/model/MockModel_best/i_am_mock.txt
    [0.7762604280531996]
    """

    def __init__(
        self,
        outputs: dict,
        return_model: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.outputs = outputs
        self.return_model = return_model
        self.verbose = verbose

    def save(self, filepath: str):
        os.makedirs(filepath, exist_ok=True)
        filename = os.path.join(filepath, "i_am_mock.txt")
        with open(filename, "wb") as f:
            if self.verbose:
                logger.info(f"\nI am Mock !\n\t->saving in {filename}")

    def __call__(self, *args, **kwargs):
        if self.verbose:
            logger.info(f"\nI am Mock !\n\t->*args: {args}\n\t->**kwargs: {kwargs}")

        outputs = {x: y(*args, **kwargs) for x, y in self.outputs.items()}
        if self.return_model:
            outputs["model"] = self

        return outputs
