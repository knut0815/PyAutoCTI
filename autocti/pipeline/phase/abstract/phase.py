from autocti.pipeline.phase.abstract.result import Result
from autofit.optimize.non_linear.nested_sampling.multi_nest import MultiNest
from autofit.optimize.non_linear.paths import convert_paths
from autofit.tools import phase as af_phase


# noinspection PyAbstractClass


class AbstractPhase(af_phase.AbstractPhase):

    Result = Result

    @convert_paths
    def __init__(self, paths, *, non_linear_class=MultiNest):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit
        models and hyper_galaxies passed to it.

        Parameters
        ----------
        non_linear_class: class
            The class of a non_linear optimizer
        phase_name: str
            The name of this phase
        """

        super().__init__(paths=paths, non_linear_class=non_linear_class)

    @property
    def phase_folders(self):
        return self.optimizer.phase_folders

    @property
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhaseProperty]
            A list of phase property collections associated with this phase. This is
            used in automated prior passing and should be overridden for any phase that
            contains its own PhasePropertys.
        """
        return []

    @property
    def path(self):
        return self.optimizer.path

    def make_result(self, result, analysis):
        return self.Result(
            samples=result.samples,
            previous_model=result.previous_model,
            analysis=analysis,
            optimizer=self.optimizer,
        )
