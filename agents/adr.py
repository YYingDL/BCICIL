"""ADR: analytic recursive learning with score-based target relaxation."""

from agents.acil import ACIL as _ScoreRelaxationLearner


class ADR(_ScoreRelaxationLearner):
    """Public ADR method entry point.

    The implementation uses the repository's analytic RLS learner and
    score-based target relaxation.  The legacy implementation class is kept
    private to this module so the public experiment name is consistently ADR.
    """

    def __init__(self, model, args):
        # Keep the tested implementation internals stable while presenting
        # ADR-named command-line parameters to users.
        args.acil_buffer_size = args.adr_buffer_size
        args.acil_gamma = args.adr_gamma
        args.acil_overconf_threshold = args.adr_overconf_threshold
        args.acil_relax_margin = args.adr_relax_margin
        super().__init__(model, args)
