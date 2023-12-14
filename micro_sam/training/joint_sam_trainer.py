from .sam_trainer import SamTrainer


# Some wrapper class around UNETR to make setting the same encoder as in the SAM model easy

class JointSamTrainer(SamTrainer):
  # some additional arguments for the UNETR / UNETR decoder
  def __init__(self, ...):
    super.__init__(...)
    # safe the unetr / unetr decoder
    self.unetr = ...

  def _train_epoch_impl(self):
    for x, y in self.train_loader:
      # forward contexts etc.
      ... = self._train_interactive(...)
      # do the backprop etc.
      ... = self._train_instance(...)
      # do the backprop etc.
