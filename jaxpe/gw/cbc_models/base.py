import abc
import jax


class WaveformModel(abc.ABC):
    """
    Abstract base class for all gravitational-wave CBC models.
    """

    @abc.abstractmethod
    def __call__(self, params: dict, grid: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Generate the waveform polarizations.

        Parameters
        ----------
        params : dict
            Dictionary of intrinsic and extrinsic parameters describing the binary.
        grid : jax.Array
            The evaluation grid (frequencies in Hz for FD models, times in s for TD models).

        Returns
        -------
        h_plus : jax.Array
            The plus polarization of the gravitational wave.
        h_cross : jax.Array
            The cross polarization of the gravitational wave.
        """
        pass


class FrequencyDomainModel(WaveformModel):
    """
    Base class for frequency-domain waveform models.
    """

    is_fd = True


class TimeDomainModel(WaveformModel):
    """
    Base class for time-domain waveform models.
    """

    is_fd = False
