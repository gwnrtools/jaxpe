import json
import subprocess
import pytest
import sys


def run_command(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"Command failed:\n{cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result


# ---------------------------------------------------------------------------
# Run configuration. These drive the CLI but never sample, so they are cheap
# enough to run by default rather than behind --run-cli.
# ---------------------------------------------------------------------------


def test_cli_write_config_round_trips(tmp_path):
    cfg_path = tmp_path / "run.json"
    run_command(f"{sys.executable} -m jaxpe.cli write-config {cfg_path}")
    assert cfg_path.exists()

    # Refuses to clobber, unless told to.
    again = subprocess.run(
        f"{sys.executable} -m jaxpe.cli write-config {cfg_path}",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert again.returncode != 0 and "already exists" in again.stderr
    run_command(f"{sys.executable} -m jaxpe.cli write-config {cfg_path} --force")

    # And what it wrote is loadable by the CLI itself.
    outdir = tmp_path / "inj"
    run_command(
        f"{sys.executable} -m jaxpe.cli generate-injections --config {cfg_path} "
        f"--n-injections 1 --outdir {outdir}"
    )


def test_cli_config_controls_the_physics(tmp_path):
    """A config file must actually change the run, and be recorded with the set."""
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "data": {
                    "duration": 16.0,
                    "sampling_rate": 2048.0,
                    "f_min": 20.0,
                    "f_ref": 20.0,
                },
                "prior": {"chirp_mass": [5.0, 12.0]},
                "injection": {"parameters": {"chirp_mass": [6.0, 10.0]}},
            }
        )
    )
    outdir = tmp_path / "inj"
    run_command(
        f"{sys.executable} -m jaxpe.cli generate-injections --config {cfg_path} "
        f"--n-injections 4 --outdir {outdir}"
    )

    # The drawn population follows the file, not the built-in defaults.
    for i in range(4):
        params = json.loads((outdir / f"inj_{i}.json").read_text())
        assert 6.0 <= params["chirp_mass"] <= 10.0, "draw ignored injection.parameters"

    # The resolved configuration is dropped beside the set so run-pe inherits it.
    resolved = json.loads((outdir / "config.json").read_text())
    assert resolved["data"]["duration"] == 16.0
    assert resolved["data"]["sampling_rate"] == 2048.0
    assert resolved["prior"]["chirp_mass"]["low"] == 5.0


def test_cli_narrow_prior_does_not_require_moving_the_fiducial(tmp_path):
    """A BNS-scale prior must not be blocked by the demo 30 Msun reference binary.

    The fiducial only matters to --fiducial runs, so it warns at load time and is
    fatal only at the point of use.
    """
    cfg_path = tmp_path / "bns.json"
    cfg_path.write_text(
        json.dumps(
            {
                "prior": {"chirp_mass": [1.0, 2.0], "mass_ratio": [0.5, 1.0]},
                "injection": {
                    "parameters": {
                        "chirp_mass": [1.1, 1.9],
                        "mass_ratio": [0.6, 1.0],
                    }
                },
            }
        )
    )
    result = run_command(
        f"{sys.executable} -m jaxpe.cli generate-injections --config {cfg_path} "
        f"--n-injections 1 --outdir {tmp_path / 'inj'}"
    )
    assert "injection.fiducial lies outside the prior" in result.stdout
    params = json.loads((tmp_path / "inj" / "inj_0.json").read_text())
    assert 1.1 <= params["chirp_mass"] <= 1.9

    # But asking for the fiducial binary under that prior is refused.
    refused = subprocess.run(
        f"{sys.executable} -m jaxpe.cli generate-injections --config {cfg_path} "
        f"--fiducial --n-injections 1 --outdir {tmp_path / 'fid'}",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "--fiducial cannot be used" in refused.stderr


def test_cli_gives_each_injection_its_own_noise_seed(tmp_path):
    """A campaign must not analyse N copies of one noise realisation.

    seeds.noise was previously passed through verbatim, so every injection in a
    --noise gaussian campaign got byte-identical noise -- which silently
    invalidates the PP test such a campaign exists to run.
    """
    outdir = tmp_path / "inj"
    run_command(
        f"{sys.executable} -m jaxpe.cli generate-injections --noise gaussian "
        f"--n-injections 3 --outdir {outdir}"
    )
    seeds, indices = [], []
    for i in range(3):
        meta = json.loads((outdir / f"inj_{i}.json").read_text())["metadata"]
        assert meta["noise_seed"] is not None, "noise seed not recorded in the artifact"
        seeds.append(meta["noise_seed"])
        indices.append(meta["index"])
    assert indices == [0, 1, 2]
    assert len(set(seeds)) == 3, f"injections share a noise seed: {seeds}"


def test_cli_noise_seeds_are_reproducible(tmp_path):
    """The same campaign seed must reproduce the same per-injection seeds."""
    seeds = []
    for run in ("a", "b"):
        outdir = tmp_path / run
        run_command(
            f"{sys.executable} -m jaxpe.cli generate-injections --noise gaussian "
            f"--n-injections 2 --outdir {outdir}"
        )
        seeds.append(
            [
                json.loads((outdir / f"inj_{i}.json").read_text())["metadata"][
                    "noise_seed"
                ]
                for i in range(2)
            ]
        )
    assert seeds[0] == seeds[1], f"campaign is not reproducible: {seeds}"


def test_cli_rejects_a_broken_config(tmp_path):
    """A typo must stop the run, not silently leave a default in place."""
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps({"data": {"sample_rate": 4096.0}}))
    result = subprocess.run(
        f"{sys.executable} -m jaxpe.cli generate-injections --config {cfg_path} "
        f"--n-injections 1 --outdir {tmp_path / 'inj'}",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown key" in result.stderr and "sample_rate" in result.stderr


def test_cli_config_rejects_injections_outside_the_prior(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(
        json.dumps(
            {
                "prior": {"chirp_mass": [20.0, 30.0]},
                "injection": {"parameters": {"chirp_mass": [10.0, 40.0]}},
            }
        )
    )
    result = subprocess.run(
        f"{sys.executable} -m jaxpe.cli generate-injections --config {cfg_path} "
        f"--n-injections 1 --outdir {tmp_path / 'inj'}",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not contained in" in result.stderr


@pytest.mark.cli
def test_cli_run_pe_inherits_the_generation_config(tmp_path):
    """run-pe must analyse under the configuration the injections were made with.

    Without this, a set generated at one conditioning could silently be analysed at
    another, and nothing in the output would record the mismatch.
    """
    outdir = tmp_path / "out"
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "data": {
                    "duration": 8.0,
                    "sampling_rate": 2048.0,
                    "f_min": 20.0,
                    "f_ref": 20.0,
                },
                "kernel": {"hmc": {"step_size": 0.005, "n_leapfrog": 4}},
            }
        )
    )
    run_command(
        f"{sys.executable} -m jaxpe.cli generate-injections --config {cfg_path} "
        f"--fiducial --network H1,L1 --noise zero --psd aligo "
        f"--n-injections 1 --outdir {outdir}"
    )
    # No --config here: it must be picked up from beside the injection.
    result = run_command(
        f"JAX_PLATFORMS=cpu {sys.executable} -m jaxpe.cli run-pe "
        f"--injection {outdir}/inj_0.json --sampler hmc --domain fd --likelihood full "
        f"--n-chains 2 --n-prelim-loops 1 --n-training-loops 1 --n-production-loops 1 "
        f"--outdir {outdir}"
    )
    assert "config.json" in result.stdout, "did not inherit the generation config"

    recorded = json.loads((outdir / "run_config.json").read_text())
    assert recorded["duration"] == 8.0
    assert recorded["sampling_rate"] == 2048.0
    # The embedded config must describe the run, including the flag overrides.
    assert recorded["config"]["kernel"]["hmc"]["n_leapfrog"] == 4
    assert recorded["config"]["sampler"]["n_chains"] == 2


@pytest.mark.cli
def test_cli_td_hmc(tmp_path):
    outdir = tmp_path / "out"

    # 1. Generate injection
    cmd_gen = f"{sys.executable} -m jaxpe.cli generate-injections --fiducial --network H1,L1 --noise zero --psd aligo --n-injections 1 --outdir {outdir}"
    run_command(cmd_gen)

    # 2. Run PE
    cmd_pe = (
        f"JAX_PLATFORMS=cpu {sys.executable} -m jaxpe.cli run-pe "
        f"--injection {outdir}/inj_0.json "
        f"--sampler hmc --domain td --likelihood full "
        f"--n-chains 2 --n-prelim-loops 1 --n-training-loops 1 --n-production-loops 1 "
        f"--outdir {outdir}"
    )
    run_command(cmd_pe)


@pytest.mark.cli
def test_cli_fd_ns(tmp_path):
    outdir = tmp_path / "out"

    # 1. Generate injection
    cmd_gen = f"{sys.executable} -m jaxpe.cli generate-injections --fiducial --network H1,L1 --noise zero --psd aligo --n-injections 1 --outdir {outdir}"
    run_command(cmd_gen)

    # 2. Run PE
    cmd_pe = (
        f"JAX_PLATFORMS=cpu {sys.executable} -m jaxpe.cli run-pe "
        f"--injection {outdir}/inj_0.json "
        f"--sampler ns --domain fd --likelihood marginalized_phase_distance "
        f"--outdir {outdir}"
    )
    run_command(cmd_pe)


@pytest.mark.cli
def test_cli_fd_gpry(tmp_path):
    outdir = tmp_path / "out"

    # 1. Generate injection
    cmd_gen = f"{sys.executable} -m jaxpe.cli generate-injections --fiducial --network H1,L1 --noise zero --psd aligo --n-injections 1 --outdir {outdir}"
    run_command(cmd_gen)

    cmd_pe = (
        f"JAX_PLATFORMS=cpu {sys.executable} -m jaxpe.cli run-pe "
        f"--injection {outdir}/inj_0.json "
        f"--sampler gpry --domain fd --likelihood marginalized_phase_distance "
        f"--outdir {outdir}"
    )
    result = subprocess.run(cmd_pe, shell=True, capture_output=True, text=True)

    # This run is not reproducible and cannot be made so from here: GPry warns
    # "Seeded runs are not supported for UltraNest", so the nested sampler under NORA
    # acquisition and the final surrogate draw both vary between identical invocations.
    # Measured on this fixed injection: 3 of 4 runs succeed and the fourth dies inside
    # ultranest/mlfriends.pyx. Enumerating those numerical failures by message is a
    # losing game -- the list grew once already.
    #
    # What this test is actually for is the *plumbing*: that jaxpe hands GPry a usable
    # likelihood, bounds and options. So a clean exit passes, a GPry/UltraNest numerical
    # failure passes, and the wiring-error family fails loudly. That distinction is what
    # caught `problem.log_prob` not existing on InferenceProblem.
    if result.returncode != 0:
        wiring_errors = [
            name
            for name in ("AttributeError", "TypeError", "NameError", "ImportError")
            if f"{name}:" in result.stderr
        ]
        assert not wiring_errors, (
            f"gpry exited {result.returncode} with a jaxpe wiring error "
            f"({', '.join(wiring_errors)}) rather than a GPry convergence failure:\n"
            f"{result.stderr[-2000:]}"
        )
