"""Tests for the declarative CLI run configuration.

The load-bearing test here is :func:`test_defaults_reproduce_bbh_priors`: the CLI's
built-in defaults must build *exactly* the prior the hardcoded constants used to
produce, or every result predating the configuration file becomes incomparable.
"""

import json
import math

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxpe import config as rc
from jaxpe.gw import bbh_priors


def write(tmp_path, payload, name="cfg.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_defaults_load_and_validate():
    cfg, warnings = rc.load_config()
    assert set(cfg) == set(rc.DEFAULT_CONFIG)
    # The defaults are smoke-test scale and say so, rather than pretending otherwise.
    assert any("nlive" in w for w in warnings)


def test_defaults_reproduce_bbh_priors():
    """The default config must build the prior the CLI's constants used to build.

    Compares names, order and log-density -- not just the box -- so a change of
    distribution family (e.g. powerlaw -> uniform in distance) cannot slip through.
    """
    cfg, _ = rc.load_config()
    trigger = 1126259462.4
    old = bbh_priors(
        chirp_mass=(10.0, 50.0),
        mass_ratio=(0.1, 1.0),
        aligned_spins=(-0.9, 0.9),
        luminosity_distance=(100.0, 2000.0),
        geocent_time=trigger,
        time_width=0.1,
    )
    new = rc.build_prior(cfg, trigger=trigger)

    assert new.names == old.names, "parameter order fixes saved column order"

    xs = old.sample(jax.random.PRNGKey(0), 300)
    diff = max(abs(float(old.log_prob(x)) - float(new.log_prob(x))) for x in xs)
    assert diff == 0.0, f"log-density differs by {diff}"


def test_default_time_width_matches_historical_value():
    cfg, _ = rc.load_config()
    assert rc.time_width(cfg) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_partial_config_changes_only_what_it_names(tmp_path):
    path = write(tmp_path, {"data": {"duration": 128.0}})
    cfg, _ = rc.load_config(path)
    assert cfg["data"]["duration"] == 128.0
    assert cfg["data"]["sampling_rate"] == rc.DEFAULT_CONFIG["data"]["sampling_rate"]
    assert cfg["prior"] == rc.load_config()[0]["prior"]


def test_underscore_keys_are_comments(tmp_path):
    path = write(
        tmp_path,
        {"_about": "notes", "data": {"_why": "because", "duration": 16.0}},
    )
    cfg, _ = rc.load_config(path)
    assert cfg["data"]["duration"] == 16.0
    assert "_about" not in cfg and "_why" not in cfg["data"]


def test_distribution_entries_replace_rather_than_merge(tmp_path):
    """Merging a powerlaw onto a uniform must not leave a uniform carrying alpha."""
    path = write(
        tmp_path,
        {
            "prior": {
                "chirp_mass": {
                    "dist": "powerlaw",
                    "alpha": 1.0,
                    "low": 5.0,
                    "high": 50.0,
                }
            }
        },
    )
    cfg, _ = rc.load_config(path)
    spec = cfg["prior"]["chirp_mass"]
    assert spec["dist"] == "powerlaw" and spec["alpha"] == 1.0
    assert "high" in spec and spec["high"] == 50.0


def test_shorthand_range_means_uniform(tmp_path):
    # Must still contain the default injection draws, or containment rightly fails.
    path = write(tmp_path, {"prior": {"chirp_mass": [12.0, 46.0]}})
    cfg, _ = rc.load_config(path)
    assert cfg["prior"]["chirp_mass"]["dist"] == "uniform"
    assert (cfg["prior"]["chirp_mass"]["low"], cfg["prior"]["chirp_mass"]["high"]) == (
        12.0,
        46.0,
    )


# ---------------------------------------------------------------------------
# Validation: things that must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "patch, expected",
    [
        ({"data": {"sample_rate": 4096.0}}, "unknown key"),
        ({"nosuchsection": {}}, "unknown section"),
        ({"data": {"sampling_rate": 512.0, "f_min": 300.0}}, "Nyquist"),
        ({"data": {"duration": 4.0, "post_trigger": 8.0}}, "shorter than"),
        ({"data": {"tukey_alpha": 5.0}}, "[0, 1]"),
        ({"prior": {"mass_ratio": [0.5, 2.0]}}, "(0, 1]"),
        ({"prior": {"spin1z": [-1.5, 1.5]}}, "[-1, 1]"),
        ({"prior": {"chirp_mass": [50.0, 10.0]}}, "inverted"),
        (
            {"prior": {"inclination": {"dist": "sine", "low": 0.0, "high": 6.3}}},
            "[0, pi]",
        ),
        ({"prior": {"chirp_mass": {"dist": "lognormal"}}}, "unknown distribution"),
        (
            {
                "prior": {
                    "luminosity_distance": {"dist": "powerlaw", "low": 1.0, "high": 2.0}
                }
            },
            "requires 'alpha'",
        ),
        (
            {
                "prior": {
                    "chirp_mass": {
                        "dist": "uniform",
                        "low": 1.0,
                        "high": 2.0,
                        "alpha": 3.0,
                    }
                }
            },
            "does not take",
        ),
        ({"kernel": {"hmc": {"step_size": -0.1}}}, "step_size"),
        ({"sampler": {"n_production_loops": 0}}, "no production loops"),
        ({"gpry": {"n_initial": 900, "max_total": 100}}, "never start"),
        ({"seeds": {"sampler": "forty-two"}}, "integer or null"),
    ],
)
def test_bad_config_is_rejected(tmp_path, patch, expected):
    path = write(tmp_path, patch)
    with pytest.raises(rc.ConfigError) as exc:
        rc.load_config(path)
    assert expected in str(exc.value)


def test_injection_outside_prior_is_rejected(tmp_path):
    """The invariant that used to be a comment: truths must lie inside the prior."""
    path = write(tmp_path, {"injection": {"parameters": {"chirp_mass": [5.0, 60.0]}}})
    with pytest.raises(rc.ConfigError, match="not contained in"):
        rc.load_config(path)


def test_fiducial_outside_prior_warns_rather_than_blocking(tmp_path):
    """Only --fiducial runs use it, so a narrow prior must not be blocked by it."""
    path = write(
        tmp_path,
        {
            "prior": {"chirp_mass": [31.0, 50.0]},
            "injection": {"parameters": {"chirp_mass": [32.0, 48.0]}},
        },
    )
    cfg, warnings = rc.load_config(path)
    assert any("fiducial" in w for w in warnings)
    # ... and the problem is reported precisely, for the CLI to make fatal on use.
    problems = rc.fiducial_errors(cfg)
    assert any("chirp_mass" in p for p in problems)


def test_fiducial_errors_is_empty_for_a_consistent_config():
    cfg, _ = rc.load_config()
    assert rc.fiducial_errors(cfg) == []


def test_omitting_a_parameter_inherits_it_rather_than_dropping_it(tmp_path):
    """A file cannot delete a parameter: merging restores it from the defaults.

    This is why the coverage check below has to be exercised against a hand-built
    config -- through load_config the prior is always complete by construction.
    """
    cfg_dict = json.loads(json.dumps(rc.DEFAULT_CONFIG))
    del cfg_dict["prior"]["psi"]
    cfg, _ = rc.load_config(write(tmp_path, cfg_dict))
    assert cfg["prior"]["psi"] == rc.load_config()[0]["prior"]["psi"]


def test_missing_parameter_is_rejected_with_a_usable_hint():
    cfg = json.loads(json.dumps(rc.DEFAULT_CONFIG))
    del cfg["prior"]["psi"]
    with pytest.raises(rc.ConfigError) as exc:
        rc.validate_config(cfg)
    assert "psi" in str(exc.value)
    assert "fixed" in str(exc.value), "should point at the way to pin a parameter"


def test_pinning_a_parameter_is_the_supported_way_to_hold_it_fixed(tmp_path):
    path = write(
        tmp_path,
        {
            "prior": {"spin1z": {"dist": "fixed", "value": 0.0}},
            "injection": {"parameters": {"spin1z": {"dist": "fixed", "value": 0.0}}},
        },
    )
    cfg, _ = rc.load_config(path)
    prior = rc.build_prior(cfg)
    assert prior.names == rc.PARAMETERS, "a pinned parameter keeps its slot"
    drawn = rc.sample_parameters(cfg, jax.random.PRNGKey(0))
    assert drawn["spin1z"] == 0.0


def test_all_problems_are_reported_at_once(tmp_path):
    path = write(
        tmp_path,
        {
            "data": {"duration": -1.0, "tukey_alpha": 5.0},
            "prior": {"mass_ratio": [2.0, 0.5]},
        },
    )
    with pytest.raises(rc.ConfigError) as exc:
        rc.load_config(path)
    assert "3 problems" in str(exc.value)


def test_unreadable_and_malformed_files(tmp_path):
    with pytest.raises(rc.ConfigError, match="not found"):
        rc.load_config(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(rc.ConfigError, match="not valid JSON"):
        rc.load_config(bad)
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]")
    with pytest.raises(rc.ConfigError, match="JSON object"):
        rc.load_config(arr)


# ---------------------------------------------------------------------------
# Validation: things that must be warned about but allowed
# ---------------------------------------------------------------------------


def test_smoke_test_nlive_warns_but_runs(tmp_path):
    path = write(tmp_path, {"ns": {"nlive": 10}})
    _, warnings = rc.load_config(path)
    assert any("smoke-test value" in w for w in warnings)


def test_f_ref_below_band_warns(tmp_path):
    path = write(tmp_path, {"data": {"f_min": 30.0, "f_ref": 20.0}})
    _, warnings = rc.load_config(path)
    assert any("f_ref" in w for w in warnings)


def test_production_config_does_not_warn_about_the_band(tmp_path):
    path = write(tmp_path, {"data": {"f_min": 20.0, "f_ref": 20.0}})
    _, warnings = rc.load_config(path)
    assert not any("f_ref" in w for w in warnings)


# ---------------------------------------------------------------------------
# The distribution language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        {"dist": "uniform", "low": 1.0, "high": 3.0},
        {"dist": "loguniform", "low": 1.0, "high": 100.0},
        {"dist": "powerlaw", "alpha": 2.0, "low": 1.0, "high": 10.0},
        {"dist": "sine", "low": 0.0, "high": math.pi},
        {"dist": "cosine", "low": -math.pi / 2, "high": math.pi / 2},
        {"dist": "gaussian", "mu": 3.0, "sigma": 1.0},
        {"dist": "fixed", "value": 2.5},
    ],
)
def test_every_distribution_builds_and_samples_in_support(spec):
    errors = []
    norm = rc.normalize_spec(spec, "test", errors)
    assert not errors
    dist = rc.build_distribution(norm)
    draws = np.asarray(dist.sample(jax.random.PRNGKey(0), (512,)))
    assert np.all(np.isfinite(draws))
    lo, hi = rc.spec_support(norm)
    assert np.all(draws >= lo - 1e-9) and np.all(draws <= hi + 1e-9)
    # spec_bounds must be finite even where the support is not.
    blo, bhi = rc.spec_bounds(norm)
    assert math.isfinite(blo) and math.isfinite(bhi) and blo < bhi


def test_powerlaw_alpha_minus_one_is_rejected_with_a_pointer():
    errors = []
    rc.normalize_spec(
        {"dist": "powerlaw", "alpha": -1.0, "low": 1.0, "high": 2.0}, "p", errors
    )
    assert any("loguniform" in e for e in errors)


def test_relative_to_trigger_shifts_the_support():
    spec = {"dist": "uniform", "low": -0.1, "high": 0.1, "relative_to_trigger": True}
    errors = []
    norm = rc.normalize_spec(spec, "t", errors)
    assert not errors
    lo, hi = rc.spec_support(norm, trigger=1126259462.4)
    assert lo == pytest.approx(1126259462.3)
    assert hi == pytest.approx(1126259462.5)


def test_isotropic_orientation_is_expressible():
    """sine/cosine must reproduce isotropy, not merely be accepted."""
    cfg, _ = rc.load_config()
    inc = rc.build_distribution(cfg["prior"]["inclination"])
    dec = rc.build_distribution(cfg["prior"]["dec"])
    n = 20000
    cos_i = np.cos(np.asarray(inc.sample(jax.random.PRNGKey(3), (n,))))
    sin_d = np.sin(np.asarray(dec.sample(jax.random.PRNGKey(4), (n,))))
    # cos(inclination) and sin(dec) are uniform on [-1, 1] for an isotropic population.
    for label, v in (("cos(inclination)", cos_i), ("sin(dec)", sin_d)):
        assert abs(v.mean()) < 5.0 / math.sqrt(n), f"{label} not centred"
        assert abs(v.var() - 1.0 / 3.0) < 0.02, f"{label} not uniform on [-1, 1]"


# ---------------------------------------------------------------------------
# The "prior" alias
# ---------------------------------------------------------------------------


def test_prior_alias_binds_injections_to_the_prior(tmp_path):
    path = write(
        tmp_path,
        {
            "prior": {"chirp_mass": [20.0, 40.0]},
            "injection": {"parameters": rc.PRIOR_ALIAS},
        },
    )
    cfg, warnings = rc.load_config(path)
    assert cfg["injection"]["parameters"] == cfg["prior"]
    assert cfg["injection"]["parameters"]["chirp_mass"]["low"] == 20.0
    # Drawing from the prior is exactly what a PP campaign wants, so no complaint.
    assert not any("differs from prior" in w for w in warnings)


def test_differing_injection_population_warns(tmp_path):
    _, warnings = rc.load_config()
    assert any("PP-plot" in w for w in warnings)


def test_bad_parameters_value_is_rejected(tmp_path):
    path = write(tmp_path, {"injection": {"parameters": "posterior"}})
    with pytest.raises(rc.ConfigError, match="must be an object"):
        rc.load_config(path)


# ---------------------------------------------------------------------------
# Round trips and the shipped examples
# ---------------------------------------------------------------------------


def test_dumped_default_config_round_trips(tmp_path):
    path = rc.dump_default_config(tmp_path / "out.json")
    cfg, _ = rc.load_config(path)
    reference, _ = rc.load_config()
    assert cfg == reference


@pytest.mark.parametrize("name", ["production_bbh", "pp_campaign"])
def test_shipped_example_configs_are_valid(name):
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    path = root / "examples" / "configs" / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    cfg, _ = rc.load_config(path)
    assert cfg["data"]["f_min"] > 0
    prior = rc.build_prior(cfg)
    assert prior.names == rc.PARAMETERS


def test_pp_campaign_actually_draws_from_the_prior():
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    path = root / "examples" / "configs" / "pp_campaign.json"
    if not path.exists():
        pytest.skip("example config not present")
    cfg, _ = rc.load_config(path)
    assert cfg["injection"]["parameters"] == cfg["prior"]


def test_fiducial_injection_covers_every_parameter():
    cfg, _ = rc.load_config()
    fid = rc.fiducial_injection(cfg)
    assert set(fid) == set(rc.PARAMETERS)
    assert fid["geocent_time"] == cfg["injection"]["geocent_time"]


def test_sampled_injection_covers_every_parameter():
    cfg, _ = rc.load_config()
    drawn = rc.sample_parameters(cfg, jax.random.PRNGKey(0))
    assert set(drawn) == set(rc.PARAMETERS)
    assert all(isinstance(v, float) for v in drawn.values())


def test_global_local_kwargs_are_accepted_by_the_dataclass():
    from jaxpe.sampler import GlobalLocalConfig

    cfg, _ = rc.load_config()
    GlobalLocalConfig(**rc.global_local_kwargs(cfg))


def test_unknown_sampler_field_is_rejected(tmp_path):
    path = write(tmp_path, {"sampler": {"n_chian": 8}})
    with pytest.raises(rc.ConfigError, match="GlobalLocalConfig"):
        rc.load_config(path)
