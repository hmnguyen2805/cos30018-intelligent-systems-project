"""
Unit tests for the pure data-prep helpers in src.detection.training.data —
shared by train_binary.py, train_category.py, and evaluate.py. Runs on small
synthetic frames — no CICIDS2017 download needed. load_clean_dataframe
(the kagglehub-fetch-and-clean glue) isn't unit tested here.
"""
import numpy as np
import pandas as pd

from src.detection.training import data


def test_clean_dataframe_strips_whitespace_from_column_names():
    df = pd.DataFrame({" Label": ["BENIGN"], " Flow Duration": [1.0]})
    cleaned = data.clean_dataframe(df)
    assert list(cleaned.columns) == ["Label", "Flow Duration"]


def test_clean_dataframe_drops_rows_with_inf_or_nan():
    df = pd.DataFrame({
        "Label": ["BENIGN", "BENIGN", "BENIGN"],
        "Flow Duration": [1.0, np.inf, np.nan],
    })
    cleaned = data.clean_dataframe(df)
    assert len(cleaned) == 1


def test_clean_dataframe_drops_duplicate_rows():
    df = pd.DataFrame({
        "Label": ["BENIGN", "BENIGN"],
        "Flow Duration": [1.0, 1.0],
    })
    cleaned = data.clean_dataframe(df)
    assert len(cleaned) == 1


def test_binarize_labels_benign_is_zero():
    labels = pd.Series(["BENIGN", "BENIGN"])
    assert list(data.binarize_labels(labels)) == [0, 0]


def test_binarize_labels_attack_is_one():
    labels = pd.Series(["DoS Hulk", "PortScan"])
    assert list(data.binarize_labels(labels)) == [1, 1]


def test_binarize_labels_is_case_and_whitespace_insensitive():
    labels = pd.Series([" benign ", "Benign"])
    assert list(data.binarize_labels(labels)) == [0, 0]


def test_select_feature_names_excludes_label_and_non_numeric_columns():
    df = pd.DataFrame({
        "Label": ["BENIGN"],
        "Flow Duration": [1.0],
        "Some Text Column": ["x"],
    })
    assert data.select_feature_names(df, label_col="Label") == ["Flow Duration"]


def test_compute_feature_stats_medians_match_pandas_median():
    df = pd.DataFrame({"Flow Duration": [1.0, 2.0, 3.0, 4.0, 5.0]})
    medians, _ = data.compute_feature_stats(df, ["Flow Duration"])
    assert medians["Flow Duration"] == 3.0


def test_compute_feature_stats_mad_is_median_absolute_deviation_from_median():
    df = pd.DataFrame({"Flow Duration": [1.0, 2.0, 3.0, 4.0, 100.0]})
    # median=3.0; abs deviations = [2, 1, 0, 1, 97]; median of those = 1.0
    _, mad = data.compute_feature_stats(df, ["Flow Duration"])
    assert mad["Flow Duration"] == 1.0


def test_compute_feature_stats_covers_every_requested_feature():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    medians, mad = data.compute_feature_stats(df, ["a", "b"])
    assert set(medians) == {"a", "b"}
    assert set(mad) == {"a", "b"}


def test_compute_feature_stats_zero_mad_for_constant_column():
    df = pd.DataFrame({"Destination Port": [80.0, 80.0, 80.0]})
    medians, mad = data.compute_feature_stats(df, ["Destination Port"])
    assert medians["Destination Port"] == 80.0
    assert mad["Destination Port"] == 0.0


# --- split_train_test -------------------------------------------------------
# The ONE split train_binary.py, train_category.py, and evaluate.py all share.

def _make_df(n_benign=40, n_attack=10):
    rows = [{"Label": "BENIGN", "x": float(i)} for i in range(n_benign)]
    rows += [{"Label": "DDoS", "x": float(100 + i)} for i in range(n_attack)]
    return pd.DataFrame(rows)


def test_split_train_test_is_deterministic():
    df = _make_df()
    train_a, test_a = data.split_train_test(df)
    train_b, test_b = data.split_train_test(df)
    assert list(train_a["x"]) == list(train_b["x"])
    assert list(test_a["x"]) == list(test_b["x"])


def test_split_train_test_rows_are_disjoint_and_complete():
    df = _make_df()
    train_df, test_df = data.split_train_test(df)
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["x"]) & set(test_df["x"]) == set()


def test_split_train_test_is_stratified_on_the_binary_label():
    df = _make_df(n_benign=80, n_attack=20)
    train_df, test_df = data.split_train_test(df)
    train_attack_frac = (train_df["Label"] == "DDoS").mean()
    test_attack_frac = (test_df["Label"] == "DDoS").mean()
    assert abs(train_attack_frac - test_attack_frac) < 0.05


# --- split_validation ---------------------------------------------------------
# Carved out of the TRAIN split (never TEST) for things like the offline
# CATEGORY_CONFIDENCE_THRESHOLD sweep — a separate fixed random_state from
# split_train_test, so it's an independent split.

def test_split_validation_is_deterministic():
    df = _make_df()
    train_df, _ = data.split_train_test(df)
    sub_a, val_a = data.split_validation(train_df)
    sub_b, val_b = data.split_validation(train_df)
    assert list(sub_a["x"]) == list(sub_b["x"])
    assert list(val_a["x"]) == list(val_b["x"])


def test_split_validation_rows_are_disjoint_and_complete_within_train():
    df = _make_df()
    train_df, _ = data.split_train_test(df)
    sub_df, val_df = data.split_validation(train_df)
    assert len(sub_df) + len(val_df) == len(train_df)
    assert set(sub_df["x"]) & set(val_df["x"]) == set()


def test_split_validation_never_overlaps_the_test_split():
    df = _make_df()
    train_df, test_df = data.split_train_test(df)
    _, val_df = data.split_validation(train_df)
    assert set(val_df["x"]) & set(test_df["x"]) == set()


def test_split_validation_is_stratified_on_the_binary_label():
    df = _make_df(n_benign=80, n_attack=20)
    train_df, _ = data.split_train_test(df)
    sub_df, val_df = data.split_validation(train_df)
    sub_attack_frac = (sub_df["Label"] == "DDoS").mean()
    val_attack_frac = (val_df["Label"] == "DDoS").mean()
    assert abs(sub_attack_frac - val_attack_frac) < 0.05


# --- map_cicids_label_to_category --------------------------------------------
# The dataset's own labels, exactly as they appear in the cached CICIDS2017 CSVs
# (verified against a real pull): ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk',
# 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration',
# 'PortScan', 'SSH-Patator', 'Web Attack � Brute Force', 'Web Attack � Sql
# Injection', 'Web Attack � XSS'] — the Web Attack labels ship with a mangled
# separator character, which is why that mapping is prefix-based, not exact.

def test_benign_maps_to_none():
    assert data.map_cicids_label_to_category("BENIGN") is None


def test_dos_variants_all_map_to_dos():
    for raw in ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"]:
        assert data.map_cicids_label_to_category(raw) == "DoS"


def test_ddos_maps_to_ddos():
    assert data.map_cicids_label_to_category("DDoS") == "DDoS"


def test_portscan_maps_to_portscan():
    assert data.map_cicids_label_to_category("PortScan") == "PortScan"


def test_patator_variants_map_to_bruteforce():
    assert data.map_cicids_label_to_category("FTP-Patator") == "BruteForce"
    assert data.map_cicids_label_to_category("SSH-Patator") == "BruteForce"


def test_web_attack_variants_map_to_webattack_despite_mangled_separator():
    for raw in ["Web Attack � Brute Force", "Web Attack � XSS", "Web Attack � Sql Injection",
                "Web Attack - Brute Force"]:
        assert data.map_cicids_label_to_category(raw) == "WebAttack"


def test_bot_maps_to_botnet():
    assert data.map_cicids_label_to_category("Bot") == "Botnet"


def test_infiltration_maps_to_infiltration():
    assert data.map_cicids_label_to_category("Infiltration") == "Infiltration"


def test_heartbleed_maps_to_unknown():
    assert data.map_cicids_label_to_category("Heartbleed") == "Unknown"


def test_mapping_is_case_and_whitespace_insensitive():
    assert data.map_cicids_label_to_category("  ddos  ") == "DDoS"
    assert data.map_cicids_label_to_category("portscan") == "PortScan"


def test_unrecognized_label_maps_to_none():
    assert data.map_cicids_label_to_category("Some Future Attack Type") is None


def test_non_string_label_maps_to_none():
    assert data.map_cicids_label_to_category(None) is None
