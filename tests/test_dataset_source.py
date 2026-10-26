"""Verify the public URL loader offline with small in-memory ZIP/ARFF fixtures."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import URLError
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from dataset_source import ARFF_FILENAME, PHISHING_DOWNLOAD_URL, SOURCE_METADATA, load_phishing_dataset
from train import main as train_main
from utils import MODEL_FILES, TARGET, evaluate_models, load_artifacts, validate_data


def zipped_arff(rows=600, invalid_feature=False, invalid_target=False, renamed_feature=False):
    """Create representative nominal attributes, UCI Result codes, and a ZIP."""
    rng = np.random.default_rng(71)
    frame = pd.DataFrame({
        name: rng.choice(SOURCE_METADATA["feature_values"][name], rows)
        for name in SOURCE_METADATA["feature_names"]
    })
    original_result = np.resize(np.array([-1, 1]), rows)
    attributes = ["@relation phishing_fixture"]
    for index, name in enumerate(SOURCE_METADATA["feature_names"]):
        allowed = list(SOURCE_METADATA["feature_values"][name])
        if invalid_feature and index == 0:
            allowed.append(99)
            frame.loc[0, name] = 99
        attribute_name = "unexpected_feature" if renamed_feature and index == 0 else name
        attributes.append(f"@attribute {attribute_name} {{{','.join(map(str, allowed))}}}")
    attributes.append("@attribute Result {-1,0,1}" if invalid_target else "@attribute Result {-1,1}")
    if invalid_target:
        original_result[0] = 0
    original = frame.assign(Result=original_result)
    arff_bytes = ("\n".join(attributes) + "\n@data\n" + original.to_csv(index=False, header=False)).encode()
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr(ARFF_FILENAME, arff_bytes)
    expected = frame.astype(np.int8)
    expected[TARGET] = (original_result == -1).astype(np.int8)
    return archive_bytes.getvalue(), arff_bytes, expected


class PublicDatasetSourceTests(unittest.TestCase):
    def test_url_zip_arff_mapping_and_checksums_are_read_in_memory(self):
        payload, arff_bytes, expected = zipped_arff()
        with patch("dataset_source.urlopen", return_value=BytesIO(payload)) as request, \
                patch.object(Path, "write_bytes", side_effect=AssertionError("The full dataset must stay in memory")), \
                patch.object(Path, "write_text", side_effect=AssertionError("The full dataset must stay in memory")):
            frame, source = load_phishing_dataset()
        self.assertEqual(request.call_args.args[0], PHISHING_DOWNLOAD_URL)
        self.assertGreater(request.call_args.kwargs["timeout"], 0)
        pd.testing.assert_frame_equal(frame, expected)
        self.assertEqual(frame.shape, (600, 31))
        self.assertEqual(set(frame[TARGET]), {0, 1})
        self.assertEqual(source["feature_names"], SOURCE_METADATA["feature_names"])
        self.assertEqual(source["download_url"], PHISHING_DOWNLOAD_URL)
        self.assertEqual(source["original_rows"], 600)
        self.assertEqual(source["archive_sha256"], sha256(payload).hexdigest())
        self.assertEqual(source["arff_sha256"], sha256(arff_bytes).hexdigest())
        self.assertEqual(source["csv_sha256"], sha256(frame.to_csv(index=False).encode()).hexdigest())

    def test_connection_failure_has_a_clear_download_error_without_local_fallback(self):
        with patch("dataset_source.urlopen", side_effect=URLError("offline")), \
                self.assertRaisesRegex(RuntimeError, "Could not download.*UCI"):
            load_phishing_dataset()

    def test_invalid_archive_or_changed_dataset_schema_is_rejected(self):
        cases = [
            (b"not a ZIP archive", "valid Training Dataset"),
            (zipped_arff(rows=499)[0], "minimum size"),
            (zipped_arff(invalid_feature=True)[0], "feature codes changed"),
            (zipped_arff(invalid_target=True)[0], "target labels changed"),
            (zipped_arff(renamed_feature=True)[0], "columns changed"),
        ]
        for payload, message in cases:
            with self.subTest(error=message), \
                    patch("dataset_source.urlopen", return_value=BytesIO(payload)), \
                    self.assertRaisesRegex(RuntimeError, message):
                load_phishing_dataset()

    def test_default_training_uses_url_data_and_saves_real_pickle_pipelines(self):
        payload, _, expected_source = zipped_arff()
        with TemporaryDirectory(prefix="url-trained-profile-") as temp:
            output = Path(temp) / "models"
            with patch("dataset_source.urlopen", return_value=BytesIO(payload)), \
                    patch("train.pd.read_csv", side_effect=AssertionError("URL training must not read a local full-dataset CSV")), \
                    redirect_stdout(StringIO()):
                train_main(["--output", str(output)])
            metadata, models = load_artifacts(output)
            self.assertEqual(metadata["dataset"]["rows"], len(expected_source))
            self.assertEqual(metadata["dataset"]["url"], SOURCE_METADATA["source_url"])
            self.assertEqual(metadata["dataset"]["csv_sha256"], sha256(expected_source.to_csv(index=False).encode()).hexdigest())
            test_data = pd.read_csv(output / "test_data.csv")
            features, target = validate_data(test_data, metadata)
            computed, _ = evaluate_models(models, features, target, metadata)
            stored = pd.read_csv(output / "metrics.csv")
            pd.testing.assert_frame_equal(computed, stored, check_exact=False, atol=1e-9, rtol=1e-9)
            for name, filename in MODEL_FILES.items():
                self.assertEqual(Path(filename).suffix, ".pkl")
                with (output / filename).open("rb") as stream:
                    saved = pickle.load(stream)
                self.assertIsInstance(saved, Pipeline)
                np.testing.assert_array_equal(saved.predict(features), models[name].predict(features))
            self.assertEqual(metadata["split"]["train_rows"] + len(test_data), len(expected_source))
            self.assertFalse((Path(temp) / "data").exists())


if __name__ == "__main__":
    unittest.main()
