import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import download_hf_models


class EnsureModelTests(unittest.TestCase):
    model_id = "example/model"

    def args(self, cache_dir: str):
        return SimpleNamespace(
            model_id=[self.model_id],
            all_defaults=False,
            cache_dir=cache_dir,
            revision=None,
            allow_pattern=[],
            ignore_pattern=[],
            token=None,
            max_workers=1,
            force_download=False,
            prepare=False,
            ensure=True,
            clean_incomplete=False,
            clean_locks=False,
            verify_only=False,
            no_verify=False,
            etag_timeout=1.0,
            list_defaults=False,
        )

    def create_snapshot(self, cache_dir: Path) -> Path:
        model_dir = cache_dir / "models--example--model"
        snapshot = model_dir / "snapshots" / "revision"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        refs = model_dir / "refs"
        refs.mkdir()
        (refs / "main").write_text("revision", encoding="utf-8")
        return snapshot

    def test_ensure_reuses_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            self.create_snapshot(cache_dir)
            with (
                patch.object(download_hf_models, "parse_args", return_value=self.args(directory)),
                patch.object(download_hf_models, "snapshot_download") as download,
            ):
                self.assertEqual(download_hf_models.main(), 0)
                download.assert_not_called()

    def test_ensure_downloads_missing_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)

            def download_snapshot(**_kwargs):
                return str(self.create_snapshot(cache_dir))

            with (
                patch.object(download_hf_models, "parse_args", return_value=self.args(directory)),
                patch.object(download_hf_models, "snapshot_download", side_effect=download_snapshot) as download,
            ):
                self.assertEqual(download_hf_models.main(), 0)
                download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
