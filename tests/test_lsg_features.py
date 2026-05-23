import unittest
from unittest import mock


class LSGFeatureMatcherTest(unittest.TestCase):
    def test_extract_does_not_cache_dense_descriptors_on_viewpoint(self):
        from utils.lsg_features import LSGFeatureMatcher

        class FakeTensor:
            def detach(self):
                return self

            def clone(self):
                return self

            def to(self, *args, **kwargs):
                return self

            def float(self):
                return self

            def __getitem__(self, key):
                return self

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeTorch:
            Tensor = FakeTensor

            @staticmethod
            def no_grad():
                return FakeNoGrad()

        class FakeExtractor:
            def __call__(self, data):
                feats = {
                    "keypoints": FakeTensor(),
                    "descriptors": FakeTensor(),
                }
                dense_desc = FakeTensor()
                return feats, dense_desc

        class FakeViewpoint:
            original_image = FakeTensor()
            priors = {}

        matcher = LSGFeatureMatcher({}, "cpu")
        matcher.sp_extractor = FakeExtractor()
        matcher.lg_matcher = object()
        viewpoint = FakeViewpoint()

        with mock.patch.dict("sys.modules", {"torch": FakeTorch}):
            cached, reason = matcher.extract(viewpoint)

        self.assertEqual(reason, "extracted")
        self.assertIs(cached, viewpoint.lsg_local_features)
        self.assertIn("features", cached)
        self.assertNotIn("dense_desc", cached)


if __name__ == "__main__":
    unittest.main()
