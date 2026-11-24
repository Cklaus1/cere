"""O369 (DevShareCap gap): the dev-loop share's pre-publish load. `native_dev.py test --suite S` runs this module
alone, in a container under `cereshare.slice` (`--cgroup-parent`), before a GPU0 share run whose share would be
created in the ticket (`dev_share_plan` mode `create`: a new library or artifact). Its one load from a SHARE class
creates and publishes the kept O47 segment (`host_share_keep=1`), so the segment's tmpfs pages are first touched, and
charged, in cereshare.slice; the suite's container in ceredev.slice then attaches to it. The segment is marked homed
(`share_charge.HOME_XATTR`) so gpu_queue's sweep never copies it again. Not discovered by `test_*.py`.
"""
from __future__ import annotations

import os
import unittest

import oracle
import dev_share

# share_charge.HOME_XATTR and _common.SHARE_SLICE (scripts/live; this container mounts no host script path).
HOME_XATTR = "user.cere_memcg"
SHARE_SLICE = "cereshare.slice"
PREPUBLISH_CONTEXT = 4096


class Prepublish(unittest.TestCase):
    def test_publish_the_dev_loop_segment(self):
        share = oracle.dev_loop_share()
        self.assertIsNotNone(share, "no dev-loop share set or stamped: nothing to pre-publish")
        oracle.release_model(oracle.create_model(PREPUBLISH_CONTEXT))
        published = sorted(n for n in os.listdir(share.directory) if dev_share.SEGMENT_NAME.fullmatch(n))
        self.assertEqual(len(published), 1, f"published segments after the pre-publish load: {published}")
        os.setxattr(os.path.join(share.directory, published[0]), HOME_XATTR, SHARE_SLICE.encode())


if __name__ == "__main__":
    unittest.main()
