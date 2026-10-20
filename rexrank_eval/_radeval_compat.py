"""
radeval 2.2.1's vendored RadGraph/allennlp code imports `RadEval.metrics...` but the
installed package is `radeval`; that only resolves on case-insensitive imports, so
it fails on Windows. Alias every `RadEval.*` import to the already-loaded (or freshly
imported) `radeval.*` module - the same module object, so allennlp's class registries
aren't populated twice. Import this before anything from radeval.
"""

import importlib
import importlib.abc
import importlib.util
import sys


class _RadEvalAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == "RadEval" or name.startswith("RadEval."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("radeval" + spec.name[len("RadEval"):])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _RadEvalAlias) for f in sys.meta_path):
    sys.meta_path.insert(0, _RadEvalAlias())
