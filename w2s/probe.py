import torch
from dataclasses import dataclass
from typing import Optional, Union

from w2s.sft_utils import literal
from w2s.logistic import Classifier
from w2s.topo import topofilter, topolabel

from simple_parsing import Serializable


@dataclass
class ProbeConfig(Serializable):
    def to_dict(self):
        irrelevant_fields = []
        return {k: v for k, v in vars(self).items() if k not in irrelevant_fields}

@dataclass
class KnnProbeConfig(ProbeConfig):
    k: int = 50

@dataclass
class LogisticProbeConfig(ProbeConfig):
    l2p: float = 1e-3

@dataclass
class TopoProbeConfig(ProbeConfig):
    k_cc: int = 50
    k_zeta: int = 50
    modified: bool = False


PROBE_CONFIGS = {
    "knn": KnnProbeConfig,
    "logreg": LogisticProbeConfig,
    "topo": TopoProbeConfig,
}


class Probe:
    def __init__(self, config: ProbeConfig):
        self.config = config

    def fit(self, acts, labels):
        raise NotImplementedError

    def predict(self, acts):
        raise NotImplementedError

    def filter(self, acts, labels, contamination):
        preds = self.predict(acts)
        disagree = (preds - labels).abs()
        # return indices for bottom (1-contamination) of examples
        return disagree.argsort(descending=True)[int(contamination * len(disagree)):]


class KnnProbe(Probe):
    def __init__(self, config: KnnProbeConfig):
        super().__init__(config)
        self.k = config.k

    def fit(self, acts, labels):
        self.acts = acts
        self.labels = labels

    def predict(self, acts):
        # compute cosine similarity
        dists = torch.cdist(acts, self.acts)
        # get top k
        topk = dists.topk(self.k, largest=False)
        # get labels
        labels = self.labels[topk.indices]
        # get majority vote
        pred = labels.float().mean(dim=-1)
        return pred


class LogisticProbe(Probe):
    def __init__(self, config: LogisticProbeConfig):
        super().__init__(config)
        self.l2p = config.l2p

    def fit(self, acts, labels):
        acts = acts.to(torch.float32)
        self.clf = Classifier(acts.shape[1], num_classes=1, device=acts.device)
        self.clf.fit(acts, labels, l2_penalty=self.l2p)

    def predict(self, acts):
        acts = acts.to(torch.float32)
        preds = torch.sigmoid(self.clf(acts))
        return preds


class TopoProbe(Probe):
    def __init__(self, config: TopoProbeConfig):
        super().__init__(config)
        self.k_cc = config.k_cc
        self.k_zeta = config.k_zeta
        self.modified = config.modified

    def fit(self, acts, labels):
        self.acts = acts
        self.labels = labels

    def predict(self, acts):
        return topolabel(self.acts, self.labels, acts, k_cc=self.k_cc, k_zeta=self.k_zeta)

    def filter(self, acts, labels, contamination):
        if not self.config.modified:
            return topofilter(acts, labels, contamination, k_cc=self.k_cc, k_zeta=self.k_zeta)
        else:
            return super().filter(acts, labels, contamination)


PROBES = {
    "knn": KnnProbe,
    "logreg": LogisticProbe,
    "topo": TopoProbe,
}
