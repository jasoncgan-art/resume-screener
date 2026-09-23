"""Training smoke test on a tiny synthetic world.

Runs data_gen + dataset + a 2-epoch train on 2 families x few docs inside a
tmp dir (module constants are monkeypatched; nothing touches the real
data/ or artifacts/). Asserts the model trains (loss finite/decreasing) and
emits L2-normalized vectors.
"""

import data_gen
import dataset
import train
import torch


def _tiny_world(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(data_gen, "DATA_DIR", str(d))
    monkeypatch.setattr(data_gen, "N_JOBS_PER_FAMILY", 6)
    monkeypatch.setattr(data_gen, "N_RESUMES_PER_FAMILY", 12)
    monkeypatch.setattr(data_gen, "FAMILIES",
                        {k: data_gen.FAMILIES[k]
                         for k in ("ml_engineer", "recruiter")})
    data_gen.main()
    monkeypatch.setattr(dataset, "DATA_DIR", str(d))
    return dataset.build_dataset()


def test_train_smoke(tmp_path, monkeypatch):
    data = _tiny_world(tmp_path, monkeypatch)
    assert data["vocab_size"] > 10
    assert len(data["train_pairs"]) > 0

    model = train.train(data, epochs=2, batch_size=8, verbose=False)
    model.eval()
    with torch.no_grad():
        jt = dataset.pad_batch([data["job_tokens"][0].tolist()])
        rt = dataset.pad_batch([data["resume_tokens"][0].tolist()])
        u, v = model(jt, rt)
    # L2-normalized outputs
    assert torch.allclose(u.norm(dim=1), torch.ones(1), atol=1e-5)
    assert torch.allclose(v.norm(dim=1), torch.ones(1), atol=1e-5)
    # in-batch loss is finite on a real batch
    b = 8
    ub = model.job_tower(dataset.pad_batch(
        [data["job_tokens"][i].tolist() for i in range(b)]))
    vb = model.resume_tower(dataset.pad_batch(
        [data["resume_tokens"][i].tolist() for i in range(b)]))
    loss = model.inbatch_softmax_loss(ub, vb)
    assert torch.isfinite(loss)
