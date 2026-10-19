import json
import pytest
from rank_documents import read_documents


def test_import_rejects_duplicate_candidate_ids(tmp_path):
    path = tmp_path / 'candidates.json'
    path.write_text(json.dumps([{'candidate_id': 'a', 'text': 'Python'},
                               {'candidate_id': 'a', 'text': 'SQL'}]))
    with pytest.raises(ValueError, match='unique'):
        read_documents(path, 'candidate_id')


def test_import_requires_text_not_unparsed_files(tmp_path):
    path = tmp_path / 'jobs.json'
    path.write_text(json.dumps([{'job_id': 'a', 'text': '!!!'}]))
    with pytest.raises(ValueError, match='untokenizable'):
        read_documents(path, 'job_id')
